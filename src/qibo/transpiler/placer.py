import math
from collections import Counter
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from qibo import Circuit, gates
from qibo.backends import Backend
from qibo.config import raise_error
from qibo.transpiler._exceptions import ConnectivityError, PlacementError
from qibo.transpiler.abstract import Placer, Router
from qibo.transpiler.asserts import assert_placement
from qibo.transpiler.router import _find_connected_qubit

if TYPE_CHECKING:
    from qibo.transpiler.hardware import HardwareModel


def _find_gates_qubits_pairs(circuit: Circuit) -> list[list[int]]:
    """Helper method for the placers in :mod:`qibo.transpiler.placer`.

    Translate circuit into a list of pairs of qubits to be used by the router and placer.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be translated.

    Returns:
        list[list[int]]: Sorted pairs of qubits targeted by two qubits gates. Measurements
        and barriers are skipped, since they do not need their qubits to be connected.
    """
    gates_qubits_pairs = []
    for gate in circuit.queue:
        if isinstance(gate, (gates.Barrier, gates.M)):
            continue
        if len(gate.qubits) == 2:
            gates_qubits_pairs.append(sorted(gate.qubits))
        elif len(gate.qubits) >= 3:
            raise_error(
                ValueError, "Gates targeting more than 2 qubits are not supported"
            )
    return gates_qubits_pairs


class Dense(Placer):
    """Places the qubits on the most connected region of the hardware.

    The :math:`n` qubits that take part in two-qubit gates are placed on a connected
    region of :math:`n` physical qubits with the largest number of connections among them.
    The qubits with more two-qubit gates are placed on the physical qubits with more
    connections in the region, and the other qubits take the remaining physical qubits
    in the order of the nodes of the connectivity.

    Args:
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            Defaults to ``None``.

    Example:
        The three qubits of a triangle of CNOT gates are placed on the triangle of the
        connectivity, which needs no SWAP gate, instead of on the first three nodes.

        .. testcode::

            import networkx as nx

            from qibo import Circuit, gates
            from qibo.transpiler import Dense

            connectivity = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (2, 4)])

            circuit = Circuit(5)
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(1, 2))
            circuit.add(gates.CNOT(0, 2))

            Dense(connectivity)(circuit)
            print(circuit.wire_names)

        .. testoutput::

            [2, 3, 4, 0, 1]
    """

    def __init__(self, connectivity: nx.Graph | None = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Find the initial layout of the given circuit on the most connected region.

        Method works in-place.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Not used by this placer, it is accepted to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.

        Raises:
            PlacementError: If no connected region of the hardware has as many qubits as
                the circuit has qubits in two-qubit gates.
        """
        assert_placement(circuit, self.connectivity)
        weights = {}
        for pair in _find_gates_qubits_pairs(circuit):
            for qubit in pair:
                weights[qubit] = weights.get(qubit, 0) + 1
        if not weights:
            return

        # qubits with more two-qubit gates first, then in the order of the circuit
        active = sorted(weights, key=lambda qubit: (-weights[qubit], qubit))
        nodes = list(self.connectivity.nodes)
        node_cost, edge_cost = self._costs(circuit, active)

        best_region, best_score = None, None
        for start in nodes:
            region = [start]
            while len(region) < len(active):
                frontier = {
                    node for member in region for node in self.connectivity[member]
                } - set(region)
                if not frontier:
                    break
                connections = {
                    node: sum(member in region for member in self.connectivity[node])
                    for node in nodes
                    if node in frontier
                }
                # among the nodes with more connections to the region, the one that costs
                # less, and the first in the order of the connectivity among the ties
                most = max(connections.values())
                region.append(
                    min(
                        (node for node in connections if connections[node] == most),
                        key=lambda node: self._cost(
                            [*region, node], node_cost, edge_cost
                        ),
                    )
                )
            if len(region) == len(active):
                score = (
                    self.connectivity.subgraph(region).number_of_edges(),
                    -self._cost(region, node_cost, edge_cost),
                )
                if best_score is None or score > best_score:
                    best_region, best_score = region, score

        if best_region is None:
            raise_error(
                PlacementError,
                f"No connected region of {len(active)} qubits in the connectivity.",
            )

        position = {node: index for index, node in enumerate(nodes)}
        members = set(best_region)
        region_order = sorted(
            best_region,
            key=lambda node: (
                -len(members.intersection(self.connectivity[node])),
                node_cost.get(node, 0),
                position[node],
            ),
        )
        placement = dict(zip(active, region_order))
        remaining_nodes = [node for node in nodes if node not in members]
        remaining_qubits = [
            qubit for qubit in range(circuit.nqubits) if qubit not in placement
        ]
        placement.update(zip(remaining_qubits, remaining_nodes))
        circuit.wire_names = [placement[qubit] for qubit in range(circuit.nqubits)]

    def _cost(self, region: list, node_cost: dict, edge_cost: dict) -> float:
        """Mean cost of the qubits and of the connections of a region.

        Args:
            region (list): Nodes of the region.
            node_cost (dict): Cost of each node, see ``_costs``.
            edge_cost (dict): Cost of each connection, in both directions.

        Returns:
            float: Mean cost of the nodes plus mean cost of the connections between them,
            or :math:`0` if there are no costs.
        """
        if not node_cost and not edge_cost:
            return 0
        edges = list(self.connectivity.subgraph(region).edges)
        return sum(node_cost[node] for node in region) / len(region) + sum(
            edge_cost[edge] for edge in edges
        ) / max(len(edges), 1)

    def _costs(self, circuit: Circuit, active: list[int]) -> tuple[dict, dict]:
        """Costs of the qubits and of the connections for the operations of the circuit,
        which are zero, so only the connections decide."""
        return {}, {}


class NoiseAwarePlacer(Dense):
    """Places the qubits on the least noisy qubits of the hardware.

    It uses the noise of a :class:`qibo.transpiler.hardware.HardwareModel` to choose where the
    qubits of the circuit go. The hardware model gives a weight :math:`w = -\\ln(1 - e)` to
    each qubit and to each coupling, where :math:`e` is the error of the operation, so
    :math:`w` is minus the logarithm of its fidelity, and the weights of independent
    operations add up as their errors multiply, see
    :meth:`qibo.transpiler.hardware.HardwareModel.weighted_connectivity`. The weights of the
    two directions of a coupling are averaged. There are two methods, set with ``method``:

    ``"vf2"`` (default)
        It looks for the layouts where every two-qubit gate of the circuit acts on coupled
        qubits, *i.e.* the subgraph monomorphisms of the interaction graph of the circuit into
        the connectivity, with the VF2 algorithm of :mod:`networkx`. It takes the layout with
        the lowest cost

        .. math::
            c(\\phi) = \\sum_{q} \\left( n_{1}(q) \\, w_{\\phi(q)}^{\\text{gate}}
            + n_{m}(q) \\, w_{\\phi(q)}^{\\text{readout}} \\right)
            + \\sum_{(a, b)} n_{2}(a, b) \\, w_{\\phi(a) \\phi(b)} \\, ,

        where :math:`\\phi` places each qubit of the circuit on a qubit of the hardware,
        :math:`n_{1}(q)` and :math:`n_{m}(q)` are the numbers of single-qubit gates and of
        measurements of the qubit :math:`q`, and :math:`n_{2}(a, b)` is the number of
        two-qubit gates between :math:`a` and :math:`b`. The cost is the weight of the
        operations of the circuit, as there is no SWAP gate. If a layout is found,
        :class:`qibo.transpiler.pipeline.Passes` does not need a router. If not, either
        because the circuit has no such layout or because a limit was reached before the
        search found any, the placer uses the method ``"dense"``, and the router has to move
        the qubits.

        The search is bounded by ``max_matches`` and ``call_limit``, which are counts and not
        times, so the placement is reproducible for a given version of :mod:`networkx`. The
        layouts are not found in the order of their cost, so a limit can leave out a better
        one: the lower the limits, the faster the placer and the farther the layout can be
        from the best one.

    ``"dense"``
        It works as :class:`qibo.transpiler.placer.Dense`, which has the details. A region
        of the hardware is grown from each qubit, adding the qubit with the most connections
        to it, and the region with the largest number of connections is taken, so that the
        circuit needs few SWAP gates. The cost decides among the regions with the same number
        of connections, and the ties when a region grows. The cost of a region :math:`R` is

        .. math::
            c(R) = \\frac{1}{\\vert R \\vert} \\sum_{q \\in R}
            \\left( n_{1} \\, w_{q}^{\\text{gate}} + n_{m} \\, w_{q}^{\\text{readout}} \\right)
            + \\frac{n_{2}}{\\vert E_{R} \\vert} \\sum_{(a, b) \\in E_{R}} w_{ab} \\, ,

        where :math:`E_{R}` are the connections of the region, and :math:`n_{1}`,
        :math:`n_{2}` and :math:`n_{m}` are the numbers of single-qubit gates, two-qubit gates
        and measurements of the circuit on the qubits that take part in two-qubit gates. It is
        the expected weight of the operations if they were spread evenly over the region,
        which does not favor the regions with more connections just because they have more
        errors. The cost also sets the order of the qubits of the region, after their
        connections. This method always places the qubits, and the router may have to add SWAP
        gates. It chooses the region, not which qubit goes to which coupling.

    With both methods, the qubits without two-qubit gates take the remaining qubits of the
    hardware, in the order of the connectivity. A hardware model without errors has no cost,
    so ``"dense"`` then does what :class:`qibo.transpiler.placer.Dense` does and ``"vf2"``
    takes the first layout that VF2 finds. The errors that are missing are completed by the
    hardware model, which warns about them when the placer is created.

    Example:
        The qubits of a chain of two-qubit gates are placed on the quietest chain of the
        hardware, with either method. A triangle of gates cannot be placed on a line without
        SWAP gates, so the method ``"vf2"`` falls back to the method ``"dense"``, which puts
        the triangle on the quietest three qubits.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import HardwareModel, NoiseAwarePlacer

            couplings = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
            errors = [0.05, 0.05, 0.02, 0.001, 0.002]
            hardware = HardwareModel(
                tuple(range(6)),
                couplings,
                errors={
                    "1q": {qubit: 0.001 * (1 + qubit / 10) for qubit in range(6)},
                    "2q": dict(zip(couplings, errors)),
                },
            )

            chain = Circuit(6)
            chain.add(gates.H(0))
            chain.add(gates.CNOT(0, 1))
            chain.add(gates.CNOT(0, 1))
            chain.add(gates.CNOT(1, 2))

            vf2 = chain.copy()
            NoiseAwarePlacer(hardware)(vf2)
            print(vf2.wire_names)

            dense = chain.copy()
            NoiseAwarePlacer(hardware, method="dense")(dense)
            print(dense.wire_names)

            triangle = Circuit(6)
            triangle.add(gates.CNOT(0, 1))
            triangle.add(gates.CNOT(1, 2))
            triangle.add(gates.CNOT(0, 2))
            NoiseAwarePlacer(hardware)(triangle)
            print(triangle.wire_names)

        .. testoutput::

            [3, 4, 5, 0, 1, 2]
            [3, 4, 5, 0, 1, 2]
            [4, 3, 5, 0, 1, 2]

    Args:
        hardware (:class:`qibo.transpiler.hardware.HardwareModel`): Hardware model. Its
            connectivity is the one of the placer, until
            :class:`qibo.transpiler.pipeline.Passes` sets another one, *e.g.* the one that
            ``on_qubits`` restricts, which must be a subgraph of it.
        method (str, optional): Method to place the qubits, ``"vf2"`` or ``"dense"``, see
            above. Defaults to ``"vf2"``.
        max_matches (int, optional): Only for the method ``"vf2"``. Number of layouts, in the
            order that VF2 finds them, after which the search stops and the best one is
            taken. Defaults to :math:`10000`.
        call_limit (int, optional): Only for the method ``"vf2"``. Number of feasibility
            checks of VF2 after which the search stops, and the best layout found until then
            is taken. It bounds the time of the circuits that have no layout, as
            :mod:`networkx` makes of the order of :math:`10^{6}` checks per second. Large
            circuits on well-connected hardware, *e.g.* about thirty qubits on a grid, can
            need more checks to find their first layout, and then the placer falls back.
            Defaults to :math:`10^{6}`.

    Raises:
        ValueError: If ``method`` is not ``"vf2"`` or ``"dense"``, or if ``max_matches`` or
            ``call_limit`` is not positive, or, when the placer is called, if a gate of the
            circuit acts on more than two qubits.
        PlacementError: If no connected region of the hardware has as many qubits as the
            circuit has qubits in two-qubit gates, and the layout is not found.
        ConnectivityError: If the connectivity of the placer is not a subgraph of the
            connectivity of the hardware model.
    """

    def __init__(
        self,
        hardware: "HardwareModel",
        method: str = "vf2",
        max_matches: int = 10_000,
        call_limit: int = 1_000_000,
    ):
        if method not in ("vf2", "dense"):
            raise_error(
                ValueError, f"``method`` must be 'vf2' or 'dense', not {method!r}."
            )
        if max_matches < 1 or call_limit < 1:
            raise_error(
                ValueError, "``max_matches`` and ``call_limit`` must be positive."
            )
        super().__init__(hardware.connectivity)
        self.hardware = hardware
        self.method = method
        self.max_matches = max_matches
        self.call_limit = call_limit
        self._weights = hardware.weighted_connectivity()

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Find the initial layout of the given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Not used by this placer, it is accepted to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        """
        if self.method == "dense":
            return super().__call__(circuit, backend)

        assert_placement(circuit, self.connectivity)
        pairs = Counter(tuple(pair) for pair in _find_gates_qubits_pairs(circuit))
        if not pairs:
            return super().__call__(circuit, backend)

        nodes = self._weights.nodes
        edge_weight = _connectivity_weights(self.connectivity, self._weights)
        singles = Counter(
            gate.qubits[0]
            for gate in circuit.queue
            if len(gate.qubits) == 1 and not isinstance(gate, (gates.Barrier, gates.M))
        )
        measured = Counter(
            qubit
            for gate in circuit.queue
            if isinstance(gate, gates.M)
            for qubit in gate.qubits
        )

        best, layout = math.inf, None
        matcher = _CallLimitedMatcher(
            self.connectivity, nx.Graph(list(pairs)), self.call_limit
        )
        for number, mapping in enumerate(matcher.subgraph_monomorphisms_iter(), 1):
            place = {qubit: node for node, qubit in mapping.items()}
            cost = sum(
                count * nodes[place[qubit]][name]
                for qubit in place
                for count, name in (
                    (singles[qubit], "gate"),
                    (measured[qubit], "readout"),
                )
                if count
            ) + sum(
                count * edge_weight[place[first], place[second]]
                for (first, second), count in pairs.items()
            )
            if cost < best:
                best, layout = cost, place
            if number == self.max_matches:
                break

        if layout is None:
            return super().__call__(circuit, backend)

        used = set(layout.values())
        free = (node for node in self.connectivity.nodes if node not in used)
        circuit.wire_names = [
            layout[qubit] if qubit in layout else next(free)
            for qubit in range(circuit.nqubits)
        ]

    def _costs(self, circuit: Circuit, active: list[int]) -> tuple[dict, dict]:
        """Costs of the qubits and of the connections for the operations of the circuit."""
        weights = self._weights
        edge_weight = _connectivity_weights(self.connectivity, weights)

        qubits = set(active)
        singles = sum(
            len(gate.qubits) == 1 and gate.qubits[0] in qubits
            for gate in circuit.queue
            if not isinstance(gate, (gates.Barrier, gates.M))
        )
        measured = sum(
            len(qubits.intersection(gate.qubits))
            for gate in circuit.queue
            if isinstance(gate, gates.M)
        )
        pairs = len(_find_gates_qubits_pairs(circuit))

        node_cost = {
            node: sum(
                count * weights.nodes[node][name]
                for count, name in ((singles, "gate"), (measured, "readout"))
                if count
            )
            for node in self.connectivity.nodes
        }
        edge_cost = {edge: pairs * weight for edge, weight in edge_weight.items()}
        return node_cost, edge_cost


class StarConnectivityPlacer(Placer):
    """Find an optimized qubit placement for the following connectivity::

             q
             |
        q -- q -- q
             |
             q

    Args:
        connectivity (:class:`networkx.Graph`, optional): Star connectivity graph.
            Defaults to ``None``.
    """

    def __init__(self, connectivity: nx.Graph | None = None):
        self.connectivity = connectivity
        self.middle_qubit = None

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): The original Qibo circuit to transform.
                Only single qubit gates, two qubits gates and barriers are supported by
                the router.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Not used by this placer, it is accepted to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        """
        assert_placement(circuit, self.connectivity)
        self._check_star_connectivity()

        middle_qubit_idx = circuit.wire_names.index(self.middle_qubit)
        wire_names = circuit.wire_names.copy()

        for i, gate in enumerate(circuit.queue):
            if isinstance(gate, gates.Barrier):
                continue
            if len(gate.qubits) > 2:
                raise_error(
                    PlacementError,
                    "Gates targeting more than 2 qubits are not supported, "
                    + "except for barriers.",
                )
            if len(gate.qubits) == 2 and middle_qubit_idx not in gate.qubits:
                new_middle = _find_connected_qubit(
                    gate.qubits,
                    circuit.queue[i + 1 :],
                    error=PlacementError,
                    mapping=list(range(circuit.nqubits)),
                )

                (
                    wire_names[middle_qubit_idx],
                    wire_names[new_middle],
                ) = (
                    wire_names[new_middle],
                    wire_names[middle_qubit_idx],
                )
                break

        circuit.wire_names = wire_names

    def _check_star_connectivity(self) -> None:
        """Check if the connectivity graph is a star graph."""
        if len(self.connectivity.nodes) != 5:
            raise_error(
                ConnectivityError,
                f"This connectivity graph is not a star graph. Length of nodes provided: {len(self.connectivity.nodes)} != 5.",
            )
        for node in self.connectivity.nodes:
            if self.connectivity.degree(node) == 4:
                self.middle_qubit = node
            elif self.connectivity.degree(node) != 1:
                raise_error(
                    ConnectivityError,
                    "This connectivity graph is not a star graph. There is a node with degree different from 1 and 4.",
                )


class Subgraph(Placer):
    """
    Subgraph isomorphism qubit placer.

    Since it is a :math:`NP`-complete problem, it can take exponential time for large circuits.
    This initialization method may fail for very short circuits.

    Args:
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            Defaults to ``None``.
    """

    def __init__(self, connectivity: nx.Graph | None = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Find the initial layout of the given circuit using subgraph isomorphism.
        Circuit must contain at least two two-qubit gates to implement subgraph placement.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Not used by this placer, it is accepted to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        """
        assert_placement(circuit, self.connectivity)
        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        if len(gates_qubits_pairs) < 2:
            raise_error(
                ValueError,
                "Circuit must contain at least two two-qubit gates "
                + "to implement subgraph placement.",
            )
        circuit_subgraph = nx.Graph()
        circuit_subgraph.add_nodes_from(list(range(circuit.nqubits)))
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            self.connectivity, circuit_subgraph
        )
        i = 0
        circuit_subgraph.add_edge(gates_qubits_pairs[i][0], gates_qubits_pairs[i][1])
        while matcher.subgraph_is_monomorphic():
            result = matcher
            i += 1
            circuit_subgraph.add_edge(
                gates_qubits_pairs[i][0], gates_qubits_pairs[i][1]
            )
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                self.connectivity, circuit_subgraph
            )
            if (
                self.connectivity.number_of_edges()
                == circuit_subgraph.number_of_edges()
                or i == len(gates_qubits_pairs) - 1
            ):
                break

        circuit.wire_names = sorted(result.mapping, key=lambda k: result.mapping[k])


class Random(Placer):
    """
    Random initialization with greedy policy, so that a maximum number of 2-qubit
    gates can be applied without introducing any SWAP gate.

    Args:
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            Defaults to ``None``.
        samples (int, optional): Number of random initializations to try. If :math:`0`,
            no random initialization is tried and the qubits are placed following the order
            of the nodes of ``connectivity``, i.e. the :math:`i`-th qubit is placed on the
            :math:`i`-th node. Defaults to :math:`100`.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
    """

    def __init__(
        self,
        connectivity: nx.Graph | None = None,
        samples: int = 100,
        seed: int | np.random.Generator | None = None,
    ):
        self.connectivity = connectivity
        self.samples = samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Find an initial layout of the given circuit using random greedy algorithm.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Not used by this placer, it is accepted to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        """
        assert_placement(circuit, self.connectivity)
        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        nodes = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())

        final_mapping = dict(zip(keys, range(nodes)))
        final_graph = nx.relabel_nodes(self.connectivity, final_mapping)
        final_cost = self._cost(final_graph, gates_qubits_pairs)
        for _ in range(self.samples):
            mapping = dict(zip(keys, self._rng.permutation(nodes)))
            graph = nx.relabel_nodes(self.connectivity, mapping)
            cost = self._cost(graph, gates_qubits_pairs)

            if cost == 0:
                final_layout = dict(zip(keys, list(mapping.values())))
                circuit.wire_names = sorted(final_layout, key=final_layout.get)
                return

            if cost < final_cost:
                final_mapping = mapping
                final_cost = cost

        final_layout = dict(zip(keys, list(final_mapping.values())))
        circuit.wire_names = sorted(final_layout, key=final_layout.get)

    def _cost(self, graph: nx.Graph, gates_qubits_pairs: list) -> int:
        """
        Compute the cost associated to an initial layout as the length of the reduced circuit.

        Args:
            graph (:class:`networkx.Graph`): Hardware connectivity.
            gates_qubits_pairs (list): Circuit representation.

        Returns:
            (int): length of the reduced circuit.
        """
        for allowed, gate in enumerate(gates_qubits_pairs):
            if gate not in graph.edges():
                return len(gates_qubits_pairs) - allowed - 1

        return 0


class ReverseTraversal(Placer):
    """
    Places qubits based on the algorithm proposed in Reference [1].

    Compatible with all the available :class:`qibo.transpiler.abstract.Router` classes.

    Args:
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
        routing_algorithm (:class:`qibo.transpiler.abstract.Router`): Router to be used.
        depth (int, optional): Number of two-qubit gates to be considered for routing.
            If ``None`` just one backward step will be implemented.
            If depth is greater than the number of two-qubit gates in the circuit,
            the circuit will be routed more than once.
            Example: on a circuit with four two-qubit gates :math:`A-B-C-D`
            using depth :math:`d = 6`, the routing will be performed
            on the circuit :math:`C-D-D-C-B-A`.
        trials (int, optional): Number of independent layout attempts. The first one
            starts from the layout of the circuit, and the others from random
            permutations of it. Each attempt finds a layout with the reverse traversal,
            and the one for which the router adds the fewest SWAP gates to the circuit
            is kept, the first one among ties. With one attempt, no routing of the
            circuit is needed to choose the layout. Defaults to :math:`1`.
        seed (int, optional): Seed of the random permutations of the attempts, which
            makes them reproducible. Defaults to ``None``.

    References:
        1. G. Li, Y. Ding, and Y. Xie,
        *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*.
        `arXiv:1809.02573 [cs.ET] <https://arxiv.org/abs/1809.02573>`_.
    """

    def __init__(
        self,
        routing_algorithm: Router,
        connectivity: nx.Graph | None = None,
        depth: int | None = None,
        trials: int = 1,
        seed: int | None = None,
    ):
        if trials < 1:
            raise_error(ValueError, f"``trials`` must be positive, but it is {trials}.")
        self.connectivity = connectivity
        self.routing_algorithm = routing_algorithm
        self.depth = depth
        self.trials = trials
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Find the initial layout of the given circuit using Reverse Traversal placement.

        Method works in-place.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Not used by this placer, it is accepted to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        """
        assert_placement(circuit, self.connectivity)
        self.routing_algorithm.connectivity = self.connectivity
        wire_names = circuit.wire_names.copy()

        best_swaps, best_layout = None, None
        for trial in range(self.trials):
            # the first attempt starts from the layout of the circuit
            start = (
                wire_names
                if trial == 0
                else [wire_names[i] for i in self._rng.permutation(len(wire_names))]
            )
            circuit.wire_names = start
            final_layout = self._routing_step(self._assemble_circuit(circuit))

            # The final layout of the reversed circuit is the initial layout of the circuit.
            layout = [start[final_layout[name]] for name in start]
            if self.trials == 1:
                best_layout = layout
                break

            candidate = circuit.copy()
            candidate.wire_names = layout
            routed_circuit, _ = self.routing_algorithm(candidate)
            swaps = sum(isinstance(gate, gates.SWAP) for gate in routed_circuit.queue)
            if best_swaps is None or swaps < best_swaps:
                best_swaps, best_layout = swaps, layout

        circuit.wire_names = best_layout

    def _assemble_circuit(self, circuit: Circuit) -> Circuit:
        """Assemble a single circuit to apply Reverse Traversal placement based on depth.

        Example: for a circuit with four two-qubit gates :math:`A-B-C-D`
        using depth :math:`d = 6`, the function will return the circuit :math:`C-D-D-C-B-A`.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be assembled.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): Assembled circuit to perform Reverse Traversal placement.
        """

        if self.depth is None:
            return circuit.invert()

        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        circuit_gates = len(gates_qubits_pairs)
        if circuit_gates == 0:
            raise_error(
                ValueError, "The circuit must contain at least a two-qubit gate."
            )
        repetitions, remainder = divmod(self.depth, circuit_gates)

        assembled_gates_qubits_pairs = []
        for _ in range(repetitions):
            assembled_gates_qubits_pairs += gates_qubits_pairs[:]
            gates_qubits_pairs.reverse()
        assembled_gates_qubits_pairs += gates_qubits_pairs[0:remainder]

        new_circuit = Circuit(circuit.nqubits, wire_names=circuit.wire_names)
        for qubits in assembled_gates_qubits_pairs:
            # As only the connectivity is important here we can replace everything with CZ gates
            new_circuit.add(gates.CZ(qubits[0], qubits[1]))

        return new_circuit.invert()

    def _routing_step(self, circuit: Circuit) -> dict:
        """Perform routing of the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be routed.

        Returns:
            dict: Final {logical: physical} qubit mapping.
        """
        _, final_mapping = self.routing_algorithm(circuit)

        return final_mapping


class _CallLimitedMatcher(nx.algorithms.isomorphism.GraphMatcher):
    """VF2 matcher that stops extending the mapping after a number of feasibility checks.

    Once the limit is reached, no candidate is feasible, so the search unwinds and ends,
    and the mappings found until then are kept.

    Args:
        G1 (:class:`networkx.Graph`): Graph where the subgraph is searched.
        G2 (:class:`networkx.Graph`): Subgraph to be found.
        call_limit (int): Number of feasibility checks allowed.
    """

    def __init__(self, G1: nx.Graph, G2: nx.Graph, call_limit: int):
        super().__init__(G1, G2)
        self.calls = 0
        self.call_limit = call_limit

    def syntactic_feasibility(self, G1_node, G2_node) -> bool:
        self.calls += 1
        return self.calls <= self.call_limit and super().syntactic_feasibility(
            G1_node, G2_node
        )


def _connectivity_weights(connectivity: nx.Graph, weights: nx.Graph) -> dict:
    """Weight of each connection of a connectivity, taken from the weights of a hardware.

    Args:
        connectivity (:class:`networkx.Graph`): Connectivity of a placer.
        weights (:class:`networkx.Graph`): Weighted connectivity of a hardware model, which
            can be directed.

    Returns:
        dict: Weight of each connection, in both directions. If the weights have the two
        directions of a connection, it is their mean.

    Raises:
        ConnectivityError: If the connectivity has qubits or connections that the weights do
            not have.
    """
    if not set(connectivity.nodes) <= set(weights.nodes) or any(
        not (weights.has_edge(*edge) or weights.has_edge(*edge[::-1]))
        for edge in connectivity.edges
    ):
        raise_error(
            ConnectivityError,
            "The connectivity must be a subgraph of the one of the hardware model.",
        )

    edge_weight = {}
    for edge in connectivity.edges:
        values = [
            weights.edges[pair]["weight"]
            for pair in (edge, edge[::-1])
            if weights.has_edge(*pair)
        ]
        edge_weight[edge] = edge_weight[edge[::-1]] = sum(values) / len(values)
    return edge_weight
