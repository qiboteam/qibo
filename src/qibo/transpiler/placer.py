import networkx as nx
import numpy as np

from qibo import Circuit, gates
from qibo.backends import Backend
from qibo.config import raise_error
from qibo.transpiler._exceptions import ConnectivityError, PlacementError
from qibo.transpiler.abstract import Placer, Router
from qibo.transpiler.asserts import assert_placement
from qibo.transpiler.router import _find_connected_qubit


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
        best_region, best_edges = None, -1
        for start in nodes:
            region = [start]
            while len(region) < len(active):
                frontier = {
                    node for member in region for node in self.connectivity[member]
                } - set(region)
                if not frontier:
                    break
                # the first node, in the order of the connectivity, among the ties
                region.append(
                    max(
                        (node for node in nodes if node in frontier),
                        key=lambda node: sum(
                            member in region for member in self.connectivity[node]
                        ),
                    )
                )
            if len(region) == len(active):
                edges = self.connectivity.subgraph(region).number_of_edges()
                if edges > best_edges:
                    best_region, best_edges = region, edges

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
