import math
from collections import Counter
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from qibo import Circuit, gates
from qibo.backends import Backend
from qibo.config import raise_error
from qibo.transpiler._exceptions import ConnectivityError, PlacementError
from qibo.transpiler.abstract import Placer, PostLayout, Router
from qibo.transpiler.asserts import _find_unconnected_gate, assert_placement
from qibo.transpiler.decompositions import direction_dec
from qibo.transpiler.hardware import _connectivity_weights
from qibo.transpiler.layout import TranspileLayout
from qibo.transpiler.router import _find_connected_qubit

if TYPE_CHECKING:
    from qibo.transpiler.hardware import HardwareModel

# the gates that do not have a single-qubit gate error: barriers and measurements are not
# gates of the hardware, and the identity, Z and RZ gates are virtual
_ERRORLESS = (gates.Barrier, gates.I, gates.M, gates.RZ, gates.Z)


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
        two-qubit gates between :math:`a` and :math:`b`. The identity, Z and RZ gates are
        virtual, so they do not count. The cost is the weight of the operations of the
        circuit, as there is no SWAP gate. If a layout is found,
        :class:`qibo.transpiler.pipeline.Passes` does not need a router. If not, either
        because the circuit has no such layout or because a limit was reached before the
        search found any, the placer uses the method ``"dense"``, and the router has to move
        the qubits.

        The search drops every partial layout whose cost, added to a lower bound of the cost of
        the qubits that are left, is not lower than the cost of the best layout found, and it
        places first the qubits with more two-qubit gates, each on the nodes that cost less.
        So the layouts that it finds get better and better, and if the search ends before a
        limit is reached, the layout is the one with the lowest cost, *i.e.* the optimal one.
        The search is bounded by ``max_matches`` and ``call_limit``, which are counts and not
        times, so the placement is reproducible for a given version of :mod:`networkx`. If a
        limit is reached, the best layout found until then is taken, and a higher limit can
        find a better one. A directed connectivity has couplings that only work in one
        direction. By default, the placer does not see it, and the cost of a coupling is the
        mean of its directions, because :class:`qibo.transpiler.optimizer.GateDirection`
        reverses the gates afterwards, with single-qubit gates if the gate is not symmetric.
        With ``strict_direction``, the gates that need them act in the direction of a coupling.

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
        and measurements of the circuit on the qubits that take part in two-qubit gates, where
        the identity, Z and RZ gates are virtual and do not count. It is
        the expected weight of the operations if they were spread evenly over the region,
        which does not favor the regions with more connections just because they have more
        errors. The cost also sets the order of the qubits of the region, after their
        connections. This method always places the qubits, and the router may have to add SWAP
        gates. It chooses the region, not which qubit goes to which coupling.

    With both methods, the qubits without two-qubit gates take the remaining qubits of the
    hardware. The ones that have single-qubit gates or measurements take the remaining qubits
    where the total cost of their operations is the lowest, and the others take what is left,
    in the order of the connectivity. A hardware model without
    errors has no cost, so ``"dense"`` then does what :class:`qibo.transpiler.placer.Dense`
    does and ``"vf2"`` takes the first layout that VF2 finds. The errors that are missing are
    completed by the hardware model, which warns about them when the placer is created.

    Example:
        The qubits of a chain of two-qubit gates are placed on the quietest chain of the
        hardware, with either method. A triangle of gates cannot be placed on a line without
        SWAP gates, so the method ``"vf2"`` falls back to the method ``"dense"``, which puts
        the triangle on the quietest three qubits. The ``stop_reason`` of the placer says that
        the layout of the chain is the best one, and that the triangle has none.

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

            placer = NoiseAwarePlacer(hardware)
            vf2 = chain.copy()
            placer(vf2)
            print(vf2.wire_names, placer.stop_reason)

            dense = chain.copy()
            NoiseAwarePlacer(hardware, method="dense")(dense)
            print(dense.wire_names)

            triangle = Circuit(6)
            triangle.add(gates.CNOT(0, 1))
            triangle.add(gates.CNOT(1, 2))
            triangle.add(gates.CNOT(0, 2))
            placer(triangle)
            print(triangle.wire_names, placer.stop_reason)

        .. testoutput::

            [3, 4, 5, 0, 1, 2] optimal
            [3, 4, 5, 0, 1, 2]
            [4, 3, 5, 0, 1, 2] none exists

        The couplings of a directed connectivity only work in one direction. Here the
        cheapest layout puts every gate against its coupling, which
        :class:`qibo.transpiler.optimizer.GateDirection` would have to reverse with
        single-qubit gates, and ``strict_direction`` places them along the couplings.

        .. testcode::

            from qibo.transpiler.unroller import NativeGates

            couplings = [(0, 1), (1, 2), (2, 3)]
            directed = HardwareModel(
                tuple(range(4)),
                couplings,
                errors={"2q": dict(zip(couplings, [0.05, 0.05, 0.001]))},
                native_gates=NativeGates.default() | NativeGates.CNOT,
                symmetric={coupling: False for coupling in couplings},
            )

            circuit = Circuit(4)
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(1, 2))

            for strict in (False, True):
                placed = circuit.copy()
                NoiseAwarePlacer(directed, strict_direction=strict)(placed)
                print(placed.wire_names)

        .. testoutput::

            [3, 2, 1, 0]
            [1, 2, 3, 0]

    Args:
        hardware (:class:`qibo.transpiler.hardware.HardwareModel`): Hardware model. Its
            connectivity is the one of the placer, until
            :class:`qibo.transpiler.pipeline.Passes` sets another one, *e.g.* the one that
            ``on_qubits`` restricts, which must be a subgraph of it.
        method (str, optional): Method to place the qubits, ``"vf2"`` or ``"dense"``, see
            above. Defaults to ``"vf2"``.
        max_matches (int, optional): Only for the method ``"vf2"``. Number of layouts, each
            better than the previous one, after which the search stops and the best one is
            taken. Defaults to :math:`10000`.
        call_limit (int | tuple[int, int], optional): Only for the method ``"vf2"``. Number of
            feasibility checks of VF2 after which the search stops, and the best layout found
            until then is taken. It bounds the time of the circuits that have no layout, as
            :mod:`networkx` makes of the order of :math:`10^{5}` to :math:`10^{6}` checks per
            second. Large circuits on well-connected hardware, *e.g.* about thirty qubits on a
            grid, can need more checks to find their first layout, and then the placer falls
            back. If it is a pair, the first number is the limit until the first layout is
            found, and the second number is the number of checks that the search can make
            after that to improve it, *i.e.* a long search for a layout and a short one to
            improve it. Defaults to :math:`10^{6}`.
        strict_direction (bool, optional): Only for the method ``"vf2"``. If ``True``, the
            two-qubit gates that are not symmetric act in the direction of a coupling of the
            hardware, so they do not have to be reversed with single-qubit gates, and they
            cost the error of the coupling in that direction, and not the mean of its
            directions. These are the gates of the native gates of the hardware model that
            :class:`qibo.transpiler.optimizer.GateDirection` reverses with single-qubit
            gates, *i.e.* ``CNOT``, ``CH``, ``CSX``, ``CSXDG``, ``CY`` and ``ECR``. The
            symmetric gates, *e.g.* ``CZ``, are reversed by swapping their qubits, so they
            can act in any direction. A gate against the direction makes the layout not
            valid, so the placer falls back to the method ``"dense"`` if no layout respects
            the directions. It has no effect if the connectivity is not directed, or if its
            native gates have no gate to be reversed. Defaults to ``False``.
        keep_layout (bool, optional): If ``True``, the placer does not search when the layout
            that the circuit has already needs no SWAP gates, *i.e.* every two-qubit gate acts
            on connected qubits. That layout is the one that
            :class:`qibo.transpiler.optimizer.Preprocessing` sets, which is the first qubits
            of the connectivity if there is no initial layout. The circuit is then not moved
            to quieter qubits, which is what to want if the qubits were chosen on purpose, and
            the placer is faster. Defaults to ``False``.

    Attributes:
        stop_reason (str | None): Why the last call ended its search, which tells what the
            placer did. It is ``None`` before the first call. With the method ``"vf2"``, it
            is ``"optimal"`` if the search ended before a limit with a layout, which is then
            the best one, ``"limited"`` if a limit ended the search with a layout, which
            higher limits can improve, ``"none exists"`` if the search ended before a limit
            without a layout, *i.e.* the circuit needs SWAP gates, ``"none found"`` if a
            limit ended the search before the first layout, and ``"no two-qubit gates"`` if
            the circuit has none. In the last three cases the placer falls back to the method
            ``"dense"``. It is ``"kept"`` if ``keep_layout`` left the layout of the circuit,
            and ``"dense"`` for the method ``"dense"``. It belongs to the placer that was
            called, so the worker processes of
            :func:`qibo.transpiler.pipeline.transpile` have their own.

    Raises:
        ValueError: If ``method`` is not ``"vf2"`` or ``"dense"``, or if ``max_matches`` is
            not positive, or if ``call_limit`` is not positive or a pair of positive
            numbers, or, when the placer is called, if a gate of the circuit acts on more
            than two qubits.
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
        call_limit: int | tuple[int, int] = 1_000_000,
        keep_layout: bool = False,
        strict_direction: bool = False,
    ):
        if method not in ("vf2", "dense"):
            raise_error(
                ValueError, f"``method`` must be 'vf2' or 'dense', not {method!r}."
            )
        limits = call_limit if isinstance(call_limit, tuple) else (call_limit,)
        if max_matches < 1 or len(limits) > 2 or min(limits, default=0) < 1:
            raise_error(
                ValueError,
                "``max_matches`` must be positive, and ``call_limit`` must be positive or "
                + "a pair of positive numbers.",
            )
        super().__init__(hardware.connectivity)
        self.hardware = hardware
        self.method = method
        self.max_matches = max_matches
        self.call_limit = call_limit
        self.keep_layout = keep_layout
        self.strict_direction = strict_direction
        self.stop_reason = None
        self._weights = hardware.weighted_connectivity()
        # the native gates that the reversal of their direction needs single-qubit gates for
        self._directional = ()
        if strict_direction:
            self._directional = tuple(
                gate
                for gate in hardware.native_gates.two_qubit_gates
                if gate in direction_dec
            )

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Find the initial layout of the given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Not used by this placer, it is accepted to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        """
        assert_placement(circuit, self.connectivity)
        pairs = _find_gates_qubits_pairs(circuit)
        if (
            self.keep_layout
            and all(
                self.connectivity.has_edge(
                    circuit.wire_names[first], circuit.wire_names[second]
                )
                for first, second in pairs
            )
            and all(
                self._weights.has_edge(
                    circuit.wire_names[first], circuit.wire_names[second]
                )
                for first, second in self._directed_gates(circuit)
            )
        ):
            self.stop_reason = "kept"
            return

        layout = None
        if self.method == "dense":
            self.stop_reason = "dense"
        else:
            layout = self._search(circuit)

        if layout is None:
            # the region of the dense placement is kept, and the free qubits are placed again
            super().__call__(circuit, backend)
            layout = {
                qubit: circuit.wire_names[qubit] for pair in pairs for qubit in pair
            }
        self._place_free(circuit, layout)

    def _costs(self, circuit: Circuit, active: list[int]) -> tuple[dict, dict]:
        """Costs of the qubits and of the connections for the operations of the circuit."""
        weights = self._weights
        edge_weight = _connectivity_weights(self.connectivity, weights)

        qubits = set(active)
        singles = sum(
            len(gate.qubits) == 1 and gate.qubits[0] in qubits
            for gate in circuit.queue
            if not isinstance(gate, _ERRORLESS)
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

    def _directed_gates(self, circuit: Circuit) -> Counter:
        """Number of two-qubit gates that have to act in the direction of a coupling.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be placed.

        Returns:
            collections.Counter: Number of gates of each ordered pair of qubits, where the
            first qubit is the first one of the gate. It is empty unless ``strict_direction``
            is ``True``.
        """
        return Counter(
            tuple(gate.qubits)
            for gate in circuit.queue
            if isinstance(gate, self._directional) and len(gate.qubits) == 2
        )

    def _operations(self, circuit: Circuit) -> tuple[Counter, Counter]:
        """Number of single-qubit gates and of measurements of each qubit of the circuit.

        The virtual gates do not count, because they have no error.
        """
        singles = Counter(
            gate.qubits[0]
            for gate in circuit.queue
            if len(gate.qubits) == 1 and not isinstance(gate, _ERRORLESS)
        )
        measured = Counter(
            qubit
            for gate in circuit.queue
            if isinstance(gate, gates.M)
            for qubit in gate.qubits
        )
        return singles, measured

    def _place_free(self, circuit: Circuit, layout: dict) -> None:
        """Place the qubits that are not in the layout, *i.e.* the ones without two-qubit gates.

        The qubits that have single-qubit gates or measurements take the nodes that are left
        where the total cost of their operations is the lowest. The other qubits take the
        nodes that are left after that, in the order of the connectivity. The ``wire_names``
        of the circuit are set in place.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be placed.
            layout (dict): Node of each qubit that takes part in two-qubit gates.
        """
        singles, measured = self._operations(circuit)
        weights = self._weights.nodes
        used = set(layout.values())
        free = [node for node in self.connectivity.nodes if node not in used]
        wire_names = [layout.get(qubit) for qubit in range(circuit.nqubits)]

        busy = [
            qubit
            for qubit in range(circuit.nqubits)
            if qubit not in layout and singles[qubit] + measured[qubit]
        ]
        if busy:
            costs = np.array(
                [
                    [
                        sum(
                            count * weights[node][name]
                            for count, name in (
                                (singles[qubit], "gate"),
                                (measured[qubit], "readout"),
                            )
                            if count
                        )
                        for node in free
                    ]
                    for qubit in busy
                ]
            )
            # an infinite cost is made finite and larger than the sum of all the others
            finite = np.isfinite(costs)
            costs = np.where(finite, costs, costs[finite].sum() + 1.0)
            for qubit, column in zip(busy, linear_sum_assignment(costs)[1]):
                wire_names[qubit] = free[column]
            taken = {wire_names[qubit] for qubit in busy}
            free = [node for node in free if node not in taken]

        rest = iter(free)
        circuit.wire_names = [
            node if node is not None else next(rest) for node in wire_names
        ]

    def _search(
        self, circuit: Circuit, initial: dict | None = None, pinned: set | None = None
    ) -> dict | None:
        """Layout with the lowest cost that needs no SWAP gates, searched with VF2.

        The search drops the partial layouts that cannot beat the best layout so far, so it
        returns the best layout if it ends before a limit is reached. With
        ``strict_direction``, the layouts that put a gate against the direction of a coupling
        are not valid, and the gates cost the error of the coupling in their direction.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be placed.
            initial (dict, optional): Layout of the circuit, from each qubit to a node. It
                is the first best layout, and it is returned unless a layout found has a
                strictly lower cost. It is not used if two-qubit gates act on nodes that
                are not connected. Defaults to ``None``.
            pinned (set, optional): Nodes that the layouts cannot use. Defaults to
                ``None``.

        Returns:
            dict | None: Layout of the qubits that take part in two-qubit gates, from each
            qubit to a node, or ``None`` if the circuit has no two-qubit gates or no layout
            was found. The reason why the search ended is in ``stop_reason``.
        """
        pairs = Counter(tuple(pair) for pair in _find_gates_qubits_pairs(circuit))
        if not pairs:
            self.stop_reason = "no two-qubit gates"
            return None

        nodes = self._weights.nodes
        edge_weight = _connectivity_weights(self.connectivity, self._weights)
        singles, measured = self._operations(circuit)
        directed = self._directed_gates(circuit)
        directed_weight = {}
        if directed:
            directed_weight = _connectivity_weights(
                self.connectivity, self._weights, directed=True
            )
        # the gates that act the same in both directions
        counts = {}
        for pair, count in pairs.items():
            count -= directed[pair] + directed[pair[::-1]]
            if count:
                counts[frozenset(pair)] = count

        target = self.connectivity
        if pinned:
            target = target.subgraph(
                node for node in target.nodes if node not in pinned
            )
        node_cost = {
            qubit: {
                node: sum(
                    count * nodes[node][name]
                    for count, name in (
                        (singles[qubit], "gate"),
                        (measured[qubit], "readout"),
                    )
                    if count
                )
                for node in target.nodes
            }
            for qubit in {qubit for pair in pairs for qubit in pair}
        }
        matcher = _LayoutMatcher(
            target,
            _constrained_order(pairs),
            self.call_limit,
            node_cost,
            edge_weight,
            counts,
            directed,
            directed_weight,
        )

        layout = None
        if initial is not None:
            initial = {qubit: initial[qubit] for pair in pairs for qubit in pair}
            if (
                all(node in target for node in initial.values())
                and all(
                    (initial[first], initial[second]) in edge_weight
                    for first, second in pairs
                )
                and all(
                    (initial[first], initial[second]) in directed_weight
                    for first, second in directed
                )
            ):
                layout, matcher.best = initial, matcher.cost(initial)

        # the layouts that the matcher yields are better than the best one at that time
        for number, mapping in enumerate(matcher.subgraph_monomorphisms_iter(), 1):
            layout = {qubit: node for node, qubit in mapping.items()}
            matcher.best = matcher.cost(layout)
            if number == self.max_matches:
                matcher.limited = True
                break

        if matcher.limited:
            self.stop_reason = "limited" if layout is not None else "none found"
        else:
            self.stop_reason = "optimal" if layout is not None else "none exists"
        return layout


class NoiseAwarePostLayout(PostLayout):
    """Moves a routed circuit to the least noisy qubits of the hardware that can run it.

    The router makes a circuit executable by adding SWAP gates, but it does not know the
    noise of the hardware, and the placer chose the qubits before knowing how many SWAP gates
    the circuit needs. A SWAP gate is three entangling gates, so the couplings that a routed
    circuit uses weigh more, and the quietest place where it fits may be another one. This
    pass finds the places of the hardware where the very same routed circuit fits, *i.e.* the
    subgraph monomorphisms of the interaction graph of the circuit into the connectivity,
    with the VF2 algorithm of :mod:`networkx`, and moves the circuit to the one with the
    lowest cost

    .. math::
        c(\\phi) = \\sum_{q} \\left( n_{1}(q) \\, w_{\\phi(q)}^{\\text{gate}}
        + n_{m}(q) \\, w_{\\phi(q)}^{\\text{readout}} \\right)
        + \\sum_{(a, b)} n_{2}(a, b) \\, w_{\\phi(a) \\phi(b)} \\, ,

    where :math:`\\phi` places each qubit of the circuit on a qubit of the hardware,
    :math:`n_{1}(q)` and :math:`n_{m}(q)` are the numbers of single-qubit gates and of
    measurements of the qubit :math:`q`, :math:`n_{2}(a, b)` is the number of two-qubit gates
    between :math:`a` and :math:`b`, and the weights :math:`w` are the ones of
    :class:`qibo.transpiler.placer.NoiseAwarePlacer`, see
    :meth:`qibo.transpiler.hardware.HardwareModel.weighted_connectivity`. The identity, Z and
    RZ gates are virtual, so they do not count.

    Only the labels of the physical qubits change. The gates of the circuit, their order and
    the SWAP gates of the router do not, so the circuit needs the same number of gates, and
    it still respects the connectivity. The layout that the circuit already has is scored
    first and it is replaced only by a layout with a strictly lower cost, so the cost never
    increases, and the circuit is not changed if it cannot be improved. The new labels are
    also written in the layout that the router returned, which still describes how the
    qubits move, and the transpiled circuit stays equivalent to the original one.

    The cost is exact if the pass runs on the native circuit, because the SWAP gates are then
    decomposed and the entangling gates of each coupling are counted.
    :func:`qibo.transpiler.pipeline.generate_preset_passes` runs it at the end of the
    pipeline, so that the optimizers have already removed the gates that they can. The only
    exception is a directed connectivity, where the pass that sets the direction of the gates
    depends on the qubits, so it runs before it, and the optimizers that follow can still
    remove gates, which changes the counts a little.

    The qubits that have single-qubit gates or measurements but no two-qubit gates are not
    moved, because the interaction graph has nothing to place them with, and the layouts that
    are searched do not use their physical qubits. The qubits without operations keep their
    labels if they are free, and take the remaining qubits otherwise.

    The pass is skipped if the circuit was placed by a
    :class:`qibo.transpiler.placer.NoiseAwarePlacer` whose ``keep_layout`` left its layout,
    because the pass would move the circuit. The pipeline does not skip it after a placer that
    found the best layout, unless ``trust_placer`` is ``True``. The placer counts the gates
    before the unroller and the optimizers, which change them, and this pass counts the final
    gates, so the best layout of the placer can still be improved.

    The search is the one of :class:`qibo.transpiler.placer.NoiseAwarePlacer`, which drops the
    partial layouts that cannot beat the best one, and the layout that the circuit has is the
    first best layout, so a search that ends before a limit is reached finds the layout with
    the lowest cost. It is bounded by ``max_matches`` and ``call_limit``, which are counts and
    not times, so the result is reproducible for a given version of :mod:`networkx`. Their
    defaults are lower than the ones of the placer, because this pass runs for every
    transpiled circuit. If a limit is reached, the best layout found until then is taken.

    Example:
        A circuit that needs a SWAP gate is routed on the first qubits of a line, which are
        noisy, by the trivial placer. The pass moves it to the quietest three qubits that
        are connected as the routed circuit, with the same number of entangling gates.

        .. testcode::

            from qibo import Circuit, gates
            from qibo.transpiler import HardwareModel, NoiseAwarePostLayout
            from qibo.transpiler.pipeline import generate_preset_passes

            couplings = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
            errors = [0.05, 0.04, 0.02, 0.001, 0.003]
            hardware = HardwareModel(
                tuple(range(6)),
                couplings,
                errors={
                    "1q": {qubit: 0.001 for qubit in range(6)},
                    "2q": dict(zip(couplings, errors)),
                },
            )

            circuit = Circuit(3)
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(1, 2))
            circuit.add(gates.CNOT(0, 2))
            circuit.add(gates.M(0, 1, 2))

            for post_layout in (None, NoiseAwarePostLayout(hardware)):
                pipeline = generate_preset_passes(
                    connectivity=hardware.connectivity,
                    post_layout=post_layout,
                    seed=1,
                )
                transpiled, layout = pipeline(circuit)
                used = {transpiled.wire_names[q] for gate in transpiled.queue for q in gate.qubits}
                cz_gates = sum(isinstance(gate, gates.CZ) for gate in transpiled.queue)
                print(sorted(used), cz_gates)

        .. testoutput::

            [0, 1, 2] 6
            [3, 4, 5] 6

    Args:
        hardware (:class:`qibo.transpiler.hardware.HardwareModel`): Hardware model. Its
            connectivity is the one of the pass, until
            :class:`qibo.transpiler.pipeline.Passes` sets another one, *e.g.* the one that
            ``on_qubits`` restricts, which must be a subgraph of it.
        max_matches (int, optional): Number of layouts, each better than the previous one,
            after which the search stops and the best one is taken. Defaults to
            :math:`2500`.
        call_limit (int | tuple[int, int], optional): Number of feasibility checks of VF2
            after which the search stops, and the best layout found until then is taken. As
            :mod:`networkx` makes of the order of :math:`10^{5}` to :math:`10^{6}` checks per
            second, the default is a search of a fraction of a second. If it is a pair, the
            second number is the number of checks that the search can make to improve the
            layout that the circuit has, and the first one is only used if that layout needs
            SWAP gates, which is not the case after the routing. Defaults to :math:`50000`.
        trust_placer (bool, optional): If ``True``, the pass is also skipped after a
            :class:`qibo.transpiler.placer.NoiseAwarePlacer` of the same hardware model that
            found the best layout, which saves the search, but gives up the small improvements
            that the final gates allow. Defaults to ``False``.

    Raises:
        ValueError: If ``max_matches`` is not positive, or if ``call_limit`` is not positive
            or a pair of positive numbers.
        ConnectivityError: If the connectivity of the pass is not a subgraph of the
            connectivity of the hardware model.
    """

    def __init__(
        self,
        hardware: "HardwareModel",
        max_matches: int = 2_500,
        call_limit: int | tuple[int, int] = 50_000,
        trust_placer: bool = False,
    ):
        self._placer = NoiseAwarePlacer(
            hardware, max_matches=max_matches, call_limit=call_limit
        )
        self.hardware = hardware
        self.trust_placer = trust_placer
        self.connectivity = hardware.connectivity

    def __call__(self, circuit: Circuit, layout: TranspileLayout) -> TranspileLayout:
        """Move the routed circuit to the least noisy qubits that can run it.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Routed circuit. Its
                ``wire_names`` change in place if a better layout is found.
            layout (:class:`qibo.transpiler.layout.TranspileLayout`): Layout of the
                circuit, which the router returns.

        Returns:
            :class:`qibo.transpiler.layout.TranspileLayout`: Layout with the labels of the
            new physical qubits, or ``layout`` if the circuit is not moved.
        """
        assert_placement(circuit, self.connectivity)
        self._placer.connectivity = self.connectivity

        labels = list(circuit.wire_names)
        with_pairs = {
            qubit for pair in _find_gates_qubits_pairs(circuit) for qubit in pair
        }
        busy = {
            gate.qubits[0]
            for gate in circuit.queue
            if len(gate.qubits) == 1 and not isinstance(gate, _ERRORLESS)
        } | {
            qubit
            for gate in circuit.queue
            if isinstance(gate, gates.M)
            for qubit in gate.qubits
        }
        pinned = busy - with_pairs
        best = self._placer._search(
            circuit,
            initial=dict(enumerate(labels)),
            pinned={labels[qubit] for qubit in pinned},
        )
        if best is None or all(labels[qubit] == node for qubit, node in best.items()):
            return layout

        # the qubits without operations keep their label if no moved qubit takes it
        new = {qubit: best.get(qubit, label) for qubit, label in enumerate(labels)}
        taken = set(best.values()) | {labels[qubit] for qubit in pinned}
        displaced = []
        for qubit in range(circuit.nqubits):
            if qubit in best or qubit in pinned:
                continue
            if labels[qubit] in taken:
                displaced.append(qubit)
            else:
                taken.add(labels[qubit])
        free = (node for node in self.connectivity.nodes if node not in taken)
        for qubit in displaced:
            new[qubit] = next(free)

        circuit.wire_names = [new[qubit] for qubit in range(circuit.nqubits)]
        rename = {label: new[qubit] for qubit, label in enumerate(labels)}
        return TranspileLayout(
            {rename[label]: index for label, index in layout.items()},
            [rename[label] for label in layout.initial_layout],
            layout.nqubits_input,
        )

    def skip(self, placer: Placer | None) -> bool:
        """Tell if the pass is not to be run on the circuit that a placer placed.

        It is skipped if the placer is a :class:`qibo.transpiler.placer.NoiseAwarePlacer` that
        left the layout of the circuit because of its ``keep_layout``, since this pass would
        move the circuit. With ``trust_placer``, it is also skipped if the placer is one of the
        same hardware model that found the best layout. Then ``stop_reason`` is ``"skipped"``.

        Args:
            placer (:class:`qibo.transpiler.abstract.Placer`, optional): Placer that decided
                the layout of the circuit, or ``None`` if there is none.

        Returns:
            bool: ``True`` if the pass is not run.
        """
        skip = isinstance(placer, NoiseAwarePlacer) and (
            placer.stop_reason == "kept"
            or (
                self.trust_placer
                and placer.hardware is self.hardware
                and placer.stop_reason == "optimal"
            )
        )

        if skip:
            self._placer.stop_reason = "skipped"

        return skip

    @property
    def stop_reason(self) -> str | None:
        """Why the last search ended, with the values of the ``stop_reason`` of
        :class:`qibo.transpiler.placer.NoiseAwarePlacer`. Here ``"optimal"`` means that no
        layout has a lower cost than the layout that the circuit has, and ``"skipped"`` that
        there was no search, because :meth:`skip` told that the pass is not to be run.
        """
        return self._placer.stop_reason


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
        keep_layout (bool, optional): If ``True``, the placer does not search when the layout
            that the circuit already has needs no SWAP gates. Defaults to ``False``.

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
        keep_layout: bool = False,
    ):
        if trials < 1:
            raise_error(ValueError, f"``trials`` must be positive, but it is {trials}.")
        self.connectivity = connectivity
        self.routing_algorithm = routing_algorithm
        self.depth = depth
        self.trials = trials
        self.seed = seed
        self.keep_layout = keep_layout
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
        if (
            self.keep_layout
            and _find_unconnected_gate(self.connectivity, circuit) is None
        ):
            return
        self.routing_algorithm.connectivity = self.connectivity
        wire_names = circuit.wire_names.copy()
        # The assembled circuit is the same in all the attempts, which only change its layout,
        # so the router can reuse the blocks of its gates.
        assembled = self._assemble_circuit(circuit)

        best_swaps, best_layout = None, None
        for trial in range(self.trials):
            # the first attempt starts from the layout of the circuit
            start = (
                wire_names
                if trial == 0
                else [wire_names[i] for i in self._rng.permutation(len(wire_names))]
            )
            circuit.wire_names = start
            assembled.wire_names = start
            final_layout = self._routing_step(assembled)

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


class _LayoutMatcher(nx.algorithms.isomorphism.GraphMatcher):
    """VF2 matcher that searches the layout with the lowest cost.

    It differs from the matcher of :mod:`networkx` in three things. The search stops extending
    the layout after a number of feasibility checks, so that it is bounded and reproducible.
    A partial layout is dropped if its cost, added to a lower bound of the cost of the qubits
    that are left, is not lower than ``best``, which the caller updates, so the layouts that
    are found get better and better, and the search that ends before the limit has found the
    best one. And the nodes of each qubit are tried from the cheapest, so that the good layouts
    come first.

    Args:
        device (:class:`networkx.Graph`): Graph where the circuit is placed.
        pattern (:class:`networkx.Graph`): Interaction graph of the circuit. The search places
            its nodes in the order in which they were inserted.
        call_limit (int | tuple[int, int]): Number of feasibility checks allowed, or a pair
            with the number allowed until the first layout is found, and the number of checks
            allowed after that, which are not counted until then.
        node_cost (dict): Cost of each qubit on each node of the device,
            ``node_cost[qubit][node]``.
        edge_weight (dict): Weight of each coupling of the device, in both directions,
            ``edge_weight[node, other]``.
        counts (dict): Number of two-qubit gates of each pair of qubits that act the same in
            both directions, with the pair as a ``frozenset``.
        directed (dict, optional): Number of two-qubit gates that have to act in the direction
            of a coupling, with the ordered pair of their qubits as key. Defaults to ``None``.
        directed_weight (dict, optional): Weight of each coupling in each direction that it
            has, ``directed_weight[node, other]``. A gate of ``directed`` on nodes that have no
            coupling in its direction costs infinity. Defaults to ``None``.
    """

    def __init__(
        self,
        device: nx.Graph,
        pattern: nx.Graph,
        call_limit: int,
        node_cost: dict,
        edge_weight: dict,
        counts: dict,
        directed: dict | None = None,
        directed_weight: dict | None = None,
    ):
        super().__init__(device, pattern)
        self.calls = 0
        # whether a limit stopped the search, which the caller can also set
        self.limited = False
        self.call_limit = call_limit
        self._first, self._extra = (
            call_limit if isinstance(call_limit, tuple) else (call_limit, None)
        )
        # feasibility checks made when the first layout was found
        self._found = None
        self._best = math.inf
        self._node_cost = node_cost
        self._edge_weight = edge_weight
        self._counts = counts
        self._directed = {} if directed is None else directed
        self._directed_weight = {} if directed_weight is None else directed_weight
        # lower bounds of the cost of a qubit and of a gate, which are zero if they are
        # infinite, so that the bounds are always finite
        self._min_node = {}
        for qubit, costs in node_cost.items():
            lowest = min(costs.values())
            self._min_node[qubit] = lowest if math.isfinite(lowest) else 0.0
        lowest = min(
            [*edge_weight.values(), *self._directed_weight.values()], default=0.0
        )
        self._min_edge = lowest if math.isfinite(lowest) else 0.0
        size = pattern.number_of_nodes() + 1
        self._total_node = sum(self._min_node.values())
        self._total_gates = sum(counts.values()) + sum(self._directed.values())
        # cost, lower bound of the qubits, and gates of the partial layout of each size
        self._partial_cost = [0.0] * size
        self._partial_nodes = [0.0] * size
        self._partial_gates = [0.0] * size

    @property
    def best(self) -> float:
        """Cost of the best layout found, which the caller updates."""
        return self._best

    @best.setter
    def best(self, cost: float) -> None:
        if self._found is None:
            self._found = self.calls
        self._best = cost

    def candidate_pairs_iter(self):
        pairs = list(super().candidate_pairs_iter())
        # a stable sort: the nodes that cost the same stay in the order of VF2
        pairs.sort(key=lambda pair: self._increment(*pair)[0])
        return iter(pairs)

    def cost(self, layout: dict) -> float:
        """Cost of a layout, from each qubit to a node, of all the qubits of the pattern."""
        total = sum(self._node_cost[qubit][node] for qubit, node in layout.items())
        for pair, count in self._counts.items():
            first, second = pair
            total += count * self._edge_weight[layout[first], layout[second]]
        for (first, second), count in self._directed.items():
            edge = layout[first], layout[second]
            total += count * self._directed_weight.get(edge, math.inf)
        return total

    def semantic_feasibility(self, G1_node, G2_node) -> bool:
        size = len(self.core_1)
        increment, completed = self._increment(G1_node, G2_node)
        cost = self._partial_cost[size] + increment
        nodes = self._partial_nodes[size] + self._min_node[G2_node]
        done = self._partial_gates[size] + completed
        bound = (
            cost
            + (self._total_node - nodes)
            + self._min_edge * (self._total_gates - done)
        )
        if bound >= self.best:
            return False
        self._partial_cost[size + 1] = cost
        self._partial_nodes[size + 1] = nodes
        self._partial_gates[size + 1] = done
        return True

    def syntactic_feasibility(self, G1_node, G2_node) -> bool:
        self.calls += 1
        limit = self._first
        if self._found is not None and self._extra is not None:
            limit = self._found + self._extra
        if self.calls > limit:
            self.limited = True
            return False
        return super().syntactic_feasibility(G1_node, G2_node)

    def _increment(self, node, qubit) -> tuple[float, int]:
        """Cost that placing the qubit on the node adds to the partial layout, and the number of
        two-qubit gates that it completes. A node that is not coupled to the nodes of the
        qubits that the qubit has gates with costs infinity, and it is not feasible."""
        cost, completed = self._node_cost[qubit][node], 0
        for other in self.G2[qubit]:
            if other in self.core_2:
                other_node = self.core_2[other]
                count = self._counts.get(frozenset((qubit, other)), 0)
                if count:
                    cost += count * self._edge_weight.get((node, other_node), math.inf)
                    completed += count
                if self._directed:
                    for count, edge in (
                        (self._directed.get((qubit, other), 0), (node, other_node)),
                        (self._directed.get((other, qubit), 0), (other_node, node)),
                    ):
                        if count:
                            cost += count * self._directed_weight.get(edge, math.inf)
                            completed += count
        return cost, completed


def _constrained_order(pairs: Counter) -> nx.Graph:
    """Interaction graph of a circuit with its qubits inserted most constrained first.

    The search of a layout places the qubits in this order. It starts with the qubit that has
    the most two-qubit gates, and it goes on with the qubit that has the most gates with the
    qubits that are already placed, and then the most gates. The ties are decided by the lowest
    index. If the graph is not connected, it goes on with the next group of qubits.

    If all the pairs have the same number of gates, there is no pair with more gates to place
    first, and a search that starts inside a chain, which has most gates, has to close it with
    its last qubit on the neighbour of the first one, which takes it very long on a device with
    many qubits. So each group of qubits starts with the qubit that has the fewest two-qubit
    gates instead, which is the end of a chain, and the rest of the order is the same.

    Args:
        pairs (collections.Counter): Number of two-qubit gates of each pair of qubits.

    Returns:
        :class:`networkx.Graph`: Interaction graph, with the same edges as the pairs.
    """
    gates = nx.Graph()
    for (first, second), count in pairs.items():
        gates.add_edge(first, second, count=count)
    total = {
        qubit: sum(edge["count"] for edge in gates[qubit].values()) for qubit in gates
    }

    uniform = len(set(pairs.values())) == 1
    order = []
    while len(order) < len(gates):
        remaining = [qubit for qubit in gates if qubit not in order]
        placed = {
            qubit: sum(
                gates[qubit][other]["count"] for other in gates[qubit] if other in order
            )
            for qubit in remaining
        }
        # none of the qubits has gates with the placed ones when a group of qubits starts
        fewest = uniform and not any(placed.values())
        order.append(
            max(
                remaining,
                key=lambda qubit: (
                    placed[qubit],
                    -total[qubit] if fewest else total[qubit],
                    -qubit,
                ),
            )
        )

    ordered = nx.Graph()
    ordered.add_nodes_from(order)
    ordered.add_edges_from(gates.edges)
    return ordered


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
