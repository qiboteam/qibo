import itertools
import math
import random
from collections.abc import Sequence
from copy import copy, deepcopy
from functools import cache
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from qibo import Circuit, gates
from qibo.config import raise_error
from qibo.gates.gates import is_nonzero_phase
from qibo.transpiler._exceptions import ConnectivityError
from qibo.transpiler.abstract import Router
from qibo.transpiler.asserts import assert_connectivity, assert_placement
from qibo.transpiler.blocks import _CZ_COSTS, Block, CircuitBlocks
from qibo.transpiler.hardware import _VIRTUAL, _connectivity_weights
from qibo.transpiler.optimizer import (
    FixedPoint,
    GateDirection,
    InverseCancellation,
    Optimize1qGatesDecomposition,
)
from qibo.transpiler.unroller import NativeGates, Unroller

if TYPE_CHECKING:
    from qibo.transpiler.hardware import HardwareModel

# Number of circuits whose blocks and DAG a router keeps, see ``Sabre._preprocessing``.
_MAX_TEMPLATES = 8

# Largest weight, *i.e.* minus the logarithm of the fidelity, that a coupling has in the
# noise heuristic of ``Sabre``. A coupling with an error of one has an infinite weight, which
# would make the distances infinite, so it only gets a very large one.
_MAX_WEIGHT = 100.0

# Weight added to the weight of every two-qubit gate in the noise heuristic of ``Sabre``. It
# is negligible for the errors of a device, and it makes the cost of a path grow with its
# length, so the router still approaches the qubits if the errors are zero or the same.
_HOP_WEIGHT = 1e-6

# Most CZ gates that a two-qubit block needs, which are the ones of a SWAP gate.
_MAX_CZ = _CZ_COSTS[gates.SWAP]

# Difference below which two costs of the noise heuristic of ``Sabre`` are the same, since
# they are sums of floats that can differ by the order of their terms.
_TOLERANCE = 1e-9


class CircuitMap:
    """Class that stores the circuit and physical-logical mapping during routing.

    Also implements the initial two-qubit block decompositions.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be routed.
        blocks (:class:`qibo.transpiler.blocks.CircuitBlocks`, optional): Circuit block representation.
            If ``None``, the blocks will be computed from the circuit. Defaults to ``None``.
        temp (bool, optional): If ``True``, only the qubit mappings are stored, which is
            used to evaluate the cost of a SWAP without modifying the circuit.
            Defaults to ``False``.
    """

    def __init__(
        self,
        circuit: Circuit | None = None,
        blocks: CircuitBlocks | None = None,
        temp: bool = False,
    ):
        self._p2l, self._l2p = [], []

        self._temporary = temp
        if self._temporary:
            return
        if circuit is None:
            raise_error(ValueError, "Circuit must be provided.")
        if blocks is not None:
            self.circuit_blocks = blocks
        else:
            self.circuit_blocks = CircuitBlocks(circuit, index_names=True)

        self.nqubits = circuit.nqubits
        self._routed_blocks = CircuitBlocks(
            Circuit(circuit.nqubits, wire_names=circuit.wire_names)
        )
        self._swaps = 0

        self.wire_names = circuit.wire_names.copy()
        self.physical_to_logical = list(range(self.nqubits))

    @property
    def physical_to_logical(self) -> list[int]:
        """Returns the physical to logical qubit mapping."""
        return self._p2l

    @property
    def logical_to_physical(self) -> list[int]:
        """Returns the logical to physical qubit mapping."""
        return self._l2p

    @physical_to_logical.setter
    def physical_to_logical(self, p2l_map: list[int]) -> None:
        """Sets the physical to logical qubit mapping and updates the logical to physical mapping.

        Args:
            p2l_map (list): Physical to logical mapping.
        """
        self._p2l = p2l_map.copy()
        self._l2p = [0] * len(self._p2l)
        for i, l in enumerate(self._p2l):
            self._l2p[l] = i

    @logical_to_physical.setter
    def logical_to_physical(self, l2p_map: list[int]) -> None:
        """Sets the logical to physical qubit mapping and updates the physical to logical mapping.

        Args:
            l2p_map (list): Logical to physical mapping.
        """
        self._l2p = l2p_map.copy()
        self._p2l = [0] * len(self._l2p)
        for i, p in enumerate(self._l2p):
            self._p2l[p] = i

    def _update_mappings_swap(self, logical_swap: tuple, physical_swap: tuple) -> None:
        """Updates the qubit mappings after applying a SWAP gate.

        Args:
            logical_swap (tuple[int]): The indices of the logical qubits to be swapped.
            physical_swap (tuple[int]): The indices of the corresponding physical qubits to be swapped.
        """
        self._p2l[physical_swap[0]], self._p2l[physical_swap[1]] = (
            logical_swap[1],
            logical_swap[0],
        )
        self._l2p[logical_swap[0]], self._l2p[logical_swap[1]] = (
            physical_swap[1],
            physical_swap[0],
        )

    def blocks_logical_qubits_pairs(self) -> list[tuple]:
        """Returns a list containing the logical qubit pairs of each block."""
        return [block.qubits for block in self.circuit_blocks()]

    def execute_block(self, block: Block) -> None:
        """Executes a block by removing it from the circuit representation
        and adding it to the routed circuit.

        Method works in-place.

        Args:
            block (:class:`qibo.transpiler.blocks.Block`): Block to be removed.
        """
        self._routed_blocks.add_block(block.on_qubits(self.get_physical_qubits(block)))
        self.circuit_blocks.remove_block(block)

    def routed_circuit(self, circuit_kwargs: dict | None = None) -> Circuit:
        """Returns the routed circuit.

        Args:
            circuit_kwargs (dict, optional): Original circuit init_kwargs.
                Defaults to ``None``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Routed circuit.
        """
        return self._routed_blocks.circuit(circuit_kwargs=circuit_kwargs)

    def final_layout(self) -> dict:
        """Returns the final {logical: physical} qubits mapping."""
        return {self.wire_names[i]: self._l2p[i] for i in range(self.nqubits)}

    def update(self, logical_swap: tuple, physical_order: tuple | None = None) -> None:
        """Updates the qubit mapping after applying a ``SWAP``.

        Adds the :class:`qibo.gates.gates.SWAP` gate to the routed blocks.
        Method works in-place.

        Args:
            logical_swap (tuple): Tuple containing the logical qubits to be swapped.
            physical_order (tuple, optional): The physical qubits of the swap, in the order
                of the qubits of the gate that is added, which decides how it is decomposed
                to native gates. If ``None``, it is the order of ``logical_swap``.
                Defaults to ``None``.
        """
        physical_swap = self.logical_pair_to_physical(logical_swap)
        if not self._temporary:
            order = physical_swap if physical_order is None else physical_order
            self._routed_blocks.add_block(
                Block(qubits=physical_swap, gates=[gates.SWAP(*order)])
            )
            self._swaps += 1

        self._update_mappings_swap(logical_swap, physical_swap)

    def undo(self) -> None:
        """Undo the last swap. Method works in-place."""
        last_swap_block = self._routed_blocks.return_last_block()
        physical_swap = last_swap_block.qubits
        logical_swap = self._p2l[physical_swap[0]], self._p2l[physical_swap[1]]

        self._routed_blocks.remove_block(last_swap_block)
        self._swaps -= 1

        self._update_mappings_swap(logical_swap, physical_swap)

    def get_physical_qubits(self, block: int | Block) -> tuple:
        """Returns the physical qubits where a block is acting on.

        Args:
            block (int or :class:`qibo.transpiler.blocks.Block`): Block to be analysed.

        Returns:
            tuple: Physical qubit numbers where a block is acting on.
        """
        if isinstance(block, int):
            block = self.circuit_blocks.search_by_index(block)

        return tuple(self._l2p[q] for q in block.qubits)

    def logical_pair_to_physical(self, logical_qubits: tuple) -> tuple:
        """Returns the physical qubits associated to the logical qubit pair.

        Args:
            logical_qubits (tuple): Logical qubit pair.

        Returns:
            tuple: Physical qubit numbers associated to the logical qubit pair.
        """
        return self._l2p[logical_qubits[0]], self._l2p[logical_qubits[1]]


class NoRouting(Router):
    """Leaves the circuit unchanged, and raises an error if it cannot run on the hardware.

    Args:
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            Defaults to ``None``.

    Example:
        The two-qubit gates of the circuit act on connected qubits, so the circuit is
        not changed and no qubit is moved.

        .. testcode::

            import networkx as nx

            from qibo import Circuit, gates
            from qibo.transpiler import NoRouting

            circuit = Circuit(3)
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(1, 2))

            routed_circuit, final_layout = NoRouting(nx.path_graph(3))(circuit)
            print(final_layout)

        .. testoutput::

            {0: 0, 1: 1, 2: 2}
    """

    def __init__(self, connectivity: nx.Graph | None = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit) -> tuple[Circuit, dict]:
        """Check that the circuit respects the connectivity.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be checked.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): The same circuit and the final
            {logical: physical} qubit mapping, which is the identity.

        Raises:
            ConnectivityError: If a gate acts on qubits that are not connected.
        """
        assert_placement(circuit, self.connectivity)
        assert_connectivity(self.connectivity, circuit)
        return circuit, {name: index for index, name in enumerate(circuit.wire_names)}


class Sabre(Router):
    """Routing algorithm proposed in Ref [1].

    Args:
        connectivity (:class:`networkx.Graph`): Hardware chip connectivity.
        lookahead (int, optional): Lookahead factor, how many dag layers will be considered
            in computing the cost. Defaults to :math:`2`.
        decay_lookahead (float, optional): Value in interval :math:`[0, 1]`.
            How the weight of the distance in the dag layers decays in computing the cost.
            Defaults to :math:`0.6`.
        delta (float, optional): Defines the number of SWAPs vs depth trade-off by deciding
            how the algorithm tends to select non-overlapping SWAPs.
            Defaults to :math:`10^{-3}`.
        swap_threshold (float, optional): Limits the number of added SWAPs in every routing iteration.
            This threshold is multiplied by the length of the longest path in the circuit connectivity.
            If the number of added SWAPs exceeds the threshold before a gate is routed,
            shortestpath routing is applied.
            Defaults to :math:`1.5`.
        seed (int, optional): Seed for the candidate random choice as tiebreaker.
            Defaults to ``None``.
        trials (int, optional): Number of independent routing attempts, which differ in
            the random choices among equally good candidates. The attempt with the
            lowest number of added SWAPs is returned, and the first one among ties.
            The attempts are reproducible if ``seed`` is given, and the first one
            does not depend on ``trials``. Defaults to :math:`1`.
        heuristic (str, optional): Cost that selects the SWAP gate to be added. It is the
            distance between the qubits of the gates in the front layer of the circuit,
            plus the terms below:

            - ``"basic"``: no other term, so ``lookahead``, ``decay_lookahead`` and
              ``delta`` are not used.
            - ``"lookahead"``: the distances of the gates in the next ``lookahead`` layers,
              weighted by ``decay_lookahead``. ``delta`` is not used.
            - ``"decay"``: the terms of ``"lookahead"``, with the distances of the qubits
              that were recently swapped increased by ``delta``, which favors
              non-overlapping SWAPs.
            - ``"noise"``: it needs ``hardware``, and it does not use the distances, but
              the infidelity that the circuit gets, see below. The cost of a candidate is
              the infidelity of its SWAP gate, plus the infidelity that the gates of the
              front layer need to be executed after it, plus the ones of the next
              ``lookahead`` layers, weighted by ``decay_lookahead``. ``delta`` is not used.
              The SWAP gates are added with the qubits in the order that costs less.

            Defaults to ``"decay"``.
        hardware (:class:`qibo.transpiler.hardware.HardwareModel`, optional): Hardware model
            whose errors weigh the qubits and the couplings, see
            :meth:`qibo.transpiler.hardware.HardwareModel.weighted_connectivity`. Its
            connectivity must contain the one of the router. The weights are stored with the
            qubits relabeled as the ones of the circuit, and the heuristic ``"noise"`` uses
            them. Defaults to ``None``.

    With the heuristic ``"noise"``, the weight of a gate is minus the logarithm of its
    fidelity, so the weights of independent gates add up, plus a negligible weight for each
    two-qubit gate that makes the paths with fewer couplings cheaper if the errors are the
    same. The error of the couplings in ``hardware`` is taken as the one of the native
    two-qubit gate, and the error of the qubits as the one of their native single-qubit gates.
    The cost of a SWAP gate on a coupling is the one of the native gates that it is
    decomposed into, see :func:`qibo.transpiler.router._native_counts`: the two-qubit gates
    on the coupling, and the single-qubit gates on each of its qubits. It is not three
    times the cost of a two-qubit gate, since it depends on the native gates of
    ``hardware``, and on the errors of the qubits. A block of the circuit costs the number of
    CZ gates that its gates have, at most three, each as many native two-qubit gates as the
    decomposition of a CZ gate has. The qubits of a SWAP gate can be in two orders, which
    decompose differently, *e.g.* with more single-qubit gates on a qubit or other, or
    against the direction of a directed coupling, and the router adds the cheapest one. The
    infidelity that a gate on the qubits :math:`a` and :math:`b` needs is the lowest one of
    moving :math:`a` to one end :math:`u` of a coupling and :math:`b` to the other end
    :math:`v`, with the SWAP gates of the couplings in between, and applying the gate on
    :math:`(u, v)`, see :func:`qibo.transpiler.router._noise_distances`. If a coupling is
    directed, only its direction has a weight, and if it has both, their mean is used. The
    blocks without two-qubit gates cost nothing. The best of the ``trials`` is the one with
    the lowest infidelity of all its gates, and not the one with the fewest SWAP gates. If a
    gate is stuck, the router moves its qubits along the path of the couplings with the
    lowest infidelity.

    Raises:
        ValueError: If ``trials`` is not positive, or ``heuristic`` is not ``"basic"``,
            ``"lookahead"``, ``"decay"`` or ``"noise"``, or it is ``"noise"`` and there is no
            ``hardware``.
        ConnectivityError: When the router is called, if its connectivity is not a subgraph
            of the connectivity of ``hardware``.

    References:
        1. G. Li, Y. Ding, and Y. Xie,
        *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*.
        `arXiv:1809.02573 [cs.ET] <https://arxiv.org/abs/1809.02573>`_.
    """

    def __init__(
        self,
        connectivity: nx.Graph | None = None,
        lookahead: int = 2,
        decay_lookahead: float = 0.6,
        delta: float = 0.001,
        swap_threshold: float = 1.5,
        seed: int | None = None,
        trials: int = 1,
        heuristic: str = "decay",
        hardware: "HardwareModel | None" = None,
    ):
        if trials < 1:
            raise_error(ValueError, f"``trials`` must be positive, but it is {trials}.")

        if heuristic not in ("basic", "lookahead", "decay", "noise"):
            raise_error(
                ValueError,
                f"Unknown heuristic {heuristic!r}. It must be one of "
                + "'basic', 'lookahead', 'decay' or 'noise'.",
            )

        if heuristic == "noise" and hardware is None:
            raise_error(ValueError, "The heuristic 'noise' needs ``hardware``.")

        self.connectivity = connectivity
        self.hardware = hardware
        self.heuristic = heuristic
        self.lookahead = lookahead
        self.decay = decay_lookahead
        self.delta = delta
        self.swap_threshold = swap_threshold
        self.trials = trials
        self._delta_register = None
        self._dist_matrix = None
        self._dag = None
        self._front_layer = None
        self.circuit_map = None
        self._memory_map = None
        self._final_measurements = None
        self._temp_added_swaps = []
        self._rng = random.Random(seed)
        self._hardware_weights = (
            None if hardware is None else hardware.weighted_connectivity()
        )
        self._weights = None
        # the data of the noise heuristic, see ``_preprocessing``
        self._block_costs = None
        self._gate_cost = None
        self._noise_cache = None
        self._noise_dist = None
        self._noise_layers = None
        self._swap_cost = None
        self._swap_ordered = None
        # the blocks and the DAG of the circuits that were routed, see ``_preprocessing``
        self._templates = {}

    def __call__(self, circuit: Circuit) -> tuple[Circuit, dict]:
        """Route the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be routed.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): Routed circuit and final {logical: physical} qubit mapping.
        """
        assert_placement(circuit, self.connectivity)
        global_phase = circuit.global_phase % (2 * math.pi)
        # The connectivity is relabeled during the preprocessing, so it is restored before
        # every trial and at the end, so that the router can be used again with the same
        # ``wire_names``.
        connectivity = self.connectivity
        best_trial = None
        try:
            for _ in range(self.trials):
                self.connectivity = connectivity
                self._preprocessing(circuit=circuit)
                longest_path = np.max(self._dist_matrix)

                while self._dag.number_of_nodes() != 0:
                    execute_block_list = self._check_execution()
                    if execute_block_list is not None:
                        self._execute_blocks(execute_block_list)
                    else:
                        self._find_new_mapping()

                    # If the number of added swaps is too high, the algorithm is stuck.
                    # Reset the circuit to the last saved state and make the nearest gate executable by manually adding SWAPs.
                    if (
                        len(self._temp_added_swaps) > self.swap_threshold * longest_path
                    ):  # threshold is arbitrary
                        while self._temp_added_swaps:
                            self._temp_added_swaps.pop()
                            self.circuit_map.undo()
                        self._temp_added_swaps = []
                        self._shortest_path_routing()

                circuit_kwargs = circuit.init_kwargs
                routed_circuit = self.circuit_map.routed_circuit(
                    circuit_kwargs=circuit_kwargs
                )
                if self._final_measurements is not None:
                    routed_circuit = self._append_final_measurements(
                        routed_circuit=routed_circuit
                    )
                if is_nonzero_phase(global_phase):
                    routed_circuit.add(gates.GlobalPhase(global_phase, trainable=False))

                # the first trial among the ones with the fewest swaps, or with the lowest
                # infidelity for the noise heuristic, is kept
                cost = (
                    self._routing_cost()
                    if self.heuristic == "noise"
                    else self.circuit_map._swaps
                )
                if best_trial is None or cost < best_trial[0]:
                    best_trial = (cost, routed_circuit, self.circuit_map)
        finally:
            self.connectivity = connectivity

        _, routed_circuit, self.circuit_map = best_trial
        return routed_circuit, self.circuit_map.final_layout()

    @property
    def added_swaps(self) -> int:
        """Returns the number of SWAP gates added to the circuit during routing."""
        return self.circuit_map._swaps

    def _preprocessing(self, circuit: Circuit) -> None:
        """The following objects will be initialised:
            - circuit: class to represent circuit and to perform {logical: physical} qubit mapping.
            - _final_measurements: measurement gates at the end of the circuit.
            - _weights: weights of the errors of ``hardware``, if there is one, on the
                connectivity with the nodes relabeled as the logical qubits.
            - _block_costs, _gate_cost, _swap_cost, _swap_ordered and _noise_dist: only for
                the noise heuristic, the number of CZ gates of each block, the infidelity of
                a CZ gate and of a SWAP gate on each coupling, the one of the SWAP gate whose
                first qubit is each of the two of the coupling, and the infidelity of each
                pair of qubits for a gate of one, two or three CZ gates, see
                :func:`qibo.transpiler.router._noise_costs` and
                :func:`qibo.transpiler.router._noise_distances`.
            - _dist_matrix: matrix reporting the shortest path length between all node pairs.
            - _dag: direct acyclic graph of the circuit based on commutativity.
            - _memory_map: list to remember previous SWAP moves.
            - _front_layer: list containing the blocks to be executed.
            - _delta_register: list containing the special weigh added to qubits
                to prevent overlapping swaps.

        The blocks of the circuit and its DAG do not depend on the layout, and routing consumes
        them, so a template of them is built once for the gates that the circuit has, and each
        call gets its own copy. This saves the work of the trials, and of the layout search
        that routes the same circuit with many layouts. The gates are identified by the objects,
        their qubits and their parameters, and the template keeps them, so that another circuit
        never gets the data of a gate that it does not have, such as the register name of a
        measurement. A circuit with parameters that cannot be hashed has no template.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be preprocessed.
        """

        mapping = {v: i for i, v in enumerate(circuit.wire_names)}
        if self._hardware_weights is not None:
            self._weights = _relabel_weights(
                self._hardware_weights, self.connectivity, mapping
            )

        if self.heuristic == "noise":
            self._set_noise_data(circuit.wire_names)

        self.connectivity = nx.relabel_nodes(self.connectivity, mapping)

        queue = tuple(circuit.queue)
        key = tuple((id(gate), gate.qubits, gate.parameters) for gate in queue)
        try:
            hash(key)
        except TypeError:
            key = None
        template = self._templates.get(key)

        if template is None:
            copied_circuit = circuit.copy(deep=True)
            # global phases act on no qubits, so they are not routed, see ``__call__``
            copied_circuit.queue[:] = [
                gate
                for gate in copied_circuit.queue
                if not isinstance(gate, gates.GlobalPhase)
            ]
            final_measurements = self._detach_final_measurements(copied_circuit)
            blocks = CircuitBlocks(copied_circuit, index_names=True)
            dag = _create_dag([block.qubits for block in blocks()])
            costs = [_block_cost(block) for block in blocks()]
            template = (queue, final_measurements, blocks, dag, costs)
            if key is not None:
                if len(self._templates) >= _MAX_TEMPLATES:
                    del self._templates[next(iter(self._templates))]
                self._templates[key] = template

        _, self._final_measurements, blocks, dag, self._block_costs = template
        # the blocks are not modified, but they are removed from the list while routing
        fresh_blocks = copy(blocks)
        fresh_blocks.block_list = blocks.block_list.copy()
        self.circuit_map = CircuitMap(circuit, blocks=fresh_blocks)
        _check_connected_components(self.circuit_map, self.connectivity)
        # the nodes are the logical qubits after the relabeling, so the matrix is indexed
        # by them, and not by the position of the nodes in the graph
        self._dist_matrix = nx.floyd_warshall_numpy(
            self.connectivity, nodelist=sorted(self.connectivity.nodes)
        )
        # The distance between qubits in different connected components is infinite. The
        # SWAP gates never change the components of the qubits, so a finite value that is
        # larger than the other distances is equivalent for the cost of the candidates,
        # and it keeps the longest path, which sets the swap threshold, finite.
        finite = np.isfinite(self._dist_matrix)
        self._dist_matrix[~finite] = self._dist_matrix[finite].max() + 1.0
        self._dag = dag.copy()
        self._memory_map = []
        self._update_dag_layers()
        self._update_front_layer()
        self._delta_register = [1.0 for _ in range(circuit.nqubits)]

    def _routing_cost(self) -> float:
        """Infidelity of the routed circuit, *i.e.* the sum of the infidelities of the SWAP
        gates that the router added, and of the blocks of the circuit, each as many CZ gates
        as it costs."""
        return sum(
            (
                # the blocks of the circuit have a name, and the SWAP gates of the router do not
                self._swap_cost[block.qubits]
                if block.name is None
                else cost * self._gate_cost[block.qubits]
            )
            for block in self.circuit_map._routed_blocks.block_list
            if (cost := _block_cost(block))
        )

    def _set_noise_data(self, wire_names: list) -> None:
        """Set the infidelities of the gates for the qubits of the circuit, for the noise
        heuristic.

        The infidelities only depend on the connectivity, so they are kept to route the
        circuits that are placed on the same connectivity, and they are ordered for each one.

        Args:
            wire_names (list): Nodes of the connectivity that the qubits of the circuit are
                placed on. The connectivity of the router is the one before the relabeling.
        """
        key = (
            frozenset(self.connectivity.nodes),
            frozenset(frozenset(edge) for edge in self.connectivity.edges),
        )
        if self._noise_cache is None or self._noise_cache[0] != key:
            nodes = list(self.connectivity.nodes)
            gate_cost, swap_ordered = _noise_costs(
                self._hardware_weights,
                self.connectivity,
                nodes,
                self.hardware.native_gates,
            )
            swap_cost = np.minimum(swap_ordered, swap_ordered.T)
            self._noise_cache = (
                key,
                nodes,
                gate_cost,
                swap_ordered,
                swap_cost,
                _noise_distances(gate_cost, swap_cost),
            )

        _, nodes, gate_cost, swap_ordered, swap_cost, distances = self._noise_cache
        position = {node: index for index, node in enumerate(nodes)}
        order = np.ix_(*[[position[name] for name in wire_names]] * 2)
        self._gate_cost = gate_cost[order]
        self._swap_ordered = swap_ordered[order]
        self._swap_cost = swap_cost[order]
        self._noise_dist = {cost: matrix[order] for cost, matrix in distances.items()}

    def _swap(self, logical_swap: tuple) -> None:
        """Add a SWAP gate, with the qubits in the order that costs less for the noise
        heuristic, and in the one of ``logical_swap`` for the others, or if the costs are the
        same.

        Args:
            logical_swap (tuple): Tuple containing the logical qubits to be swapped.
        """
        order = None
        if self.heuristic == "noise":
            first, second = self.circuit_map.logical_pair_to_physical(logical_swap)
            if self._swap_ordered[second, first] < (
                self._swap_ordered[first, second] - _TOLERANCE
            ):
                order = (second, first)
        self.circuit_map.update(logical_swap, order)

    def _detach_final_measurements(self, circuit: Circuit) -> list[gates.M] | None:
        """Detach measurement gates at the end of the circuit for separate handling."""
        final_measurements = []
        for gate in circuit.queue[::-1]:
            if isinstance(gate, gates.M):
                final_measurements.append(gate)
                circuit.queue.remove(gate)
            else:
                break
        if not final_measurements:
            return None
        return final_measurements[::-1]

    def _append_final_measurements(self, routed_circuit: Circuit) -> Circuit:
        """Appends final measurement gates on the correct qubits conserving
        the measurement register.

        Args:
            routed_circuit (:class:`qibo.models.circuit.Circuit`): Routed circuit without
                the final measurements.

        Returns:
            (:class:`qibo.models.circuit.Circuit`) Routed circuit.
        """
        for measurement in self._final_measurements:
            original_qubits = measurement.qubits
            routed_qubits = [
                self.circuit_map.logical_to_physical[qubit] for qubit in original_qubits
            ]
            routed_circuit.add(
                measurement.on_qubits(dict(zip(original_qubits, routed_qubits)))
            )
        return routed_circuit

    def _update_dag_layers(self) -> None:
        """Update dag layers and put them in topological order.

        Method works in-place.
        """
        self._noise_layers = []
        for layer, nodes in enumerate(nx.topological_generations(self._dag)):
            for node in nodes:
                self._dag.nodes[node]["layer"] = layer
            # the gates that the noise heuristic looks at, which are the ones that need routing
            if self.heuristic == "noise" and layer <= self.lookahead:
                self._noise_layers.append(
                    [
                        (self._dag.nodes[node]["qubits"], self._block_costs[node])
                        for node in nodes
                        if self._block_costs[node]
                    ]
                )

    def _update_front_layer(self) -> None:
        """Update the front layer of the dag.

        Method works in-place.
        """
        self._front_layer = self._get_dag_layer(0)

    def _get_dag_layer(self, n_layer: int, qubits: bool = False) -> list:
        """Return the :math:`n`-topological layer of the dag.

        Args:
            n_layer (int): Layer number.
            qubits (bool, optional): If ``True``, return the target qubits of the blocks in the layer.
                If ``False``, return the block numbers. Defaults to ``False``.

        Returns:
            (list): List of block numbers or target qubits.
        """
        if qubits:
            return [
                node[1]["qubits"]
                for node in self._dag.nodes(data=True)
                if node[1]["layer"] == n_layer
            ]

        return [node[0] for node in self._dag.nodes(data="layer") if node[1] == n_layer]

    def _find_new_mapping(self) -> None:
        """Find the new best mapping by adding one swap."""
        candidates_evaluation = {}

        self._memory_map.append(self.circuit_map.physical_to_logical.copy())
        for candidate in self._swap_candidates():
            candidates_evaluation[candidate] = self._compute_cost(candidate)

        best_cost = min(candidates_evaluation.values())
        # the costs of the noise heuristic are sums of floats, so the ones that only differ
        # by the order of their terms are the same
        tolerance = _TOLERANCE if self.heuristic == "noise" else 0.0
        best_candidates = [
            key
            for key, value in candidates_evaluation.items()
            if value <= best_cost + tolerance
        ]
        best_candidate = self._rng.choice(best_candidates)

        if self.heuristic == "decay":
            for qubit in self.circuit_map.logical_pair_to_physical(best_candidate):
                self._delta_register[qubit] += self.delta
        self._swap(best_candidate)
        self._temp_added_swaps.append(best_candidate)

    def _compute_cost(self, candidate: tuple) -> float:
        """Compute the cost associated to a possible SWAP candidate."""
        temporary_circuit = CircuitMap(temp=True)
        temporary_circuit.physical_to_logical = self.circuit_map.physical_to_logical
        temporary_circuit.update(candidate)

        if temporary_circuit.physical_to_logical in self._memory_map:
            return float("inf")

        if self.heuristic == "noise":
            first, second = self.circuit_map.logical_pair_to_physical(candidate)
            tot_infidelity = self._swap_cost[first, second]

            weight = 1.0
            for layer_gates in self._noise_layers:
                for lq_pair, cost in layer_gates:
                    qubits = temporary_circuit.logical_pair_to_physical(lq_pair)
                    tot_infidelity += (
                        weight * self._noise_dist[cost][qubits[0], qubits[1]]
                    )
                weight *= self.decay

            return float(tot_infidelity)

        tot_distance = 0.0
        weight = 1.0
        lookahead = 0 if self.heuristic == "basic" else self.lookahead
        for layer in range(lookahead + 1):
            layer_gates = self._get_dag_layer(layer, qubits=True)
            avg_layer_distance = 0.0
            for lq_pair in layer_gates:
                qubits = temporary_circuit.logical_pair_to_physical(lq_pair)
                avg_layer_distance += (
                    max(self._delta_register[i] for i in qubits)
                    * (self._dist_matrix[qubits[0], qubits[1]] - 1.0)
                    / len(layer_gates)
                )
            tot_distance += weight * avg_layer_distance
            weight *= self.decay

        return tot_distance

    def _swap_candidates(self) -> list[tuple]:
        """Returns a list of possible candidate SWAPs to be applied on logical qubits directly.

        The possible candidates are the ones sharing at least one qubit
        with a block in the front layer.

        Returns:
            (list): List of candidates.
        """
        candidates = []
        for block in self._front_layer:
            for qubit in self.circuit_map.get_physical_qubits(block):
                for connected in self.connectivity.neighbors(qubit):
                    candidate = tuple(
                        sorted(
                            (
                                self.circuit_map.physical_to_logical[qubit],
                                self.circuit_map.physical_to_logical[connected],
                            )
                        )
                    )
                    if candidate not in candidates:
                        candidates.append(candidate)

        return candidates

    def _check_execution(self) -> list | None:
        """Check if some blocks in the front layer can be executed in the current configuration.

        Returns:
            (list): Executable blocks if there are, ``None`` otherwise.
        """
        executable_blocks = []
        for block in self._front_layer:
            if (
                self.circuit_map.get_physical_qubits(block) in self.connectivity.edges
                or not self.circuit_map.circuit_blocks.search_by_index(block).entangled
            ):
                executable_blocks.append(block)

        if len(executable_blocks) == 0:
            return None

        return executable_blocks

    def _execute_blocks(self, blocklist: list) -> None:
        """Executes a list of blocks:
        -Remove the correspondent nodes from the dag and circuit representation.
        -Add the executed blocks to the routed circuit.
        -Update the dag layers and front layer.
        -Reset the mapping memory.

        Method works in-place.

        Args:
            blocklist (list): List of blocks.
        """
        for block_id in blocklist:
            block = self.circuit_map.circuit_blocks.search_by_index(block_id)
            self.circuit_map.execute_block(block)
            self._dag.remove_node(block_id)
        self._update_dag_layers()
        self._update_front_layer()
        self._memory_map = []
        self._delta_register = [1.0 for _ in self._delta_register]
        self._temp_added_swaps = []

    def _shortest_path_routing(self) -> None:
        """Route a gate in the front layer using the shortest path. This method is executed when the standard SABRE fails to find an optimized solution.

        Method works in-place.
        """
        min_distance = float("inf")
        shortest_path_qubits = None

        for block in self._front_layer:
            q1, q2 = self.circuit_map.get_physical_qubits(block)
            distance = self._dist_matrix[q1, q2]
            if self.heuristic == "noise":
                distance = self._noise_dist[self._block_costs[block]][q1, q2]

            if distance < min_distance:
                min_distance = distance
                shortest_path_qubits = [q1, q2]

        if self.heuristic == "noise":
            # the path of the couplings with the lowest infidelity
            shortest_path = nx.shortest_path(
                self.connectivity,
                *shortest_path_qubits,
                weight=lambda first, second, _: self._swap_cost[first, second],
            )
        else:
            shortest_path = nx.bidirectional_shortest_path(
                self.connectivity, shortest_path_qubits[0], shortest_path_qubits[1]
            )

        # move q1
        q1 = self.circuit_map.physical_to_logical[shortest_path[0]]
        for q2 in shortest_path[1:-1]:
            self._swap((q1, self.circuit_map.physical_to_logical[q2]))


class ShortestPaths(Router):
    """A class to perform initial qubit mapping and connectivity matching.

    Args:
        connectivity (:class:`networkx.Graph`): Chip connectivity.
        seed (int, optional): Seed for the random number generator.
            If ``None``, defaults to :math:`42`. Defaults to ``None``.
        hardware (:class:`qibo.transpiler.hardware.HardwareModel`, optional): Hardware model
            whose errors weigh the qubits and the couplings, see
            :meth:`qibo.transpiler.hardware.HardwareModel.weighted_connectivity`. Its
            connectivity must contain the one of the router. The weights are stored with the
            qubits relabeled as the ones of the circuit. Defaults to ``None``.

    Raises:
        ConnectivityError: When the router is called, if its connectivity is not a subgraph
            of the connectivity of ``hardware``.
    """

    def __init__(
        self,
        connectivity: nx.Graph | None = None,
        seed: int | None = None,
        hardware: "HardwareModel | None" = None,
    ):
        self.connectivity = connectivity
        self.hardware = hardware
        self._hardware_weights = (
            None if hardware is None else hardware.weighted_connectivity()
        )
        self._weights = None
        self._front_layer = None
        self.circuit_map = None
        self._dag = None
        self._final_measurements = None
        self._node_mapping_inv = None
        if seed is None:
            seed = 42
        self._rng = random.Random(seed)

    @property
    def added_swaps(self) -> int:
        """Returns the number of SWAP gates added to the circuit during routing."""
        return self.circuit_map._swaps

    def __call__(self, circuit: Circuit) -> tuple[Circuit, dict]:
        """Circuit connectivity matching.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be matched to hardware connectivity.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): Routed circuit and final {logical: physical} qubit mapping.
        """
        assert_placement(circuit, self.connectivity)
        global_phase = circuit.global_phase % (2 * math.pi)
        # The connectivity is relabeled during the preprocessing. It is restored at the
        # end so that the router can be used again with the same ``wire_names``.
        connectivity = self.connectivity
        try:
            self._preprocessing(circuit=circuit)
            while self._dag.number_of_nodes() != 0:
                execute_block_list = self._check_execution()
                if execute_block_list is not None:
                    self._execute_blocks(execute_block_list)
                else:
                    self._find_new_mapping()

            circuit_kwargs = circuit.init_kwargs
            routed_circuit = self.circuit_map.routed_circuit(
                circuit_kwargs=circuit_kwargs
            )
            if self._final_measurements is not None:
                routed_circuit = self._append_final_measurements(
                    routed_circuit=routed_circuit
                )
            if is_nonzero_phase(global_phase):
                routed_circuit.add(gates.GlobalPhase(global_phase, trainable=False))
        finally:
            self.connectivity = connectivity

        return routed_circuit, self.circuit_map.final_layout()

    def _find_new_mapping(self) -> None:
        """Find new qubit mapping. Mapping is found by looking for the shortest path.

        Method works in-place.
        """
        candidates_evaluation = []
        for candidate in self._candidates():
            cost = self._compute_cost(candidate)
            candidates_evaluation.append((candidate, cost))
        best_cost = min(candidate[1] for candidate in candidates_evaluation)
        best_candidates = [
            candidate[0]
            for candidate in candidates_evaluation
            if candidate[1] == best_cost
        ]
        best_candidate = self._rng.choice(best_candidates)
        self._add_swaps(best_candidate, self.circuit_map)

    def _candidates(self) -> list[tuple]:
        """Returns all possible shortest paths in a ``list`` that contains
        the new mapping and a second ``list`` containing the path meeting point.
        """
        target_qubits = self.circuit_map.get_physical_qubits(self._front_layer[0])
        path_list = list(
            nx.all_shortest_paths(
                self.connectivity, source=target_qubits[0], target=target_qubits[1]
            )
        )
        all_candidates = []
        for path in path_list:
            for meeting_point in range(len(path) - 1):
                all_candidates.append((path, meeting_point))

        return all_candidates

    @staticmethod
    def _add_swaps(candidate: tuple, circuitmap: CircuitMap) -> None:
        """Adds swaps to the circuit to move qubits.

        Method works in-place.

        Args:
            candidate (tuple): Contains path to move qubits and qubit meeting point in the path.
            circuitmap (CircuitMap): Representation of the circuit.
        """
        path = candidate[0]
        meeting_point = candidate[1]
        forward = path[0 : meeting_point + 1]
        backward = list(reversed(path[meeting_point + 1 :]))
        # the first qubit of each half of the path moves to the meeting point,
        # swapping at every step with the next qubit of the path
        for previous, current in itertools.pairwise(forward):
            circuitmap.update(
                (
                    circuitmap.physical_to_logical[current],
                    circuitmap.physical_to_logical[previous],
                )
            )
        for previous, current in itertools.pairwise(backward):
            circuitmap.update(
                (
                    circuitmap.physical_to_logical[current],
                    circuitmap.physical_to_logical[previous],
                )
            )

    def _compute_cost(self, candidate: tuple) -> int:
        """Greedy algorithm that decides which path to take and how qubits should be walked.

        The cost is computed as minus the number of successive gates that can be executed.

        Args:
            candidate (tuple): Contains path to move qubits and qubit meeting point in the path.

        Returns:
            int: Minus the number of successive gates that can be executed.
        """
        temporary_circuit = CircuitMap(
            circuit=Circuit(self.circuit_map.nqubits),
            blocks=deepcopy(self.circuit_map.circuit_blocks),
        )

        temporary_circuit.physical_to_logical = self.circuit_map.physical_to_logical
        self._add_swaps(candidate, temporary_circuit)
        temporary_dag = deepcopy(self._dag)
        successive_executed_gates = 0
        while temporary_dag.number_of_nodes() != 0:
            for layer, nodes in enumerate(nx.topological_generations(temporary_dag)):
                for node in nodes:
                    temporary_dag.nodes[node]["layer"] = layer
            temporary_front_layer = [
                node[0] for node in temporary_dag.nodes(data="layer") if node[1] == 0
            ]
            all_executed = True
            for block in temporary_front_layer:
                if (
                    temporary_circuit.get_physical_qubits(block)
                    in self.connectivity.edges
                    or not temporary_circuit.circuit_blocks.search_by_index(
                        block
                    ).entangled
                ):
                    successive_executed_gates += 1
                    temporary_circuit.execute_block(
                        temporary_circuit.circuit_blocks.search_by_index(block)
                    )
                    temporary_dag.remove_node(block)
                else:
                    all_executed = False
            if not all_executed:
                break

        return -successive_executed_gates

    def _check_execution(self) -> list | None:
        """Check if some blocks in the front layer can be executed in the current configuration.

        Returns:
            (list): Executable blocks if there are, ``None`` otherwise.
        """
        executable_blocks = []
        for block in self._front_layer:
            if (
                self.circuit_map.get_physical_qubits(block) in self.connectivity.edges
                or not self.circuit_map.circuit_blocks.search_by_index(block).entangled
            ):
                executable_blocks.append(block)
        if len(executable_blocks) == 0:
            return None

        return executable_blocks

    def _execute_blocks(self, blocklist: list) -> None:
        """Executes a list of blocks:
            -Remove the correspondent nodes from the dag and circuit representation.
            -Add the executed blocks to the routed circuit.
            -Update the dag layers and front layer.

        Method works in-place.

        Args:
            blocklist (list): List of blocks.
        """
        for block_id in blocklist:
            block = self.circuit_map.circuit_blocks.search_by_index(block_id)
            self.circuit_map.execute_block(block)
            self._dag.remove_node(block_id)
        self._update_front_layer()

    def _update_front_layer(self) -> None:
        """Updates the front layer of the dag.

        Method works in-place.
        """
        for layer, nodes in enumerate(nx.topological_generations(self._dag)):
            for node in nodes:
                self._dag.nodes[node]["layer"] = layer
        self._front_layer = [
            node[0] for node in self._dag.nodes(data="layer") if node[1] == 0
        ]

    def _preprocessing(self, circuit: Circuit) -> None:
        """The following objects will be initialised:
            - circuit: class to represent circuit and to perform {logical: physical} qubit mapping.
            - _final_measurements: measurement gates at the end of the circuit.
            - _weights: weights of the errors of ``hardware``, if there is one, on the
                connectivity with the nodes relabeled as the logical qubits.
            - _front_layer: list containing the blocks to be executed.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be preprocessed.
        """
        mapping = {v: i for i, v in enumerate(circuit.wire_names)}
        if self._hardware_weights is not None:
            self._weights = _relabel_weights(
                self._hardware_weights, self.connectivity, mapping
            )
        self.connectivity = nx.relabel_nodes(self.connectivity, mapping)
        copied_circuit = circuit.copy(deep=True)
        # global phases act on no qubits, so they are not routed, see ``__call__``
        copied_circuit.queue[:] = [
            gate
            for gate in copied_circuit.queue
            if not isinstance(gate, gates.GlobalPhase)
        ]
        self._final_measurements = self._detach_final_measurements(copied_circuit)
        self.circuit_map = CircuitMap(copied_circuit)
        _check_connected_components(self.circuit_map, self.connectivity)
        self._dag = _create_dag(self.circuit_map.blocks_logical_qubits_pairs())
        self._update_front_layer()

    def _detach_final_measurements(self, circuit: Circuit) -> list[gates.M] | None:
        """Detaches measurement gates at the end of the circuit for separate handling.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuits to be processed.

        Returns:
            (NoneType or list): List of measurements. If no measurements, returns ``None``.
        """
        final_measurements = []
        for gate in circuit.queue[::-1]:
            if isinstance(gate, gates.M):
                final_measurements.append(gate)
                circuit.queue.remove(gate)
            else:
                break
        if not final_measurements:
            return None

        return final_measurements[::-1]

    def _append_final_measurements(self, routed_circuit: Circuit) -> Circuit:
        """Appends the final measurement gates on the correct qubits
        conserving the measurement register."""
        for measurement in self._final_measurements:
            original_qubits = measurement.qubits
            routed_qubits = [
                self.circuit_map.logical_to_physical[qubit] for qubit in original_qubits
            ]
            routed_circuit.add(
                measurement.on_qubits(dict(zip(original_qubits, routed_qubits)))
            )

        return routed_circuit


class StarConnectivityRouter(Router):
    """Transforms an arbitrary circuit to one that can be executed on hardware.

    This transpiler produces a circuit that respects the following connectivity
    by adding SWAP gates when needed::

             q
             |
        q -- q -- q
             |
             q

    Args:
        connectivity (:class:`networkx.Graph`): Star connectivity graph.
    """

    def __init__(self, connectivity: nx.Graph | None = None):
        self.connectivity = connectivity
        self.middle_qubit = None

    def __call__(self, circuit: Circuit) -> tuple[Circuit, dict]:
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): The original Qibo circuit to transform.
                Only single qubit gates, two qubits gates and barriers are supported by
                the router.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): Routed circuit and final {logical: physical} qubit mapping.
        """
        self._check_star_connectivity()
        assert_placement(circuit, self.connectivity)

        middle_qubit_idx = circuit.wire_names.index(self.middle_qubit)
        nqubits = circuit.nqubits
        new = Circuit(**circuit.init_kwargs)
        l2p = list(range(nqubits))

        for i, gate in enumerate(circuit.queue):
            routed_qubits = [l2p[q] for q in gate.qubits]

            if isinstance(gate, gates.M):
                new_gate = gates.M(*routed_qubits, **gate.init_kwargs)
                new_gate.result = gate.result
                new.add(new_gate)
                continue

            if isinstance(gate, gates.Barrier):
                new.add(gate.on_qubits(dict(zip(gate.qubits, routed_qubits))))
                continue

            if len(routed_qubits) > 2:
                raise_error(
                    ConnectivityError,
                    "Gates targeting more than two qubits are not supported, "
                    + "except for barriers.",
                )

            if len(routed_qubits) == 2 and middle_qubit_idx not in routed_qubits:
                # find which qubit should be moved
                new_middle = _find_connected_qubit(
                    routed_qubits,
                    circuit.queue[i + 1 :],
                    error=ConnectivityError,
                    mapping=l2p,
                )

                new.add(gates.SWAP(new_middle, middle_qubit_idx))
                idx1, idx2 = l2p.index(middle_qubit_idx), l2p.index(new_middle)
                l2p[idx1], l2p[idx2] = l2p[idx2], l2p[idx1]

            routed_qubits = [l2p[q] for q in gate.qubits]

            # add gate to the hardware circuit
            if isinstance(gate, gates.Unitary):
                # gates.Unitary requires matrix as first argument
                matrix = gate.init_args[0]
                new.add(gate.__class__(matrix, *routed_qubits, **gate.init_kwargs))
            else:
                new.add(gate.__class__(*routed_qubits, **gate.init_kwargs))
        return new, {circuit.wire_names[i]: l2p[i] for i in range(nqubits)}

    def _check_star_connectivity(self) -> None:
        """Check if the connectivity graph is a star graph."""
        if len(self.connectivity.nodes) != 5:  # pragma: no cover
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


def _block_cost(block: Block) -> int:
    """Helper method for :class:`qibo.transpiler.router.Sabre`.

    Number of CZ gates that the two-qubit gates of a block need, at most three, because any
    two-qubit block can be rewritten with three. A SWAP gate costs three and an iSWAP gate
    two, like in :meth:`qibo.transpiler.blocks.Block.kak_decompose`, and any other gate one.

    Args:
        block (:class:`qibo.transpiler.blocks.Block`): Block of gates.

    Returns:
        int: Number of CZ gates, which is zero if the block has no two-qubit gate.
    """
    return min(
        _MAX_CZ,
        sum(
            _CZ_COSTS.get(type(gate), 1)
            for gate in block.gates
            if len(gate.qubits) == 2 and not isinstance(gate, gates.Barrier)
        ),
    )


def _check_connected_components(
    circuit_map: CircuitMap, connectivity: nx.Graph
) -> None:
    """Check that the two-qubit gates act on qubits of the same connected component.

    The SWAP gates move the qubits inside their connected component, so a gate between
    qubits of different components can never be routed, and the routers would not stop.

    Args:
        circuit_map (:class:`qibo.transpiler.router.CircuitMap`): Circuit with the initial
            layout.
        connectivity (:class:`networkx.Graph`): Hardware connectivity, with the nodes
            relabeled as the physical qubits of ``circuit_map``.

    Raises:
        ConnectivityError: If a two-qubit gate acts on qubits of different connected
            components of the connectivity.
    """
    component = {
        node: index
        for index, nodes in enumerate(nx.connected_components(connectivity))
        for node in nodes
    }
    for block in circuit_map.circuit_blocks():
        if block.entangled:
            first, second = circuit_map.get_physical_qubits(block)
            if component[first] != component[second]:
                raise_error(
                    ConnectivityError,
                    f"A gate acts on the qubits {block.qubits} of the circuit, which are "
                    + "in different connected components of the connectivity graph, so "
                    + "it cannot be routed.",
                )


def _create_dag(gates_qubits_pairs: list) -> nx.DiGraph:
    """Helper method for :class:`qibo.transpiler.router.Sabre`.

    Create direct acyclic graph (dag) of the circuit based on two qubit gates
    commutativity relations.

    Args:
        gates_qubits_pairs (list): List of qubits tuples where gates/blocks acts. The tuples
            have two qubits, except for the ones of barriers, which can have more.

    Returns:
        (:class:`networkx.DiGraph`): Adjoint of the circuit.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(gates_qubits_pairs)))

    for i, _ in enumerate(gates_qubits_pairs):
        dag.nodes[i]["qubits"] = gates_qubits_pairs[i]

    # Find all successors
    connectivity_list = []
    for idx, gate in enumerate(gates_qubits_pairs):
        saturated_qubits = []
        for next_idx, next_gate in enumerate(gates_qubits_pairs[idx + 1 :]):
            for qubit in gate:
                if (qubit in next_gate) and (qubit not in saturated_qubits):
                    saturated_qubits.append(qubit)
                    connectivity_list.append((idx, next_idx + idx + 1))
            if len(saturated_qubits) >= len(gate):
                break
    dag.add_edges_from(connectivity_list)

    return _remove_redundant_connections(dag)


def _find_connected_qubit(
    qubits: Sequence[int], queue: list, error: type[Exception], mapping: list[int]
) -> int:
    """Helper method for :class:`qibo.transpiler.router.StarConnectivityRouter`
    and :class:`qibo.transpiler.placer.StarConnectivityPlacer`.

    Finds which qubit should be mapped to hardware middle qubit
    by looking at the two-qubit gates that follow.
    """
    possible_qubits = {qubits[0], qubits[1]}
    for next_gate in queue:
        if isinstance(next_gate, gates.Barrier):
            continue
        if len(next_gate.qubits) > 2:
            raise_error(
                error,
                "Gates targeting more than 2 qubits are not supported",
            )
        if len(next_gate.qubits) == 2:
            possible_qubits &= {
                mapping[next_gate.qubits[0]],
                mapping[next_gate.qubits[1]],
            }

            if len(possible_qubits) == 0:
                return qubits[0]
            if len(possible_qubits) == 1:
                return possible_qubits.pop()

    return qubits[0]


@cache
def _native_counts(native_gates: NativeGates) -> tuple[int, dict]:
    """Helper method for :class:`qibo.transpiler.router.Sabre`.

    Native gates that a CZ gate and a SWAP gate are decomposed into.

    The gates are decomposed as the preset pipeline does at the end: they are unrolled to the
    native gates, the gates of a directed coupling are reversed if they act against its
    direction and unrolled again, and the single-qubit gates are optimized, if the native gates
    allow it. The single-qubit gates that are virtual, *i.e.* the rotations around Z, have no
    error, so they are not counted. The result is the same for the same ``native_gates``.

    Args:
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates of the
            hardware.

    Returns:
        tuple[int, dict]: Number of native two-qubit gates of a CZ gate, and the counts of
        the SWAP gate. The counts are the values of a dictionary whose keys are
        ``(directed, order)``. ``directed`` tells if the SWAP gate is on a coupling from
        the qubit :math:`0` to the qubit :math:`1`, and then the gates against it are
        reversed. ``order`` is :math:`0` if the gate is ``SWAP(0, 1)`` and :math:`1` if it is
        ``SWAP(1, 0)``, which can decompose differently. Each value is the number of native
        two-qubit gates, and of native single-qubit gates on the qubits :math:`0` and
        :math:`1`.
    """
    unroller = Unroller(native_gates)
    optimizer = Optimize1qGatesDecomposition(basis=native_gates.gate_names)
    direction = GateDirection()
    direction.connectivity = nx.DiGraph([(0, 1)])

    swaps = {}
    for directed in (False, True):
        for order in (0, 1):
            circuit = Circuit(2)
            circuit.add(gates.SWAP(order, 1 - order))
            circuit = unroller(circuit)
            if directed:
                circuit = unroller(direction(circuit))
            # the optimizer does nothing if the native gates have no Euler basis
            if optimizer.euler_bases:
                circuit = FixedPoint([optimizer, InverseCancellation()])(circuit)

            singles = [
                gate.qubits[0]
                for gate in circuit.queue
                if len(gate.qubits) == 1
                and not isinstance(gate, (*_VIRTUAL, gates.Barrier, gates.M))
            ]
            swaps[directed, order] = (
                sum(len(gate.qubits) == 2 for gate in circuit.queue),
                singles.count(0),
                singles.count(1),
            )

    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    return (
        sum(len(gate.qubits) == 2 for gate in unroller(circuit).queue),
        swaps,
    )


def _noise_costs(
    weights: nx.Graph, connectivity: nx.Graph, nodes: list, native_gates: NativeGates
) -> tuple[np.ndarray, np.ndarray]:
    """Helper method for :class:`qibo.transpiler.router.Sabre`.

    Infidelity of a CZ gate and of a SWAP gate on each coupling.

    The weight of a coupling is the one of ``weights``, which is minus the logarithm of the
    fidelity of its native two-qubit gate, at most ``_MAX_WEIGHT``, plus ``_HOP_WEIGHT``. If
    ``weights`` is directed and only has one direction of the coupling, it is the one of that
    direction, and the gates against it are reversed. Otherwise, it is the mean of the weights
    of its directions. The weight of a qubit is the one of its native single-qubit gates, at
    most ``_MAX_WEIGHT``. A CZ gate costs as many native two-qubit gates as its
    decomposition has, and a SWAP gate costs the native gates that its decomposition has, see
    :func:`qibo.transpiler.router._native_counts`: the two-qubit gates, with the weight of
    the coupling, and the single-qubit gates on each qubit, with its weight.

    Args:
        weights (:class:`networkx.Graph`): Weighted connectivity of a hardware model, which
            can be directed, with the nodes in ``nodes``, see
            :meth:`qibo.transpiler.hardware.HardwareModel.weighted_connectivity`.
        connectivity (:class:`networkx.Graph`): Connectivity of the router, whose couplings
            are in ``weights`` in any direction.
        nodes (list): Nodes of ``connectivity``, which set the order of the rows and the
            columns.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates of the
            hardware.

    Returns:
        tuple[:class:`numpy.ndarray`, :class:`numpy.ndarray`]: Infidelity of a CZ gate on each
        coupling, and the one of the SWAP gate whose first qubit is the row and whose second
        qubit is the column, which are infinite where there is no coupling.
    """
    per_cz, swaps = _native_counts(native_gates)
    index = {node: position for position, node in enumerate(nodes)}
    single = {node: min(weights.nodes[node]["gate"], _MAX_WEIGHT) for node in nodes}

    gate_cost = np.full((len(nodes), len(nodes)), np.inf)
    swap_ordered = np.full((len(nodes), len(nodes)), np.inf)
    for first, second in connectivity.edges:
        forward = weights.has_edge(first, second)
        directed = not (forward and weights.has_edge(second, first))
        # the gates of a directed coupling go from the source to the target
        source, target = (first, second) if forward else (second, first)
        values = [
            weights.edges[pair]["weight"]
            for pair in ((first, second), (second, first))
            if weights.has_edge(*pair)
        ]
        weight = min(sum(values) / len(values), _MAX_WEIGHT) + _HOP_WEIGHT

        gate_cost[index[first], index[second]] = per_cz * weight
        gate_cost[index[second], index[first]] = per_cz * weight
        for start, end in ((first, second), (second, first)):
            # the qubits are numbered 0 and 1 as in the SWAP gate if the coupling is not
            # directed, and as the source and the target if it is
            if directed:
                qubits = (source, target)
                number = swaps[True, int(start != source)]
            else:
                qubits = (start, end)
                number = swaps[False, 0]
            two, first_singles, second_singles = number
            swap_ordered[index[start], index[end]] = (
                two * weight
                + first_singles * single[qubits[0]]
                + second_singles * single[qubits[1]]
            )

    return gate_cost, swap_ordered


def _noise_distances(
    gate_cost: np.ndarray, swap_cost: np.ndarray
) -> dict[int, np.ndarray]:
    """Helper method for :class:`qibo.transpiler.router.Sabre`.

    Infidelity that a gate needs for each pair of qubits.

    Let :math:`S_{ab}` be the lowest cost of the SWAP gates that move a qubit from :math:`a`
    to :math:`b`. The cost of a gate that needs :math:`k` CZ gates between the qubits
    :math:`a` and :math:`b` is the lowest of

    .. math::
        S_{au} + k \\, c_{uv} + S_{vb} \\, ,

    over the couplings :math:`(u, v)` in both directions, where :math:`c_{uv}` is the cost of
    a CZ gate: the qubits are moved to the ends of a coupling, where the gate is applied. The
    qubits of different connected components have a finite distance that is larger than the
    other ones, like in :class:`qibo.transpiler.router.Sabre`, since no gate acts on them.

    Args:
        gate_cost (:class:`numpy.ndarray`): Cost of a CZ gate on each coupling, which is
            infinite where there is none, see :func:`qibo.transpiler.router._noise_costs`.
        swap_cost (:class:`numpy.ndarray`): Cost of a SWAP gate on each coupling, which is
            symmetric and infinite where there is none.

    Returns:
        dict: Matrices of the infidelity for the number of CZ gates from one to three.
    """
    size = len(gate_cost)
    swap = swap_cost.copy()
    np.fill_diagonal(swap, 0.0)
    for middle in range(size):
        swap = np.minimum(swap, swap[:, middle, None] + swap[None, middle, :])

    couplings = np.argwhere(np.isfinite(np.triu(gate_cost)))
    distances = {}
    for cost in range(1, _MAX_CZ + 1):
        distance = np.full((size, size), np.inf)
        for first, second in couplings:
            for start, end in ((first, second), (second, first)):
                distance = np.minimum(
                    distance,
                    swap[:, start, None]
                    + cost * gate_cost[first, second]
                    + swap[None, end, :],
                )
        finite = np.isfinite(distance)
        distance[~finite] = distance[finite].max() + 1.0
        distances[cost] = distance

    return distances


def _relabel_weights(
    weights: nx.Graph, connectivity: nx.Graph, mapping: dict
) -> nx.Graph:
    """Helper method for :class:`qibo.transpiler.router.Sabre` and
    :class:`qibo.transpiler.router.ShortestPaths`.

    Weights of a hardware model on the connectivity of a router, with the nodes relabeled.

    Args:
        weights (:class:`networkx.Graph`): Weighted connectivity of a hardware model, see
            :meth:`qibo.transpiler.hardware.HardwareModel.weighted_connectivity`.
        connectivity (:class:`networkx.Graph`): Connectivity of the router, with the nodes
            before the relabeling.
        mapping (dict): New label of each node of ``connectivity``.

    Returns:
        :class:`networkx.Graph`: Graph with the nodes and the couplings of ``connectivity``,
        relabeled with ``mapping``. Each node has the attributes ``"gate"``, ``"readout"``
        and ``"weight"`` that ``weights`` has, and each coupling has the attribute
        ``"weight"``, which is the mean of its directions if ``weights`` is directed.

    Raises:
        ConnectivityError: If ``connectivity`` has qubits or couplings that ``weights`` does
            not have.
    """
    edge_weight = _connectivity_weights(connectivity, weights)
    graph = nx.Graph()
    graph.add_nodes_from((node, weights.nodes[node]) for node in connectivity.nodes)
    graph.add_weighted_edges_from(
        (*edge, edge_weight[edge]) for edge in connectivity.edges
    )
    return nx.relabel_nodes(graph, mapping)


def _remove_redundant_connections(dag: nx.DiGraph) -> nx.DiGraph:
    """Helper method for :func:`qibo.transpiler.router._create_dag`.

    Remove redundant connection from a DAG using transitive reduction.

    Args:
        dag (:class:`networkx.DiGraph`): DAG to be reduced.

    Returns:
        (:class:`networkx.DiGraph`): Reduced DAG.
    """
    new_dag = nx.DiGraph()
    new_dag.add_nodes_from(dag.nodes(data=True))
    transitive_reduction = nx.transitive_reduction(dag)
    new_dag.add_edges_from(transitive_reduction.edges)

    return new_dag
