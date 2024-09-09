import random
from copy import deepcopy
from typing import Optional, Union

import networkx as nx
import numpy as np

from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit
from qibo.transpiler._exceptions import ConnectivityError
from qibo.transpiler.abstract import Router
from qibo.transpiler.blocks import Block, CircuitBlocks


def assert_connectivity(connectivity: nx.Graph, circuit: Circuit):
    """Assert if a circuit can be executed on Hardware.

    No gates acting on more than two qubits.
    All two-qubit operations can be performed on hardware.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to check.
        connectivity (:class:`networkx.Graph`): chip connectivity.
    """
    if list(connectivity.nodes) != list(range(connectivity.number_of_nodes())):
        node_mapping = {node: i for i, node in enumerate(connectivity.nodes)}
        new_connectivity = nx.Graph()
        new_connectivity.add_edges_from(
            [(node_mapping[u], node_mapping[v]) for u, v in connectivity.edges]
        )
        connectivity = new_connectivity
    for gate in circuit.queue:
        if len(gate.qubits) > 2 and not isinstance(gate, gates.M):
            raise_error(ConnectivityError, f"{gate.name} acts on more than two qubits.")
        if len(gate.qubits) == 2:
            # physical_qubits = tuple(sorted((circuit.wire_names[gate.qubits[0]], circuit.wire_names[gate.qubits[1]])))
            physical_qubits = tuple(sorted(gate.qubits))  # for q_i naming
            if physical_qubits not in connectivity.edges:
                raise_error(
                    ConnectivityError,
                    f"The circuit does not respect the connectivity. {gate.name} acts on {physical_qubits} but only the following qubits are directly connected: {connectivity.edges}.",
                )


def _relabel_connectivity(connectivity, layout):
    """Relabels the connectivity graph using the passed layout.

    Args:
        connectivity (nx.Graph): input connectivity.
        layout (dict): input qubit layout.
    Returns:
        (dict) the updated connectivity.
    """
    node_mapping = {}
    layout = dict(
        sorted(layout.items(), key=lambda item: int(item[0][1:]))
    )  # for q_i naming
    for i, node in enumerate(list(layout.keys())):
        node_mapping[int(node[1:])] = i  # for q_i naming
    new_connectivity = nx.relabel_nodes(connectivity, node_mapping)
    return new_connectivity


class StarConnectivityRouter(Router):
    """Transforms an arbitrary circuit to one that can be executed on hardware.

    This transpiler produces a circuit that respects the following connectivity:

             q
             |
        q -- q -- q
             |
             q

    by adding SWAP gates when needed.

    Args:
        connectivity (:class:`networkx.Graph`): chip connectivity, not used for this transpiler.
        middle_qubit (int, optional): qubit id of the qubit that is in the middle of the star.
    """

    def __init__(self, connectivity=None, middle_qubit: int = 2):
        self.middle_qubit = middle_qubit
        if connectivity is not None:  # pragma: no cover
            log.warning(
                "StarConnectivityRouter does not use the connectivity graph."
                "The connectivity graph will be ignored."
            )

    def __call__(self, circuit: Circuit, initial_layout: dict):
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): The original Qibo circuit to transform.
                Only single qubit gates and two qubits gates are supported by the router.
            initial_layout (dict): initial physical-to-logical qubit mapping,
                use `qibo.transpiler.placer.StarConnectivityPlacer` for better performance.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, list): circuit that performs the same operation
                as the original but respects the hardware connectivity,
                and list that maps logical to hardware qubits.
        """

        middle_qubit = self.middle_qubit
        nqubits = max(circuit.nqubits, middle_qubit + 1)
        # new circuit object that will be compatible with hardware connectivity
        new = Circuit(nqubits)
        # list to maps logical to hardware qubits
        hardware_qubits = list(initial_layout.values())

        for i, gate in enumerate(circuit.queue):
            # map gate qubits to hardware
            qubits = tuple(hardware_qubits.index(q) for q in gate.qubits)
            if isinstance(gate, gates.M):
                new_gate = gates.M(*qubits, **gate.init_kwargs)
                new_gate.result = gate.result
                new.add(new_gate)
                continue

            if len(qubits) > 2:
                raise_error(
                    ConnectivityError,
                    "Gates targeting more than two qubits are not supported.",
                )

            if len(qubits) == 2 and middle_qubit not in qubits:
                # find which qubit should be moved
                new_middle = _find_connected_qubit(
                    qubits,
                    circuit.queue[i + 1 :],
                    hardware_qubits,
                    error=ConnectivityError,
                )
                # update hardware qubits according to the swap
                hardware_qubits[middle_qubit], hardware_qubits[new_middle] = (
                    hardware_qubits[new_middle],
                    hardware_qubits[middle_qubit],
                )
                new.add(gates.SWAP(middle_qubit, new_middle))
                # update gate qubits according to the new swap
                qubits = tuple(hardware_qubits.index(q) for q in gate.qubits)

            # add gate to the hardware circuit
            if isinstance(gate, gates.Unitary):
                # gates.Unitary requires matrix as first argument
                matrix = gate.init_args[0]
                new.add(gate.__class__(matrix, *qubits, **gate.init_kwargs))
            else:
                new.add(gate.__class__(*qubits, **gate.init_kwargs))
        hardware_qubits_keys = ["q" + str(i) for i in range(5)]
        return new, dict(zip(hardware_qubits_keys, hardware_qubits))


def _find_connected_qubit(qubits, queue, hardware_qubits, error):
    """Helper method for :meth:`qibo.transpiler.router.StarConnectivityRouter`
    and :meth:`qibo.transpiler.router.StarConnectivityPlacer`.

    Finds which qubit should be mapped to hardware middle qubit
    by looking at the two-qubit gates that follow.
    """
    possible_qubits = set(qubits)
    for next_gate in queue:
        if len(next_gate.qubits) > 2:
            raise_error(
                error,
                "Gates targeting more than 2 qubits are not supported",
            )
        if len(next_gate.qubits) == 2:
            possible_qubits &= {hardware_qubits.index(q) for q in next_gate.qubits}

            if not possible_qubits:
                return qubits[0]

            if len(possible_qubits) == 1:
                return possible_qubits.pop()

    return qubits[0]


class CircuitMap:
    """Class that stores the circuit and physical-logical mapping during routing.

    Also implements the initial two-qubit block decompositions.

    Args:
        initial_layout (dict): initial physical to logical qubit mapping.
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be routed.
        blocks (:class:`qibo.transpiler.blocks.CircuitBlocks`, optional): circuit
            block representation. If ``None``, the blocks will be computed from the circuit.
            Defaults to ``None``.
    """

    def __init__(
        self,
        initial_layout: Optional[dict] = None,
        circuit: Optional[Circuit] = None,
        blocks: Optional[CircuitBlocks] = None,
        temp: Optional[bool] = False,
    ):
        self._p2l, self._l2p = [], []

        self._temporary = temp
        if self._temporary:
            return
        elif circuit is None:
            raise_error(ValueError, "Circuit must be provided.")

        if blocks is not None:
            self.circuit_blocks = blocks
        else:
            self.circuit_blocks = CircuitBlocks(circuit, index_names=True)

        self._nqubits = circuit.nqubits
        self._routed_blocks = CircuitBlocks(Circuit(circuit.nqubits))
        self._swaps = 0

        if initial_layout is None:
            return

        self.wire_names = list(initial_layout.keys())
        self.physical_to_logical = list(initial_layout.values())

    @property
    def physical_to_logical(self):
        """Returns the physical to logical qubit mapping."""
        return self._p2l

    @property
    def logical_to_physical(self):
        """Returns the logical to physical qubit mapping."""
        return self._l2p

    @physical_to_logical.setter
    def physical_to_logical(self, p2l_map: list):
        """Sets the physical to logical qubit mapping and updates the logical to physical mapping.

        Args:
            p2l_map (list): physical to logical mapping.
        """
        self._p2l = p2l_map.copy()
        self._l2p = [0] * len(self._p2l)
        for i, l in enumerate(self._p2l):
            self._l2p[l] = i

    @logical_to_physical.setter
    def logical_to_physical(self, l2p_map: list):
        """Sets the logical to physical qubit mapping and updates the physical to logical mapping.

        Args:
            l2p_map (list): logical to physical mapping.
        """
        self._l2p = l2p_map.copy()
        self._p2l = [0] * len(self._l2p)
        for i, p in enumerate(self._l2p):
            self._p2l[p] = i

    def _update_mappings_swap(self, logical_swap: tuple, physical_swap: tuple):
        """Updates the qubit mappings after applying a SWAP gate.

        Args:
            logical_swap (tuple[int]): the indices of the logical qubits to be swapped.
            physical_swap (tuple[int]): the indices of the corresponding physical qubits to be swapped.
        """
        self._p2l[physical_swap[0]], self._p2l[physical_swap[1]] = (
            logical_swap[1],
            logical_swap[0],
        )
        self._l2p[logical_swap[0]], self._l2p[logical_swap[1]] = (
            physical_swap[1],
            physical_swap[0],
        )

    def blocks_logical_qubits_pairs(self):
        """Returns a list containing the logical qubit pairs of each block."""
        return [block.qubits for block in self.circuit_blocks()]

    def execute_block(self, block: Block):
        """Executes a block by removing it from the circuit representation
        and adding it to the routed circuit.

        Method works in-place.

        Args:
            block (:class:`qibo.transpiler.blocks.Block`): block to be removed.
        """
        self._routed_blocks.add_block(block.on_qubits(self.get_physical_qubits(block)))
        self.circuit_blocks.remove_block(block)

    def routed_circuit(self, circuit_kwargs: Optional[dict] = None):
        """Returns the routed circuit.

        Args:
            circuit_kwargs (dict): original circuit ``init_kwargs``.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Routed circuit.
        """
        return self._routed_blocks.circuit(circuit_kwargs=circuit_kwargs)

    def final_layout(self):
        """Returns the final physical-logical qubits mapping."""

        return {self.wire_names[i]: self._p2l[i] for i in range(self._nqubits)}

    def update(self, logical_swap: tuple):
        """Updates the qubit mapping after applying a ``SWAP``

        Adds the :class:`qibo.gates.gates.SWAP` gate to the routed blocks.
        Method works in-place.

        Args:
            swap (tuple): tuple containing the logical qubits to be swapped.
        """

        physical_swap = self.logical_pair_to_physical(logical_swap)
        if not self._temporary:
            self._routed_blocks.add_block(
                Block(qubits=physical_swap, gates=[gates.SWAP(*physical_swap)])
            )
            self._swaps += 1

        self._update_mappings_swap(logical_swap, physical_swap)

    def undo(self):
        """Undo the last swap. Method works in-place."""
        last_swap_block = self._routed_blocks.return_last_block()
        physical_swap = last_swap_block.qubits
        logical_swap = self._p2l[physical_swap[0]], self._p2l[physical_swap[1]]
        self._routed_blocks.remove_block(last_swap_block)
        self._swaps -= 1

        self._update_mappings_swap(logical_swap, physical_swap)

    def get_physical_qubits(self, block: Union[int, Block]):
        """Returns the physical qubits where a block is acting on.

        Args:
            block (int or :class:`qibo.transpiler.blocks.Block`): block to be analysed.

        Returns:
            tuple: physical qubit numbers where a block is acting on.
        """
        if isinstance(block, int):
            block = self.circuit_blocks.search_by_index(block)

        return tuple(self._l2p[q] for q in block.qubits)

    def logical_pair_to_physical(self, logical_qubits: tuple):
        """Returns the physical qubits associated to the logical qubit pair.

        Args:
            logical_qubits (tuple): logical qubit pair.

        Returns:
            tuple: physical qubit numbers associated to the logical qubit pair.
        """
        return self._l2p[logical_qubits[0]], self._l2p[logical_qubits[1]]


class ShortestPaths(Router):
    """A class to perform initial qubit mapping and connectivity matching.

    Args:
        connectivity (:class:`networkx.Graph`): chip connectivity.
        seed (int, optional): seed for the random number generator.
            If ``None``, defaults to :math:`42`. Defaults to ``None``.
    """

    def __init__(self, connectivity: nx.Graph, seed: Optional[int] = None):
        self.connectivity = connectivity
        self._front_layer = None
        self.circuit_map = None
        self._dag = None
        self._final_measurements = None
        self._node_mapping_inv = None
        if seed is None:
            seed = 42
        random.seed(seed)

    @property
    def added_swaps(self):
        """Returns the number of SWAP gates added to the circuit during routing."""
        return self.circuit_map._swaps

    def __call__(self, circuit: Circuit, initial_layout: dict):
        """Circuit connectivity matching.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be matched
                to hardware connectivity.
            initial_layout (dict): initial physical-to-logical qubit mapping

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): circut mapped to hardware topology,
                and final physical-to-logical qubit mapping.
        """
        self._preprocessing(circuit=circuit, initial_layout=initial_layout)
        while self._dag.number_of_nodes() != 0:
            execute_block_list = self._check_execution()
            if execute_block_list is not None:
                self._execute_blocks(execute_block_list)
            else:
                self._find_new_mapping()

        circuit_kwargs = circuit.init_kwargs
        circuit_kwargs["wire_names"] = list(initial_layout.keys())
        routed_circuit = self.circuit_map.routed_circuit(circuit_kwargs=circuit_kwargs)
        if self._final_measurements is not None:
            routed_circuit = self._append_final_measurements(
                routed_circuit=routed_circuit
            )

        return routed_circuit, self.circuit_map.final_layout()

    def _find_new_mapping(self):
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
        best_candidate = random.choice(best_candidates)
        self._add_swaps(best_candidate, self.circuit_map)

    def _candidates(self):
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
    def _add_swaps(candidate: tuple, circuitmap: CircuitMap):
        """Adds swaps to the circuit to move qubits.

        Method works in-place.

        Args:
            candidate (tuple): contains path to move qubits and qubit meeting point in the path.
            circuitmap (CircuitMap): representation of the circuit.
        """
        path = candidate[0]
        meeting_point = candidate[1]
        forward = path[0 : meeting_point + 1]
        backward = list(reversed(path[meeting_point + 1 :]))
        for f in forward[1:]:
            circuitmap.update(
                (
                    circuitmap.physical_to_logical[f],
                    circuitmap.physical_to_logical[forward[0]],
                )
            )
        for b in backward[1:]:
            circuitmap.update(
                (
                    circuitmap.physical_to_logical[b],
                    circuitmap.physical_to_logical[backward[0]],
                )
            )

    def _compute_cost(self, candidate: tuple):
        """Greedy algorithm that decides which path to take and how qubits should be walked.

        The cost is computed as minus the number of successive gates that can be executed.

        Args:
            candidate (tuple): contains path to move qubits and qubit meeting point in the path.

        Returns:
            (list, int): best path to move qubits and qubit meeting point in the path.
        """
        temporary_circuit = CircuitMap(
            circuit=Circuit(self.circuit_map._nqubits),
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

    def _check_execution(self):
        """Check if some blocks in the front layer can be executed in the current configuration.

        Returns:
            (list): executable blocks if there are, ``None`` otherwise.
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

    def _execute_blocks(self, blocklist: list):
        """Executes a list of blocks:
            -Remove the correspondent nodes from the dag and circuit representation.
            -Add the executed blocks to the routed circuit.
            -Update the dag layers and front layer.

        Method works in-place.

        Args:
            blocklist (list): list of blocks.
        """
        for block_id in blocklist:
            block = self.circuit_map.circuit_blocks.search_by_index(block_id)
            self.circuit_map.execute_block(block)
            self._dag.remove_node(block_id)
        self._update_front_layer()

    def _update_front_layer(self):
        """Updates the front layer of the dag.

        Method works in-place.
        """
        for layer, nodes in enumerate(nx.topological_generations(self._dag)):
            for node in nodes:
                self._dag.nodes[node]["layer"] = layer
        self._front_layer = [
            node[0] for node in self._dag.nodes(data="layer") if node[1] == 0
        ]

    def _preprocessing(self, circuit: Circuit, initial_layout: dict):
        """The following objects will be initialised:
            - circuit: class to represent circuit and to perform logical-physical qubit mapping.
            - _final_measurements: measurement gates at the end of the circuit.
            - _front_layer: list containing the blocks to be executed.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be preprocessed.
            initial_layout (dict): initial physical-to-logical qubit mapping.
        """

        self.connectivity = _relabel_connectivity(self.connectivity, initial_layout)

        copied_circuit = circuit.copy(deep=True)
        self._final_measurements = self._detach_final_measurements(copied_circuit)
        self.circuit_map = CircuitMap(initial_layout, copied_circuit)
        self._dag = _create_dag(self.circuit_map.blocks_logical_qubits_pairs())
        self._update_front_layer()

    def _detach_final_measurements(self, circuit: Circuit):
        """Detaches measurement gates at the end of the circuit for separate handling.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuits to be processed.

        Returns:
            (NoneType or list): list of measurements. If no measurements, returns ``None``.
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

    def _append_final_measurements(self, routed_circuit: Circuit):
        """Appends the final measurment gates on the correct qubits
        conserving the measurement register."""
        for measurement in self._final_measurements:
            original_qubits = measurement.qubits
            routed_qubits = list(
                self.circuit_map.logical_to_physical[qubit] for qubit in original_qubits
            )
            routed_circuit.add(
                measurement.on_qubits(dict(zip(original_qubits, routed_qubits)))
            )

        return routed_circuit


class Sabre(Router):
    """Routing algorithm proposed in Ref [1].

    Args:
        connectivity (:class:`networkx.Graph`): hardware chip connectivity.
        lookahead (int, optional): lookahead factor, how many dag layers will be considered
            in computing the cost. Defaults to :math:`2`.
        decay_lookahead (float, optional): value in interval :math:`[0, 1]`.
            How the weight of the distance in the dag layers decays in computing the cost.
            Defaults to :math:`0.6`.
        delta (float, optional): defines the number of SWAPs vs depth trade-off by deciding
            how the algorithm tends to select non-overlapping SWAPs.
            Defaults to math:`10^{-3}`.
        seed (int, optional): seed for the candidate random choice as tiebraker.
            Defaults to ``None``.
        swap_threshold (float, optional): limits the number of added SWAPs in every routing iteration.
            This threshold is multiplied by the length of the longest path in the circuit connectivity.
            If the number of added SWAPs exceeds the threshold before a gate is routed,
            shortestpath routing is applied.
            Defaults to :math:`1.5`.

    References:
        1. G. Li, Y. Ding, and Y. Xie,
        *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*.
        `arXiv:1809.02573 [cs.ET] <https://arxiv.org/abs/1809.02573>`_.
    """

    def __init__(
        self,
        connectivity: nx.Graph,
        lookahead: int = 2,
        decay_lookahead: float = 0.6,
        delta: float = 0.001,
        swap_threshold: float = 1.5,
        seed: Optional[int] = None,
    ):
        self.connectivity = connectivity
        self.lookahead = lookahead
        self.decay = decay_lookahead
        self.delta = delta
        self.swap_threshold = swap_threshold
        self._delta_register = None
        self._dist_matrix = None
        self._dag = None
        self._front_layer = None
        self.circuit_map = None
        self._memory_map = None
        self._final_measurements = None
        self._temp_added_swaps = []
        random.seed(seed)

    def __call__(self, circuit: Circuit, initial_layout: dict):
        """Route the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be routed.
            initial_layout (dict): initial physical to logical qubit mapping.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): routed circuit and final layout.
        """
        self._preprocessing(circuit=circuit, initial_layout=initial_layout)
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
        circuit_kwargs["wire_names"] = list(initial_layout.keys())
        routed_circuit = self.circuit_map.routed_circuit(circuit_kwargs=circuit_kwargs)
        if self._final_measurements is not None:
            routed_circuit = self._append_final_measurements(
                routed_circuit=routed_circuit
            )

        return routed_circuit, self.circuit_map.final_layout()

    @property
    def added_swaps(self):
        """Returns the number of SWAP gates added to the circuit during routing."""
        return self.circuit_map._swaps

    def _preprocessing(self, circuit: Circuit, initial_layout: dict):
        """The following objects will be initialised:
            - circuit: class to represent circuit and to perform logical-physical qubit mapping.
            - _final_measurements: measurement gates at the end of the circuit.
            - _dist_matrix: matrix reporting the shortest path lengh between all node pairs.
            - _dag: direct acyclic graph of the circuit based on commutativity.
            - _memory_map: list to remember previous SWAP moves.
            - _front_layer: list containing the blocks to be executed.
            - _delta_register: list containing the special weigh added to qubits
                to prevent overlapping swaps.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be preprocessed.
            initial_layout (dict): initial physical-to-logical qubit mapping.
        """

        self.connectivity = _relabel_connectivity(self.connectivity, initial_layout)

        copied_circuit = circuit.copy(deep=True)
        self._final_measurements = self._detach_final_measurements(copied_circuit)
        self.circuit_map = CircuitMap(initial_layout, copied_circuit)
        self._dist_matrix = nx.floyd_warshall_numpy(self.connectivity)
        self._dag = _create_dag(self.circuit_map.blocks_logical_qubits_pairs())
        self._memory_map = []
        self._update_dag_layers()
        self._update_front_layer()
        self._delta_register = [1.0 for _ in range(circuit.nqubits)]

    def _detach_final_measurements(self, circuit: Circuit):
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

    def _append_final_measurements(self, routed_circuit: Circuit):
        """Appends final measurment gates on the correct qubits conserving the measurement register.

        Args:
            routed_circuit (:class:`qibo.models.circuit.Circuit`): original circuit.

        Returns:
            (:class:`qibo.models.circuit.Circuit`) routed circuit.
        """
        for measurement in self._final_measurements:
            original_qubits = measurement.qubits
            routed_qubits = list(
                self.circuit_map.logical_to_physical[qubit] for qubit in original_qubits
            )
            routed_circuit.add(
                measurement.on_qubits(dict(zip(original_qubits, routed_qubits)))
            )
        return routed_circuit

    def _update_dag_layers(self):
        """Update dag layers and put them in topological order.

        Method works in-place.
        """
        for layer, nodes in enumerate(nx.topological_generations(self._dag)):
            for node in nodes:
                self._dag.nodes[node]["layer"] = layer

    def _update_front_layer(self):
        """Update the front layer of the dag.

        Method works in-place.
        """
        self._front_layer = self._get_dag_layer(0)

    def _get_dag_layer(self, n_layer, qubits=False):
        """Return the :math:`n`-topological layer of the dag.

        Args:
            n_layer (int): layer number.
            qubits (bool, optional): if ``True``, return the target qubits of the blocks in the layer.
                If ``False``, return the block numbers. Defaults to ``False``.

        Returns:
            (list): list of block numbers or target qubits.
        """

        if qubits:
            return [
                node[1]["qubits"]
                for node in self._dag.nodes(data=True)
                if node[1]["layer"] == n_layer
            ]

        return [node[0] for node in self._dag.nodes(data="layer") if node[1] == n_layer]

    def _find_new_mapping(self):
        """Find the new best mapping by adding one swap."""
        candidates_evaluation = {}

        self._memory_map.append(self.circuit_map.physical_to_logical.copy())
        for candidate in self._swap_candidates():
            candidates_evaluation[candidate] = self._compute_cost(candidate)

        best_cost = min(candidates_evaluation.values())
        best_candidates = [
            key for key, value in candidates_evaluation.items() if value == best_cost
        ]
        best_candidate = random.choice(best_candidates)

        for qubit in self.circuit_map.logical_pair_to_physical(best_candidate):
            self._delta_register[qubit] += self.delta
        self.circuit_map.update(best_candidate)
        self._temp_added_swaps.append(best_candidate)

    def _compute_cost(self, candidate: int):
        """Compute the cost associated to a possible SWAP candidate."""

        temporary_circuit = CircuitMap(temp=True)
        temporary_circuit.physical_to_logical = self.circuit_map.physical_to_logical
        temporary_circuit.update(candidate)

        if temporary_circuit.physical_to_logical in self._memory_map:
            return float("inf")

        tot_distance = 0.0
        weight = 1.0
        for layer in range(self.lookahead + 1):
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

    def _swap_candidates(self):
        """Returns a list of possible candidate SWAPs to be applied on logical qubits directly.

        The possible candidates are the ones sharing at least one qubit
        with a block in the front layer.

        Returns:
            (list): list of candidates.
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

    def _check_execution(self):
        """Check if some blocks in the front layer can be executed in the current configuration.

        Returns:
            (list): executable blocks if there are, ``None`` otherwise.
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

    def _execute_blocks(self, blocklist: list):
        """Executes a list of blocks:
        -Remove the correspondent nodes from the dag and circuit representation.
        -Add the executed blocks to the routed circuit.
        -Update the dag layers and front layer.
        -Reset the mapping memory.

        Method works in-place.

        Args:
            blocklist (list): list of blocks.
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

    def _shortest_path_routing(self):
        """Route a gate in the front layer using the shortest path. This method is executed when the standard SABRE fails to find an optimized solution.

        Method works in-place.
        """

        min_distance = float("inf")
        shortest_path_qubits = None

        for block in self._front_layer:
            q1, q2 = self.circuit_map.get_physical_qubits(block)
            distance = self._dist_matrix[q1, q2]

            if distance < min_distance:
                min_distance = distance
                shortest_path_qubits = [q1, q2]

        shortest_path = nx.bidirectional_shortest_path(
            self.connectivity, shortest_path_qubits[0], shortest_path_qubits[1]
        )

        # move q1
        q1 = self.circuit_map.physical_to_logical[shortest_path[0]]
        for q2 in shortest_path[1:-1]:
            self.circuit_map.update((q1, self.circuit_map.physical_to_logical[q2]))


def _create_dag(gates_qubits_pairs: list):
    """Helper method for :meth:`qibo.transpiler.router.Sabre`.

    Create direct acyclic graph (dag) of the circuit based on two qubit gates
    commutativity relations.

    Args:
        gates_qubits_pairs (list): list of qubits tuples where gates/blocks acts.

    Returns:
        (:class:`networkx.DiGraph`): adjoint of the circuit.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(gates_qubits_pairs)))

    for i in range(len(gates_qubits_pairs)):
        dag.nodes[i]["qubits"] = gates_qubits_pairs[i]

    # Find all successors
    connectivity_list = []
    for idx, gate in enumerate(gates_qubits_pairs):
        saturated_qubits = []
        for next_idx, next_gate in enumerate(gates_qubits_pairs[idx + 1 :]):
            for qubit in gate:
                if (qubit in next_gate) and (not qubit in saturated_qubits):
                    saturated_qubits.append(qubit)
                    connectivity_list.append((idx, next_idx + idx + 1))
            if len(saturated_qubits) >= 2:
                break
    dag.add_edges_from(connectivity_list)

    return _remove_redundant_connections(dag)


def _remove_redundant_connections(dag: nx.DiGraph):
    """Helper method for :func:`qibo.transpiler.router._create_dag`.

    Remove redundant connection from a DAG using transitive reduction.

    Args:
        dag (:class:`networkx.DiGraph`): dag to be reduced.

    Returns:
        (:class:`networkx.DiGraph`): reduced dag.
    """
    new_dag = nx.DiGraph()
    new_dag.add_nodes_from(dag.nodes(data=True))
    transitive_reduction = nx.transitive_reduction(dag)
    new_dag.add_edges_from(transitive_reduction.edges)

    return new_dag
