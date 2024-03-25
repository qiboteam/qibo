import random
from copy import deepcopy
from typing import Optional, Union

import networkx as nx

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
            if (gate.qubits[0], gate.qubits[1]) not in connectivity.edges:
                raise_error(
                    ConnectivityError,
                    f"Circuit does not respect connectivity. {gate.name} acts on {gate.qubits}.",
                )


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
                raise ConnectivityError(
                    "Gates targeting more than two qubits are not supported."
                )

            elif len(qubits) == 2 and middle_qubit not in qubits:
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
            elif len(possible_qubits) == 1:
                return possible_qubits.pop()
    return qubits[0]


class CircuitMap:
    """Class that stores the circuit and physical-logical mapping during routing.

    Also implements the initial two-qubit block decompositions.

    Args:
        initial_layout (dict): initial logical-to-physical qubit mapping.
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be routed.
        blocks (:class:`qibo.transpiler.blocks.CircuitBlocks`, optional): circuit block representation.
            If ``None``, the blocks will be computed from the circuit.
            Defaults to ``None``.
    """

    def __init__(
        self,
        initial_layout: dict,
        circuit: Circuit,
        blocks: Optional[CircuitBlocks] = None,
    ):
        if blocks is not None:
            self.circuit_blocks = blocks
        else:
            self.circuit_blocks = CircuitBlocks(circuit, index_names=True)
        self.initial_layout = initial_layout
        self._graph_qubits_names = [int(key[1:]) for key in initial_layout.keys()]
        self._circuit_logical = list(range(len(initial_layout)))
        self._physical_logical = list(initial_layout.values())
        self._routed_blocks = CircuitBlocks(Circuit(circuit.nqubits))
        self._swaps = 0

    def set_circuit_logical(self, circuit_logical_map: list):
        """Sets the current circuit to logical qubit mapping.

        Method works in-place.

        Args:
            circuit_logical_map (list): logical mapping.
        """
        self._circuit_logical = circuit_logical_map

    def blocks_qubits_pairs(self):
        """Returns a list containing the qubit pairs of each block."""
        return [block.qubits for block in self.circuit_blocks()]

    def execute_block(self, block: Block):
        """Executes a block by removing it from the circuit representation
        and adding it to the routed circuit.

        Method works in-place.

        Args:
            block (:class:`qibo.transpiler.blocks.Block`): block to be removed.
        """
        self._routed_blocks.add_block(
            block.on_qubits(self.get_physical_qubits(block, index=True))
        )
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
        """Returns the final physical-circuit qubits mapping."""
        unsorted_dict = {
            "q" + str(self.circuit_to_physical(i)): i
            for i in range(len(self._circuit_logical))
        }

        return dict(sorted(unsorted_dict.items()))

    def update(self, swap: tuple):
        """Updates the logical-physical qubit mapping after applying a ``SWAP``

        Adds the :class:`qibo.gates.gates.SWAP` gate to the routed blocks.
        Method works in-place.

        Args:
            swap (tuple): tuple containing the logical qubits to be swapped.
        """
        physical_swap = self.logical_to_physical(swap, index=True)
        self._routed_blocks.add_block(
            Block(qubits=physical_swap, gates=[gates.SWAP(*physical_swap)])
        )
        self._swaps += 1
        idx_0, idx_1 = self._circuit_logical.index(
            swap[0]
        ), self._circuit_logical.index(swap[1])
        self._circuit_logical[idx_0], self._circuit_logical[idx_1] = swap[1], swap[0]

    def get_logical_qubits(self, block: Block):
        """Returns the current logical qubits where a block is acting on.

        Args:
            block (:class:`qibo.transpiler.blocks.Block`): block to be analysed.

        Returns:
            (tuple): logical qubits where a block is acting on.
        """
        return self.circuit_to_logical(block.qubits)

    def get_physical_qubits(self, block: Union[int, Block], index: bool = False):
        """Returns the physical qubits where a block is acting on.

        Args:
            block (int or :class:`qibo.transpiler.blocks.Block`): block to be analysed.
            index (bool, optional): If ``True``, qubits are returned as indices of
                the connectivity nodes. Defaults to ``False``.

        Returns:
            (tuple): physical qubits where a block is acting on.

        """
        if isinstance(block, int):
            block = self.circuit_blocks.search_by_index(block)

        return self.logical_to_physical(self.get_logical_qubits(block), index=index)

    def logical_to_physical(self, logical_qubits: tuple, index: bool = False):
        """Returns the physical qubits associated to the logical qubits.

        Args:
            logical_qubits (tuple): physical qubits.
            index (bool, optional): If ``True``, qubits are returned as indices of
                `the connectivity nodes. Defaults to ``False``.

        Returns:
            (tuple): physical qubits associated to the logical qubits.
        """
        if not index:
            return tuple(
                self._graph_qubits_names[
                    self._physical_logical.index(logical_qubits[i])
                ]
                for i in range(2)
            )

        return tuple(self._physical_logical.index(logical_qubits[i]) for i in range(2))

    def circuit_to_logical(self, circuit_qubits: tuple):
        """Returns the current logical qubits associated to the initial circuit qubits.

        Args:
            circuit_qubits (tuple): circuit qubits.

        Returns:
            (tuple): logical qubits.
        """
        return tuple(self._circuit_logical[circuit_qubits[i]] for i in range(2))

    def circuit_to_physical(self, circuit_qubit: int):
        """Returns the current physical qubit associated to an initial circuit qubit.

        Args:
            circuit_qubit (int): circuit qubit.

        Returns:
            (int): physical qubit.
        """
        return self._graph_qubits_names[
            self._physical_logical.index(self._circuit_logical[circuit_qubit])
        ]

    def physical_to_logical(self, physical_qubit: int):
        """Returns the current logical qubit associated to a physical qubit (connectivity graph node).

        Args:
            physical_qubit (int): physical qubit.

        Returns:
            (int): logical qubit.
        """
        physical_qubit_index = self._graph_qubits_names.index(physical_qubit)

        return self._physical_logical[physical_qubit_index]


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
        self.circuit = None
        self._dag = None
        self._final_measurements = None
        if seed is None:
            seed = 42
        random.seed(seed)

    @property
    def added_swaps(self):
        """Returns the number of SWAP gates added to the circuit during routing."""
        return self.circuit._swaps

    def __call__(self, circuit: Circuit, initial_layout: dict):
        """Circuit connectivity matching.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be matched to hardware connectivity.
            initial_layout (dict): initial physical-to-logical qubit mapping

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): circut mapped to hardware topology, and final physical-to-logical qubit mapping.
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
        routed_circuit = self.circuit.routed_circuit(circuit_kwargs=circuit_kwargs)
        if self._final_measurements is not None:
            routed_circuit = self._append_final_measurements(
                routed_circuit=routed_circuit
            )

        return routed_circuit, self.circuit.final_layout()

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
        self._add_swaps(best_candidate, self.circuit)

    def _candidates(self):
        """Returns all possible shortest paths in a ``list`` that contains
        the new mapping and a second ``list`` containing the path meeting point.
        """
        target_qubits = self.circuit.get_physical_qubits(self._front_layer[0])
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
        if len(forward) > 1:
            for f1, f2 in zip(forward[:-1], forward[1:]):
                circuitmap.update(
                    (
                        circuitmap.physical_to_logical(f1),
                        circuitmap.physical_to_logical(f2),
                    )
                )
        if len(backward) > 1:
            for b1, b2 in zip(backward[:-1], backward[1:]):
                circuitmap.update(
                    (
                        circuitmap.physical_to_logical(b1),
                        circuitmap.physical_to_logical(b2),
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
            initial_layout=self.circuit.initial_layout,
            circuit=Circuit(len(self.circuit.initial_layout)),
            blocks=deepcopy(self.circuit.circuit_blocks),
        )
        temporary_circuit.set_circuit_logical(deepcopy(self.circuit._circuit_logical))
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
        """Checks if some blocks in the front layer can be executed in the current configuration.

        Returns:
            (list): executable blocks if there are, ``None`` otherwise.
        """
        executable_blocks = []
        for block in self._front_layer:
            if (
                self.circuit.get_physical_qubits(block) in self.connectivity.edges
                or not self.circuit.circuit_blocks.search_by_index(block).entangled
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
            block = self.circuit.circuit_blocks.search_by_index(block_id)
            self.circuit.execute_block(block)
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
        copied_circuit = circuit.copy(deep=True)
        self._final_measurements = self._detach_final_measurements(copied_circuit)
        self.circuit = CircuitMap(initial_layout, copied_circuit)
        self._dag = _create_dag(self.circuit.blocks_qubits_pairs())
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
        """Appends the final measurment gates on the correct qubits conserving the measurement register."""
        for measurement in self._final_measurements:
            original_qubits = measurement.qubits
            routed_qubits = (
                self.circuit.circuit_to_physical(qubit) for qubit in original_qubits
            )
            routed_circuit.add(
                measurement.on_qubits(dict(zip(original_qubits, routed_qubits)))
            )

        return routed_circuit


class Sabre(Router):
    def __init__(
        self,
        connectivity: nx.Graph,
        lookahead: int = 2,
        decay_lookahead: float = 0.6,
        delta: float = 0.001,
        seed: Optional[int] = None,
    ):
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

        References:
            1. G. Li, Y. Ding, and Y. Xie, *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*.
            `arXiv:1809.02573 [cs.ET] <https://arxiv.org/abs/1809.02573>`_.
        """
        self.connectivity = connectivity
        self.lookahead = lookahead
        self.decay = decay_lookahead
        self.delta = delta
        self._delta_register = None
        self._dist_matrix = None
        self._dag = None
        self._front_layer = None
        self.circuit = None
        self._memory_map = None
        self._final_measurements = None
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
        while self._dag.number_of_nodes() != 0:
            execute_block_list = self._check_execution()
            if execute_block_list is not None:
                self._execute_blocks(execute_block_list)
            else:
                self._find_new_mapping()

        circuit_kwargs = circuit.init_kwargs
        circuit_kwargs["wire_names"] = list(initial_layout.keys())
        routed_circuit = self.circuit.routed_circuit(circuit_kwargs=circuit_kwargs)
        if self._final_measurements is not None:
            routed_circuit = self._append_final_measurements(
                routed_circuit=routed_circuit
            )

        return routed_circuit, self.circuit.final_layout()

    @property
    def added_swaps(self):
        """Returns the number of SWAP gates added to the circuit during routing."""
        return self.circuit._swaps

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
        copied_circuit = circuit.copy(deep=True)
        self._final_measurements = self._detach_final_measurements(copied_circuit)
        self.circuit = CircuitMap(initial_layout, copied_circuit)
        self._dist_matrix = nx.floyd_warshall_numpy(self.connectivity)
        self._dag = _create_dag(self.circuit.blocks_qubits_pairs())
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
            routed_qubits = (
                self.circuit.circuit_to_physical(qubit) for qubit in original_qubits
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

    def _get_dag_layer(self, n_layer):
        """Return the :math:`n`-topological layer of the dag."""
        return [node[0] for node in self._dag.nodes(data="layer") if node[1] == n_layer]

    def _find_new_mapping(self):
        """Find the new best mapping by adding one swap."""
        candidates_evaluation = {}
        self._memory_map.append(deepcopy(self.circuit._circuit_logical))
        for candidate in self._swap_candidates():
            candidates_evaluation[candidate] = self._compute_cost(candidate)

        best_cost = min(candidates_evaluation.values())
        best_candidates = [
            key for key, value in candidates_evaluation.items() if value == best_cost
        ]
        best_candidate = random.choice(best_candidates)

        for qubit in self.circuit.logical_to_physical(best_candidate, index=True):
            self._delta_register[qubit] += self.delta
        self.circuit.update(best_candidate)

    def _compute_cost(self, candidate: int):
        """Compute the cost associated to a possible SWAP candidate."""
        temporary_circuit = CircuitMap(
            initial_layout=self.circuit.initial_layout,
            circuit=Circuit(len(self.circuit.initial_layout)),
            blocks=self.circuit.circuit_blocks,
        )
        temporary_circuit.set_circuit_logical(deepcopy(self.circuit._circuit_logical))
        temporary_circuit.update(candidate)

        if temporary_circuit._circuit_logical in self._memory_map:
            return float("inf")

        tot_distance = 0.0
        weight = 1.0
        for layer in range(self.lookahead + 1):
            layer_gates = self._get_dag_layer(layer)
            avg_layer_distance = 0.0
            for gate in layer_gates:
                qubits = temporary_circuit.get_physical_qubits(gate, index=True)
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
            for qubit in self.circuit.get_physical_qubits(block):
                for connected in self.connectivity.neighbors(qubit):
                    candidate = tuple(
                        sorted(
                            (
                                self.circuit.physical_to_logical(qubit),
                                self.circuit.physical_to_logical(connected),
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
                self.circuit.get_physical_qubits(block) in self.connectivity.edges
                or not self.circuit.circuit_blocks.search_by_index(block).entangled
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
            block = self.circuit.circuit_blocks.search_by_index(block_id)
            self.circuit.execute_block(block)
            self._dag.remove_node(block_id)
        self._update_dag_layers()
        self._update_front_layer()
        self._memory_map = []
        self._delta_register = [1.0 for _ in self._delta_register]


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
    new_dag.add_nodes_from(range(dag.number_of_nodes()))
    transitive_reduction = nx.transitive_reduction(dag)
    new_dag.add_edges_from(transitive_reduction.edges)

    return new_dag
