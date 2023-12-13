import random
from copy import deepcopy

import networkx as nx
import numpy as np

from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit
from qibo.transpiler.abstract import Router
from qibo.transpiler.blocks import Block, CircuitBlocks
from qibo.transpiler.exceptions import ConnectivityError


def assert_connectivity(connectivity: nx.Graph, circuit: Circuit):
    """Assert if a circuit can be executed on Hardware.

    No gates acting on more than two qubits.
    All two-qubit operations can be performed on hardware.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to check.
        connectivity (:class:`networkx.Graph`): chip connectivity.
    """

    for gate in circuit.queue:
        if len(gate.qubits) > 2 and not isinstance(gate, gates.M):
            raise_error(ConnectivityError, f"{gate.name} acts on more than two qubits.")
        if len(gate.qubits) == 2:
            if (gate.qubits[0], gate.qubits[1]) not in connectivity.edges:
                raise_error(
                    ConnectivityError,
                    f"Circuit does not respect connectivity. {gate.name} acts on {gate.qubits}.",
                )


# TODO: make this class work with CircuitMap
class ShortestPaths(Router):
    """A class to perform initial qubit mapping and connectivity matching.

    Args:
        connectivity (:class:`networkx.Graph`): chip connectivity.
        sampling_split (float, optional): fraction of paths tested
            (between :math:`0` and :math:`1`). Defaults to :math:`1.0`.
        verbose (bool, optional): If ``True``, print info messages. Defaults to ``False``.
    """

    def __init__(
        self, connectivity: nx.Graph, sampling_split: float = 1.0, verbose: bool = False
    ):
        self.connectivity = connectivity
        self.sampling_split = sampling_split
        self.verbose = verbose

        self.initial_layout = None
        self._added_swaps = 0
        self.final_map = None
        self._gates_qubits_pairs = None
        self._mapping = None
        self._swap_map = None
        self._added_swaps_list = []
        self._graph = None
        self._qubit_map = None
        self._transpiled_circuit = None
        self._circuit_position = 0

    def __call__(self, circuit: Circuit, initial_layout: dict):
        """Circuit connectivity matching.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be matched to hardware connectivity.
            initial_layout (dict): initial physical-to-logical qubit mapping

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): circut mapped to hardware topology, and final qubit mapping.
        """
        self._mapping = initial_layout
        init_qubit_map = np.asarray(list(initial_layout.values()))
        self._initial_checks(circuit.nqubits)
        self._gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        self._mapping = dict(zip(range(len(initial_layout)), initial_layout.values()))
        self._graph = nx.relabel_nodes(self.connectivity, self._mapping)
        self._qubit_map = np.sort(init_qubit_map)
        self._swap_map = deepcopy(init_qubit_map)
        self._first_transpiler_step(circuit)

        while len(self._gates_qubits_pairs) != 0:
            self._transpiler_step(circuit)
        hardware_mapped_circuit = self._remap_circuit(np.argsort(init_qubit_map))
        final_mapping = {
            "q" + str(j): self._swap_map[j]
            for j in range(self._graph.number_of_nodes())
        }

        return hardware_mapped_circuit, final_mapping

    @property
    def added_swaps(self):
        """Number of added swaps during transpiling."""
        return self._added_swaps

    @property
    def sampling_split(self):
        """Fraction of possible shortest paths to be analyzed."""
        return self._sampling_split

    @sampling_split.setter
    def sampling_split(self, sampling_split: float):
        """Set the sampling split, the fraction of possible shortest paths to be analyzed.

        Args:
            sampling_split (float): define fraction of shortest path tested.
        """

        if 0.0 < sampling_split <= 1.0:
            self._sampling_split = sampling_split
        else:
            raise_error(ValueError, "Sampling_split must be in (0:1].")

    def _transpiler_step(self, circuit: Circuit):
        """Transpilation step. Find new mapping, add swap gates and apply gates that can be run with this configuration.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.
        """
        len_before_step = len(self._gates_qubits_pairs)
        path, meeting_point = self._relocate()
        self._add_swaps(path, meeting_point)
        self._update_qubit_map()
        self._add_gates(circuit, len_before_step - len(self._gates_qubits_pairs))

    def _first_transpiler_step(self, circuit: Circuit):
        """First transpilation step. Apply gates that can be run with the initial qubit mapping.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.
        """
        self._circuit_position = 0
        self._added_swaps = 0
        self._added_swaps_list = []
        len_2q_circuit = len(self._gates_qubits_pairs)
        self._gates_qubits_pairs = self._reduce(self._graph)
        self._add_gates(circuit, len_2q_circuit - len(self._gates_qubits_pairs))

    def _reduce(self, graph: nx.Graph):
        """Reduces the circuit by deleting two-qubit gates if it can be applied on the current configuration.

        Args:
            graph (:class:`networkx.Graph`): current hardware qubit mapping.

        Returns:
            (list): reduced circuit.
        """
        new_circuit = self._gates_qubits_pairs.copy()
        while (
            new_circuit != []
            and (new_circuit[0][0], new_circuit[0][1]) in graph.edges()
        ):
            del new_circuit[0]
        return new_circuit

    def _map_list(self, path: list):
        """Return all possible walks of qubits, or a fraction, for a given path.

        Args:
            path (list): path to move qubits.

        Returns:
            (list, list): all possible walks of qubits, or a fraction of them based on self.sampling_split, for a given path, and qubit meeting point for each path.
        """
        path_ends = [path[0], path[-1]]
        path_middle = path[1:-1]
        mapping_list = []
        meeting_point_list = []
        test_paths = range(len(path) - 1)
        if self.sampling_split != 1.0:
            test_paths = np.random.choice(
                test_paths,
                size=int(np.ceil(len(test_paths) * self.sampling_split)),
                replace=False,
            )
        for i in test_paths:
            values = path_middle[:i] + path_ends + path_middle[i:]
            mapping = dict(zip(path, values))
            mapping_list.append(mapping)
            meeting_point_list.append(i)

        return mapping_list, meeting_point_list

    def _relocate(self):
        """Greedy algorithm to decide which path to take, and how qubits should walk.

        Returns:
            (list, int): best path to move qubits and qubit meeting point in the path.
        """
        nodes = self._graph.number_of_nodes()
        circuit = self._reduce(self._graph)
        final_circuit = circuit
        keys = list(range(nodes))
        final_graph = self._graph
        final_mapping = dict(zip(keys, keys))
        # Consider all shortest paths
        path_list = [
            p
            for p in nx.all_shortest_paths(
                self._graph, source=circuit[0][0], target=circuit[0][1]
            )
        ]
        self._added_swaps += len(path_list[0]) - 2
        # Here test all paths
        for path in path_list:
            # map_list uses self.sampling_split
            list_, meeting_point_list = self._map_list(path)
            for j, mapping in enumerate(list_):
                new_graph = nx.relabel_nodes(self._graph, mapping)
                new_circuit = self._reduce(new_graph)
                # Greedy looking for the optimal path and the optimal walk on this path
                if len(new_circuit) < len(final_circuit):
                    final_graph = new_graph
                    final_circuit = new_circuit
                    final_mapping = mapping
                    final_path = path
                    meeting_point = meeting_point_list[j]
        self._graph = final_graph
        self._mapping = final_mapping
        self._gates_qubits_pairs = final_circuit

        return final_path, meeting_point

    def _initial_checks(self, qubits: int):
        """Initializes the transpiled circuit and check if it can be mapped to the defined connectivity.

        Args:
            qubits (int): number of qubits in the circuit to be transpiled.
        """
        nodes = self.connectivity.number_of_nodes()
        if qubits > nodes:
            raise_error(
                ValueError,
                "There are not enough physical qubits in the hardware to map the circuit.",
            )
        if qubits == nodes:
            new_circuit = Circuit(nodes)
        else:
            if self.verbose:
                log.info(
                    "You are using more physical qubits than required by the circuit, some ancillary qubits will be added to the circuit."
                )
            new_circuit = Circuit(nodes)
        self._transpiled_circuit = new_circuit

    def _add_gates(self, circuit: Circuit, matched_gates: int):
        """Adds one and two qubit gates to transpiled circuit until connectivity is matched.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.
            matched_gates (int): number of two-qubit gates that
                can be applied with the current qubit mapping.
        """
        index = 0
        while self._circuit_position < len(circuit.queue):
            gate = circuit.queue[self._circuit_position]
            if isinstance(gate, gates.M):
                measured_qubits = gate.qubits
                self._transpiled_circuit.add(
                    gate.on_qubits(
                        {
                            measured_qubits[i]: self._qubit_map[measured_qubits[i]]
                            for i in range(len(measured_qubits))
                        }
                    )
                )
                self._circuit_position += 1
            elif len(gate.qubits) == 1:
                self._transpiled_circuit.add(
                    gate.on_qubits({gate.qubits[0]: self._qubit_map[gate.qubits[0]]})
                )
                self._circuit_position += 1
            else:
                index += 1
                if index == matched_gates + 1:
                    break
                self._transpiled_circuit.add(
                    gate.on_qubits(
                        {
                            gate.qubits[0]: self._qubit_map[gate.qubits[0]],
                            gate.qubits[1]: self._qubit_map[gate.qubits[1]],
                        }
                    )
                )
                self._circuit_position += 1

    def _add_swaps(self, path: list, meeting_point: int):
        """Adds swaps to the transpiled circuit to move qubits.

        Args:
            path (list): path to move qubits.
            meeting_point (int): qubit meeting point in the path.
        """
        forward = path[0 : meeting_point + 1]
        backward = list(reversed(path[meeting_point + 1 :]))
        if len(forward) > 1:
            for f1, f2 in zip(forward[:-1], forward[1:]):
                gate = gates.SWAP(self._qubit_map[f1], self._qubit_map[f2])
                self._transpiled_circuit.add(gate)
                self._added_swaps_list.append(gate)

        if len(backward) > 1:
            for b1, b2 in zip(backward[:-1], backward[1:]):
                gate = gates.SWAP(self._qubit_map[b1], self._qubit_map[b2])
                self._transpiled_circuit.add(gate)
                self._added_swaps_list.append(gate)

    def _update_swap_map(self, swap: tuple):
        """Updates the qubit swap map."""
        temp = self._swap_map[swap[0]]
        self._swap_map[swap[0]] = self._swap_map[swap[1]]
        self._swap_map[swap[1]] = temp

    def _update_qubit_map(self):
        """Update the qubit mapping after adding swaps."""
        old_mapping = self._qubit_map.copy()
        for key, value in self._mapping.items():
            self._qubit_map[value] = old_mapping[key]

    def _remap_circuit(self, qubit_map):
        """Map logical to physical qubits in a circuit.

        Args:
            qubit_map (ndarray): new qubit mapping.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): transpiled circuit mapped with initial qubit mapping.
        """
        new_circuit = Circuit(self._transpiled_circuit.nqubits)
        for gate in self._transpiled_circuit.queue:
            new_circuit.add(gate.on_qubits({q: qubit_map[q] for q in gate.qubits}))
            if gate in self._added_swaps_list:
                self._update_swap_map(
                    tuple(qubit_map[gate.qubits[i]] for i in range(2))
                )
        return new_circuit


def _find_gates_qubits_pairs(circuit: Circuit):
    """Helper method for :meth:`qibo.transpiler.router.ShortestPaths`.
    Translate qibo circuit into a list of pairs of qubits to be used by the router and placer.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

    Returns:
        (list): list containing qubits targeted by two qubit gates.
    """
    translated_circuit = []
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            pass
        elif len(gate.qubits) == 2:
            translated_circuit.append(sorted(gate.qubits))
        elif len(gate.qubits) >= 3:
            raise_error(
                ValueError, "Gates targeting more than 2 qubits are not supported"
            )

    return translated_circuit


class CircuitMap:
    """Class to keep track of the circuit and physical-logical mapping during routing,
    this class also implements the initial two qubit blocks decomposition.

    Args:
        initial_layout (dict): initial logical-to-physical qubit mapping.
        circuit (Circuit): circuit to be routed.
        blocks (CircuitBlocks): circuit blocks representation, if None the blocks will be computed from the circuit.
    """

    def __init__(self, initial_layout: dict, circuit: Circuit, blocks=None):
        if blocks is not None:
            self.circuit_blocks = blocks
        else:
            self.circuit_blocks = CircuitBlocks(circuit, index_names=True)
        self.initial_layout = initial_layout
        self._circuit_logical = list(range(len(initial_layout)))
        self._physical_logical = list(initial_layout.values())
        self._routed_blocks = CircuitBlocks(Circuit(circuit.nqubits))
        self._swaps = 0

    def set_circuit_logical(self, circuit_logical_map: list):
        """Set the current circuit to logical qubit mapping."""
        self._circuit_logical = circuit_logical_map

    def blocks_qubits_pairs(self):
        """Returns a list containing the qubit pairs of each block."""
        return [block.qubits for block in self.circuit_blocks()]

    def execute_block(self, block: Block):
        """Executes a block by removing it from the circuit representation
        and adding it to the routed circuit.
        """
        self._routed_blocks.add_block(block.on_qubits(self.get_physical_qubits(block)))
        self.circuit_blocks.remove_block(block)

    def routed_circuit(self, circuit_kwargs=None):
        """Return the routed circuit.

        Args:
            circuit_kwargs (dict): original circuit init_kwargs.
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
        """Updates the logical-physical qubit mapping after applying a SWAP
        and add the SWAP gate to the routed blocks, the swap is represented by a tuple containing
        the logical qubits to be swapped.
        """
        physical_swap = self.logical_to_physical(swap)
        self._routed_blocks.add_block(
            Block(qubits=physical_swap, gates=[gates.SWAP(*physical_swap)])
        )
        self._swaps += 1
        idx_0, idx_1 = self._circuit_logical.index(
            swap[0]
        ), self._circuit_logical.index(swap[1])
        self._circuit_logical[idx_0], self._circuit_logical[idx_1] = swap[1], swap[0]

    def get_logical_qubits(self, block: Block):
        """Returns the current logical qubits where a block is acting"""
        return self.circuit_to_logical(block.qubits)

    def get_physical_qubits(self, block: Block or int):
        """Returns the physical qubits where a block is acting."""
        if isinstance(block, int):
            block = self.circuit_blocks.search_by_index(block)
        return self.logical_to_physical(self.get_logical_qubits(block))

    def logical_to_physical(self, logical_qubits: tuple):
        """Returns the physical qubits associated to the logical qubits."""
        return tuple(self._physical_logical.index(logical_qubits[i]) for i in range(2))

    def circuit_to_logical(self, circuit_qubits: tuple):
        """Returns the current logical qubits associated to the initial circuit qubits."""
        return tuple(self._circuit_logical[circuit_qubits[i]] for i in range(2))

    def circuit_to_physical(self, circuit_qubit: int):
        """Returns the current physical qubit associated to an initial circuit qubit."""
        return self._physical_logical.index(self._circuit_logical[circuit_qubit])


class Sabre(Router):
    def __init__(
        self,
        connectivity: nx.Graph,
        lookahead: int = 2,
        decay_lookahead: float = 0.6,
        delta: float = 0.001,
        seed=None,
    ):
        """Routing algorithm proposed in Ref [1].

        Args:
            connectivity (dict): hardware chip connectivity.
            lookahead (int): lookahead factor, how many dag layers will be considered in computing the cost.
            decay_lookahead (float): value in interval [0,1].
                How the weight of the distance in the dag layers decays in computing the cost.
            delta (float): this parameter defines the number of swaps vs depth trade-off by deciding
                how the algorithm tends to select non-overlapping SWAPs.
            seed (int): seed for the candidate random choice as tiebraker.

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

        routed_circuit = self.circuit.routed_circuit(circuit_kwargs=circuit.init_kwargs)
        if self._final_measurements is not None:
            routed_circuit = self._append_final_measurements(
                routed_circuit=routed_circuit
            )

        return routed_circuit, self.circuit.final_layout()

    @property
    def added_swaps(self):
        """Number of SWAP gates added to the circuit during routing."""
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
        """
        copy_circuit = self._copy_circuit(circuit)
        self._final_measurements = self._detach_final_measurements(copy_circuit)
        self.circuit = CircuitMap(initial_layout, copy_circuit)
        self._dist_matrix = nx.floyd_warshall_numpy(self.connectivity)
        self._dag = _create_dag(self.circuit.blocks_qubits_pairs())
        self._memory_map = []
        self._update_dag_layers()
        self._update_front_layer()
        self._delta_register = [1.0 for _ in range(circuit.nqubits)]

    @staticmethod
    def _copy_circuit(circuit: Circuit):
        """Return a copy of the circuit to avoid altering the original circuit.
        This copy conserves the registers of the measurement gates."""
        new_circuit = Circuit(circuit.nqubits)
        for gate in circuit.queue:
            new_circuit.add(gate)
        return new_circuit

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
        """Append the final measurment gates on the correct qubits conserving the measurement register."""
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
        """Update dag layers and put them in topological order."""
        for layer, nodes in enumerate(nx.topological_generations(self._dag)):
            for node in nodes:
                self._dag.nodes[node]["layer"] = layer

    def _update_front_layer(self):
        """Update the front layer of the dag."""
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
        for qubit in self.circuit.logical_to_physical(best_candidate):
            self._delta_register[qubit] += self.delta
        self.circuit.update(best_candidate)

    def _compute_cost(self, candidate):
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
                qubits = temporary_circuit.get_physical_qubits(gate)
                avg_layer_distance += (
                    max(self._delta_register[i] for i in qubits)
                    * (self._dist_matrix[qubits[0], qubits[1]] - 1.0)
                    / len(layer_gates)
                )
            tot_distance += weight * avg_layer_distance
            weight *= self.decay
        return tot_distance

    def _swap_candidates(self):
        """Return a list of possible candidate SWAPs (to be applied on logical qubits directly).
        The possible candidates are the ones sharing at least one qubit with a block in the front layer.
        """
        candidates = []
        for block in self._front_layer:
            for qubit in self.circuit.get_physical_qubits(block):
                for connected in self.connectivity.neighbors(qubit):
                    candidate = tuple(
                        sorted(
                            (
                                self.circuit._physical_logical[qubit],
                                self.circuit._physical_logical[connected],
                            )
                        )
                    )
                    if candidate not in candidates:
                        candidates.append(candidate)
        return candidates

    def _check_execution(self):
        """Check if some gatesblocks in the front layer can be executed in the current configuration.

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
        """Execute a list of blocks:
        -Remove the correspondent nodes from the dag and circuit representation.
        -Add the executed blocks to the routed circuit.
        -Update the dag layers and front layer.
        -Reset the mapping memory.
        """
        for block_id in blocklist:
            block = self.circuit.circuit_blocks.search_by_index(block_id)
            self.circuit.execute_block(block)
            self._dag.remove_node(block_id)
        self._update_dag_layers()
        self._update_front_layer()
        self._memory_map = []
        self._delta_register = [1.0 for _ in self._delta_register]


def _create_dag(gates_qubits_pairs):
    """Helper method for :meth:`qibo.transpiler.router.Sabre`.
    Create direct acyclic graph (dag) of the circuit based on two qubit gates commutativity relations.

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


def _remove_redundant_connections(dag: nx.Graph):
    """Remove redundant connection from a DAG unsing transitive reduction."""
    new_dag = nx.DiGraph()
    new_dag.add_nodes_from(range(dag.number_of_nodes()))
    transitive_reduction = nx.transitive_reduction(dag)
    new_dag.add_edges_from(transitive_reduction.edges)
    return new_dag
