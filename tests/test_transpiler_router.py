import networkx as nx
import numpy as np
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler._exceptions import ConnectivityError
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import (
    assert_circuit_equivalence,
    restrict_connectivity_qubits,
)
from qibo.transpiler.placer import Custom, Random, Subgraph, Trivial, assert_placement
from qibo.transpiler.router import CircuitMap, Sabre, ShortestPaths, assert_connectivity


def star_connectivity():
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


def grid_connectivity():
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[0], Q[1]), (Q[1], Q[2]), (Q[2], Q[3]), (Q[3], Q[0]), (Q[0], Q[4])]
    chip.add_edges_from(graph_list)
    return chip


def generate_random_circuit(nqubits, ngates, seed=42):
    """Generate a random circuit with RX and CZ gates."""
    np.random.seed(seed)
    one_qubit_gates = [gates.RX, gates.RY, gates.RZ]
    two_qubit_gates = [gates.CZ, gates.CNOT, gates.SWAP]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta, trainable=False))
        else:
            circuit.add(gate(*q))
    return circuit


def star_circuit():
    circuit = Circuit(5)
    for i in range(1, 5):
        circuit.add(gates.CNOT(i, 0))
    return circuit


def matched_circuit():
    """Return a simple circuit that can be executed on star connectivity"""
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.Z(1))
    circuit.add(gates.CZ(2, 1))
    circuit.add(gates.M(0))
    return circuit


def test_assert_connectivity():
    assert_connectivity(star_connectivity(), matched_circuit())


def test_assert_connectivity_false():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)


def test_assert_connectivity_3q():
    circuit = Circuit(5)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)


@pytest.mark.parametrize("gates", [5, 25])
@pytest.mark.parametrize("placer", [Trivial, Random])
@pytest.mark.parametrize("connectivity", [star_connectivity(), grid_connectivity()])
def test_random_circuits_5q(gates, placer, connectivity):
    placer = placer(connectivity=connectivity)
    layout_circ = Circuit(5)
    initial_layout = placer(layout_circ)
    transpiler = ShortestPaths(connectivity=connectivity)
    circuit = generate_random_circuit(nqubits=5, ngates=gates)
    transpiled_circuit, final_qubit_map = transpiler(circuit, initial_layout)
    assert transpiler.added_swaps >= 0
    assert_connectivity(connectivity, transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert gates + transpiler.added_swaps == transpiled_circuit.ngates
    qubit_matcher = Preprocessing(connectivity=connectivity)
    new_circuit = qubit_matcher(circuit=circuit)
    assert_circuit_equivalence(
        original_circuit=new_circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
        initial_map=initial_layout,
    )


def test_star_circuit():
    placer = Subgraph(star_connectivity())
    initial_layout = placer(star_circuit())
    transpiler = ShortestPaths(connectivity=star_connectivity())
    transpiled_circuit, final_qubit_map = transpiler(star_circuit(), initial_layout)
    assert transpiler.added_swaps == 0
    assert_connectivity(star_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert_circuit_equivalence(
        original_circuit=star_circuit(),
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
        initial_map=initial_layout,
    )


def test_star_circuit_custom_map():
    placer = Custom(map=[1, 0, 2, 3, 4], connectivity=star_connectivity())
    initial_layout = placer()
    transpiler = ShortestPaths(connectivity=star_connectivity())
    transpiled_circuit, final_qubit_map = transpiler(star_circuit(), initial_layout)
    assert transpiler.added_swaps == 1
    assert_connectivity(star_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert_circuit_equivalence(
        original_circuit=star_circuit(),
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
        initial_map=initial_layout,
    )


def test_routing_with_measurements():
    placer = Trivial(connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 2, 3))
    initial_layout = placer(circuit=circuit)
    transpiler = ShortestPaths(connectivity=star_connectivity())
    transpiled_circuit, final_qubit_map = transpiler(circuit, initial_layout)
    assert transpiled_circuit.ngates == 3
    measured_qubits = transpiled_circuit.queue[2].qubits
    assert measured_qubits == (0, 1, 3)
    assert_circuit_equivalence(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
        initial_map=initial_layout,
    )


def test_circuit_map():
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(2, 3))
    initial_layout = {"q0": 2, "q1": 0, "q2": 1, "q3": 3}
    circuit_map = CircuitMap(initial_layout=initial_layout, circuit=circ)
    block_list = circuit_map.circuit_blocks
    # test blocks_qubits_pairs
    assert circuit_map.blocks_qubits_pairs() == [(0, 1), (1, 2), (0, 1), (2, 3)]
    # test execute_block and routed_circuit
    circuit_map.execute_block(block_list.search_by_index(0))
    routed_circuit = circuit_map.routed_circuit()
    assert isinstance(routed_circuit.queue[0], gates.H)
    assert len(routed_circuit.queue) == 4
    assert routed_circuit.queue[2].qubits == (1, 2)
    # test update
    circuit_map.update((0, 2))
    routed_circuit = circuit_map.routed_circuit()
    assert isinstance(routed_circuit.queue[4], gates.SWAP)
    assert routed_circuit.queue[4].qubits == (1, 0)
    assert circuit_map._swaps == 1
    assert circuit_map._circuit_logical == [2, 1, 0, 3]
    circuit_map.update((1, 2))
    routed_circuit = circuit_map.routed_circuit()
    assert routed_circuit.queue[5].qubits == (2, 0)
    assert circuit_map._circuit_logical == [1, 2, 0, 3]
    # test execute_block after multiple swaps
    circuit_map.execute_block(block_list.search_by_index(1))
    circuit_map.execute_block(block_list.search_by_index(2))
    circuit_map.execute_block(block_list.search_by_index(3))
    routed_circuit = circuit_map.routed_circuit()
    assert isinstance(routed_circuit.queue[6], gates.CZ)
    # circuit to logical map: [1,2,0,3]. initial map: {"q0": 2, "q1": 0, "q2": 1, "q3": 3}.
    assert routed_circuit.queue[6].qubits == (0, 1)  # initial circuit qubits (1,2)
    assert routed_circuit.queue[7].qubits == (2, 0)  # (0,1)
    assert routed_circuit.queue[8].qubits == (1, 3)  # (2,3)
    assert len(circuit_map.circuit_blocks()) == 0
    # test final layout
    assert circuit_map.final_layout() == {"q0": 1, "q1": 2, "q2": 0, "q3": 3}


def test_sabre_matched():
    placer = Trivial()
    layout_circ = Circuit(5)
    initial_layout = placer(layout_circ)
    router = Sabre(connectivity=star_connectivity())
    routed_circuit, final_map = router(
        circuit=matched_circuit(), initial_layout=initial_layout
    )
    assert router.added_swaps == 0
    assert final_map == {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
    assert_connectivity(circuit=routed_circuit, connectivity=star_connectivity())
    assert_circuit_equivalence(
        original_circuit=matched_circuit(),
        transpiled_circuit=routed_circuit,
        final_map=final_map,
        initial_map=initial_layout,
    )


@pytest.mark.parametrize("seed", [42])
def test_sabre_simple(seed):
    placer = Trivial()
    circ = Circuit(5)
    circ.add(gates.CZ(0, 1))
    initial_layout = placer(circ)
    router = Sabre(connectivity=star_connectivity(), seed=seed)
    routed_circuit, final_map = router(circuit=circ, initial_layout=initial_layout)
    assert router.added_swaps == 1
    assert final_map == {"q0": 2, "q1": 1, "q2": 0, "q3": 3, "q4": 4}
    assert routed_circuit.queue[0].qubits == (0, 2)
    assert isinstance(routed_circuit.queue[0], gates.SWAP)
    assert isinstance(routed_circuit.queue[1], gates.CZ)
    assert_connectivity(circuit=routed_circuit, connectivity=star_connectivity())
    assert_circuit_equivalence(
        original_circuit=circ,
        transpiled_circuit=routed_circuit,
        final_map=final_map,
        initial_map=initial_layout,
    )


@pytest.mark.parametrize("n_gates", [10, 40])
@pytest.mark.parametrize("look", [0, 5])
@pytest.mark.parametrize("decay", [0.5, 1.0])
@pytest.mark.parametrize("placer", [Trivial, Random])
@pytest.mark.parametrize("connectivity", [star_connectivity(), grid_connectivity()])
def test_sabre_random_circuits(n_gates, look, decay, placer, connectivity):
    placer = placer(connectivity=connectivity)
    layout_circ = Circuit(5)
    initial_layout = placer(layout_circ)
    router = Sabre(connectivity=connectivity, lookahead=look, decay_lookahead=decay)
    circuit = generate_random_circuit(nqubits=5, ngates=n_gates)
    measurement = gates.M(*range(5))
    circuit.add(measurement)
    transpiled_circuit, final_qubit_map = router(circuit, initial_layout)
    assert router.added_swaps >= 0
    assert_connectivity(connectivity, transpiled_circuit)
    assert_placement(transpiled_circuit, final_qubit_map)
    assert n_gates + router.added_swaps + 1 == transpiled_circuit.ngates
    assert_circuit_equivalence(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
        initial_map=initial_layout,
    )
    circuit_result = transpiled_circuit.execute(nshots=100)
    assert circuit_result.frequencies() == measurement.result.frequencies()
    assert transpiled_circuit.queue[-1].result is measurement.result


def test_sabre_memory_map():
    placer = Trivial()
    layout_circ = Circuit(5)
    initial_layout = placer(layout_circ)
    router = Sabre(connectivity=star_connectivity())
    router._preprocessing(circuit=star_circuit(), initial_layout=initial_layout)
    router._memory_map = [[1, 0, 2, 3, 4]]
    value = router._compute_cost((0, 1))
    assert value == float("inf")


def test_sabre_intermediate_measurements():
    measurement = gates.M(1)
    circ = Circuit(3, density_matrix=True)
    circ.add(gates.H(0))
    circ.add(measurement)
    circ.add(gates.CNOT(0, 2))
    connectivity = nx.Graph()
    connectivity.add_nodes_from([0, 1, 2])
    connectivity.add_edges_from([(0, 1), (1, 2)])
    router = Sabre(connectivity=connectivity)
    initial_layout = {"q0": 0, "q1": 1, "q2": 2}
    routed_circ, final_layout = router(circuit=circ, initial_layout=initial_layout)
    circuit_result = routed_circ.execute(nshots=100)
    assert routed_circ.queue[3].result is measurement.result


def test_sabre_restrict_qubits():
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 2))
    circ.add(gates.CZ(2, 1))
    initial_layout = {"q0": 0, "q2": 2, "q3": 1}
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2, 3])
    router = Sabre(connectivity=restricted_connectivity)
    routed_circ, final_layout = router(circuit=circ, initial_layout=initial_layout)
    assert_circuit_equivalence(
        original_circuit=circ,
        transpiled_circuit=routed_circ,
        final_map=final_layout,
        initial_map=initial_layout,
    )
    assert_connectivity(restricted_connectivity, routed_circ)
    assert_placement(routed_circ, final_layout, connectivity=restricted_connectivity)


def test_shortest_paths_restrict_qubits():
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 2))
    circ.add(gates.CZ(2, 1))
    initial_layout = {"q0": 0, "q2": 2, "q3": 1}
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2, 3])
    router = ShortestPaths(connectivity=restricted_connectivity)
    routed_circ, final_layout = router(circuit=circ, initial_layout=initial_layout)
    assert_circuit_equivalence(
        original_circuit=circ,
        transpiled_circuit=routed_circ,
        final_map=final_layout,
        initial_map=initial_layout,
    )
    assert_connectivity(restricted_connectivity, routed_circ)
    assert_placement(routed_circ, final_layout, connectivity=restricted_connectivity)
