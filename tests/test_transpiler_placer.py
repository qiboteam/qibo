import networkx as nx
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler._exceptions import PlacementError
from qibo.transpiler.pipeline import restrict_connectivity_qubits
from qibo.transpiler.placer import (
    Custom,
    Random,
    ReverseTraversal,
    StarConnectivityPlacer,
    Subgraph,
    Trivial,
    _find_gates_qubits_pairs,
    assert_mapping_consistency,
    assert_placement,
)
from qibo.transpiler.router import ShortestPaths


def star_connectivity():
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [
        (Q[0], Q[2]),
        (Q[1], Q[2]),
        (Q[3], Q[2]),
        (Q[4], Q[2]),
    ]
    chip.add_edges_from(graph_list)
    return chip


def star_circuit():
    circuit = Circuit(5)
    for i in range(1, 5):
        circuit.add(gates.CNOT(i, 0))
    return circuit


def test_assert_placement_true():
    layout = {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
    circuit = Circuit(5)
    assert_placement(circuit, layout)


@pytest.mark.parametrize("qubits", [5, 3])
@pytest.mark.parametrize(
    "layout", [{"q0": 0, "q1": 1, "q2": 2, "q3": 3}, {"q0": 0, "q0": 1, "q2": 2}]
)
def test_assert_placement_false(qubits, layout):
    circuit = Circuit(qubits)
    with pytest.raises(PlacementError):
        assert_placement(circuit, layout)


def test_mapping_consistency():
    layout = {"q0": 0, "q1": 2, "q2": 1, "q3": 4, "q4": 3}
    assert_mapping_consistency(layout)


@pytest.mark.parametrize(
    "layout",
    [
        {"q0": 0, "q1": 0, "q2": 1, "q3": 4, "q4": 3},
        {"q0": 0, "q1": 2, "q0": 1, "q3": 4, "q4": 3},
    ],
)
def test_mapping_consistency_error(layout):
    with pytest.raises(PlacementError):
        assert_mapping_consistency(layout)


def test_mapping_consistency_restricted():
    layout = {"q0": 0, "q2": 1}
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2])
    assert_mapping_consistency(layout, restricted_connectivity)


@pytest.mark.parametrize(
    "layout",
    [
        {"q0": 0, "q2": 2},
        {"q0": 0, "q1": 1},
    ],
)
def test_mapping_consistency_restricted_error(layout):
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2])
    with pytest.raises(PlacementError):
        assert_mapping_consistency(layout, restricted_connectivity)


def test_gates_qubits_pairs():
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.M(1, 2))
    gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
    assert gates_qubits_pairs == [[0, 1], [1, 2]]


def test_gates_qubits_pairs_error():
    circuit = Circuit(5)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ValueError):
        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)


def test_trivial():
    circuit = Circuit(5)
    connectivity = star_connectivity()
    placer = Trivial(connectivity=connectivity)
    layout = placer(circuit)
    assert layout == {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
    assert_placement(circuit, layout)


def test_trivial_restricted():
    circuit = Circuit(2)
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2])
    placer = Trivial(connectivity=restricted_connectivity)
    layout = placer(circuit)
    assert layout == {"q0": 0, "q2": 1}
    assert_placement(
        circuit=circuit, layout=layout, connectivity=restricted_connectivity
    )


def test_trivial_error():
    circuit = Circuit(4)
    connectivity = star_connectivity()
    placer = Trivial(connectivity=connectivity)
    with pytest.raises(PlacementError):
        layout = placer(circuit)


@pytest.mark.parametrize(
    "custom_layout", [[4, 3, 2, 1, 0], {"q0": 4, "q1": 3, "q2": 2, "q3": 1, "q4": 0}]
)
@pytest.mark.parametrize("give_circuit", [True, False])
@pytest.mark.parametrize("give_connectivity", [True, False])
def test_custom(custom_layout, give_circuit, give_connectivity):
    if give_circuit:
        circuit = Circuit(5)
    else:
        circuit = None
    if give_connectivity:
        connectivity = star_connectivity()
    else:
        connectivity = None
    placer = Custom(connectivity=connectivity, map=custom_layout)
    layout = placer(circuit)
    assert layout == {"q0": 4, "q1": 3, "q2": 2, "q3": 1, "q4": 0}


@pytest.mark.parametrize("custom_layout", [[1, 0], {"q0": 1, "q2": 0}])
def test_custom_restricted(custom_layout):
    circuit = Circuit(2)
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2])
    placer = Custom(connectivity=restricted_connectivity, map=custom_layout)
    layout = placer(circuit)
    assert layout == {"q0": 1, "q2": 0}
    assert_placement(
        circuit=circuit, layout=layout, connectivity=restricted_connectivity
    )


def test_custom_error_circuit():
    circuit = Circuit(3)
    custom_layout = [4, 3, 2, 1, 0]
    connectivity = star_connectivity()
    placer = Custom(connectivity=connectivity, map=custom_layout)
    with pytest.raises(PlacementError):
        layout = placer(circuit)


def test_custom_error_no_circuit():
    connectivity = star_connectivity()
    custom_layout = {"q0": 4, "q1": 3, "q2": 2, "q3": 0, "q4": 0}
    placer = Custom(connectivity=connectivity, map=custom_layout)
    with pytest.raises(PlacementError):
        layout = placer()


def test_custom_error_type():
    circuit = Circuit(5)
    connectivity = star_connectivity()
    layout = 1
    placer = Custom(connectivity=connectivity, map=layout)
    with pytest.raises(TypeError):
        layout = placer(circuit)


def test_subgraph_perfect():
    connectivity = star_connectivity()
    placer = Subgraph(connectivity=connectivity)
    layout = placer(star_circuit())
    assert layout["q2"] == 0
    assert_placement(star_circuit(), layout)


def imperfect_circuit():
    circuit = Circuit(5)
    circuit.add(gates.CNOT(1, 3))
    circuit.add(gates.CNOT(2, 4))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(4, 3))
    circuit.add(gates.CNOT(3, 2))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(4, 3))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(3, 1))
    return circuit


def test_subgraph_non_perfect():
    connectivity = star_connectivity()
    placer = Subgraph(connectivity=connectivity)
    layout = placer(imperfect_circuit())
    assert_placement(imperfect_circuit(), layout)


def test_subgraph_error():
    connectivity = star_connectivity()
    placer = Subgraph(connectivity=connectivity)
    circuit = Circuit(5)
    with pytest.raises(ValueError):
        layout = placer(circuit)


def test_subgraph_restricted():
    circuit = Circuit(4)
    circuit.add(gates.CNOT(0, 3))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(3, 2))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(3, 1))
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2, 3, 4])
    placer = Subgraph(connectivity=restricted_connectivity)
    layout = placer(circuit)
    assert_placement(
        circuit=circuit, layout=layout, connectivity=restricted_connectivity
    )


@pytest.mark.parametrize("reps", [1, 10, 100])
def test_random(reps):
    connectivity = star_connectivity()
    placer = Random(connectivity=connectivity, samples=reps)
    layout = placer(star_circuit())
    assert_placement(star_circuit(), layout)


def test_random_perfect():
    circ = Circuit(5)
    circ.add(gates.CZ(0, 1))
    connectivity = star_connectivity()
    placer = Random(connectivity=connectivity, samples=1000)
    layout = placer(circ)
    assert_placement(star_circuit(), layout)


def test_random_restricted():
    circuit = Circuit(4)
    circuit.add(gates.CNOT(1, 3))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(3, 2))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(3, 1))
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2, 3, 4])
    placer = Random(connectivity=restricted_connectivity, samples=100)
    layout = placer(circuit)
    assert_placement(
        circuit=circuit, layout=layout, connectivity=restricted_connectivity
    )


@pytest.mark.parametrize("gates", [None, 5, 13])
def test_reverse_traversal(gates):
    circuit = star_circuit()
    connectivity = star_connectivity()
    routing = ShortestPaths(connectivity=connectivity)
    placer = ReverseTraversal(connectivity, routing, depth=gates)
    layout = placer(circuit)
    assert_placement(circuit, layout)


def test_reverse_traversal_no_gates():
    connectivity = star_connectivity()
    routing = ShortestPaths(connectivity=connectivity)
    placer = ReverseTraversal(connectivity, routing, depth=10)
    circuit = Circuit(5)
    with pytest.raises(ValueError):
        layout = placer(circuit)


def test_reverse_traversal_restricted():
    circuit = Circuit(4)
    circuit.add(gates.CNOT(1, 3))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(3, 2))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(3, 1))
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2, 3, 4])
    routing = ShortestPaths(connectivity=restricted_connectivity)
    placer = ReverseTraversal(
        connectivity=restricted_connectivity, routing_algorithm=routing, depth=5
    )
    layout = placer(circuit)
    assert_placement(
        circuit=circuit, layout=layout, connectivity=restricted_connectivity
    )


def test_star_connectivity_placer():
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.CZ(0, 2))
    placer = StarConnectivityPlacer(middle_qubit=2)
    layout = placer(circ)
    assert_placement(circ, layout)
    assert layout == {"q0": 0, "q1": 2, "q2": 1}


@pytest.mark.parametrize("first", [True, False])
def test_star_connectivity_placer_error(first):
    circ = Circuit(3)
    if first:
        circ.add(gates.CZ(0, 1))
    circ.add(gates.TOFFOLI(0, 1, 2))
    placer = StarConnectivityPlacer(middle_qubit=2)
    with pytest.raises(PlacementError):
        layout = placer(circ)
