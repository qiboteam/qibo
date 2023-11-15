import networkx as nx
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.placer import (
    Custom,
    PlacementError,
    Random,
    ReverseTraversal,
    Subgraph,
    Trivial,
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


def test_mapping_consistency_true():
    layout = {"q0": 0, "q1": 2, "q2": 1, "q3": 4, "q4": 3}
    assert_mapping_consistency(layout)


@pytest.mark.parametrize(
    "layout",
    [
        {"q0": 0, "q1": 0, "q2": 1, "q3": 4, "q4": 3},
        {"q0": 0, "q1": 2, "q0": 1, "q3": 4, "q4": 3},
    ],
)
def test_mapping_consistency_false(layout):
    with pytest.raises(PlacementError):
        assert_mapping_consistency(layout)


def test_trivial():
    circuit = Circuit(5)
    connectivity = star_connectivity()
    placer = Trivial(connectivity=connectivity)
    layout = placer(circuit)
    assert layout == {"q0": 0, "q1": 1, "q2": 2, "q3": 3, "q4": 4}
    assert_placement(circuit, layout)


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
def test_custom(custom_layout, give_circuit):
    if give_circuit:
        circuit = Circuit(5)
    else:
        circuit = None
    connectivity = star_connectivity()
    placer = Custom(connectivity=connectivity, map=custom_layout)
    layout = placer(circuit)
    assert layout == {"q0": 4, "q1": 3, "q2": 2, "q3": 1, "q4": 0}


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
