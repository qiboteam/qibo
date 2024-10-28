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
)
from qibo.transpiler.router import ShortestPaths
from qibo.transpiler.utils import assert_mapping_consistency, assert_placement


def star_circuit(names=["q0", "q1", "q2", "q3", "q4"]):
    circuit = Circuit(5, wire_names=names)
    for i in range(1, 5):
        circuit.add(gates.CNOT(i, 0))
    return circuit


def test_assert_placement_true(star_connectivity):
    circuit = Circuit(5)
    assert_placement(circuit, connectivity=star_connectivity())


@pytest.mark.parametrize(
    "qubits, names", [(5, ["A", "B", "C", "D", "F"]), (3, ["A", "B", "C"])]
)
def test_assert_placement_false(qubits, names, star_connectivity):
    connectivity = star_connectivity()
    circuit = Circuit(qubits, wire_names=names)
    with pytest.raises(PlacementError):
        assert_placement(circuit, connectivity)


@pytest.mark.parametrize("qubits", [10, 1])
def test_assert_placement_error(qubits, star_connectivity):
    connectivity = star_connectivity()
    circuit = Circuit(qubits)
    with pytest.raises(PlacementError):
        assert_placement(circuit, connectivity)


@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_mapping_consistency(names, star_connectivity):
    assert_mapping_consistency(names, star_connectivity(names))


def test_mapping_consistency_error(star_connectivity):
    with pytest.raises(PlacementError):
        assert_mapping_consistency(["A", "B", "C", "D", "F"], star_connectivity())


@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_mapping_consistency_restricted(names, star_connectivity):
    connectivity = star_connectivity(names)
    on_qubit = [names[0], names[2]]
    restricted_connectivity = restrict_connectivity_qubits(connectivity, on_qubit)
    assert_mapping_consistency(on_qubit, restricted_connectivity)


@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_mapping_consistency_restricted_error(names, star_connectivity):
    connectivity = star_connectivity(names)
    on_qubit = [names[0], names[2]]
    restricted_connectivity = restrict_connectivity_qubits(connectivity, on_qubit)
    with pytest.raises(PlacementError):
        assert_mapping_consistency([names[3], names[4]], restricted_connectivity)


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


def test_trivial(star_connectivity):
    names = ["q4", "q3", "q2", "q1", "q0"]
    circuit = Circuit(5, wire_names=names)
    connectivity = star_connectivity(names)
    placer = Trivial(connectivity=connectivity)
    placer(circuit)
    assert circuit.wire_names == names
    assert_placement(circuit, connectivity)


def test_trivial_restricted(star_connectivity):
    names = ["q0", "q2"]
    circuit = Circuit(2, wire_names=names)
    connectivity = star_connectivity(["q0", "q1", "q2", "q3", "q4"])
    restricted_connectivity = restrict_connectivity_qubits(connectivity, names)
    placer = Trivial(connectivity=restricted_connectivity)
    placer(circuit)
    assert circuit.wire_names == names
    assert_placement(circuit, restricted_connectivity)


@pytest.mark.parametrize(
    "custom_layout",
    [["E", "D", "C", "B", "A"], {"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}],
)
def test_custom(custom_layout, star_connectivity):
    circuit = Circuit(5)
    connectivity = star_connectivity(["A", "B", "C", "D", "E"])
    placer = Custom(connectivity=connectivity, initial_map=custom_layout)
    placer(circuit)
    assert circuit.wire_names == ["E", "D", "C", "B", "A"]


@pytest.mark.parametrize(
    "custom_layout", [[4, 3, 2, 1, 0], {4: 0, 3: 1, 2: 2, 1: 3, 0: 4}]
)
def test_custom_int(custom_layout, star_connectivity):
    names = [0, 1, 2, 3, 4]
    circuit = Circuit(5, wire_names=names)
    connectivity = star_connectivity(names)
    placer = Custom(connectivity=connectivity, initial_map=custom_layout)
    placer(circuit)
    assert circuit.wire_names == [4, 3, 2, 1, 0]


@pytest.mark.parametrize("custom_layout", [["D", "C"], {"C": 1, "D": 0}])
def test_custom_restricted(custom_layout, star_connectivity):
    circuit = Circuit(2, wire_names=["C", "D"])
    connectivity = star_connectivity(["A", "B", "C", "D", "E"])
    restricted_connectivity = restrict_connectivity_qubits(connectivity, ["C", "D"])
    placer = Custom(connectivity=restricted_connectivity, initial_map=custom_layout)
    placer(circuit)
    assert circuit.wire_names == ["D", "C"]
    assert_placement(circuit, restricted_connectivity)


def test_custom_error_circuit(star_connectivity):
    circuit = Circuit(3)
    custom_layout = [4, 3, 2, 1, 0]
    connectivity = star_connectivity(names=custom_layout)
    placer = Custom(connectivity=connectivity, initial_map=custom_layout)
    with pytest.raises(ValueError):
        placer(circuit)


def test_custom_error_type(star_connectivity):
    circuit = Circuit(5)
    connectivity = star_connectivity()
    placer = Custom(connectivity=connectivity, initial_map=1)
    with pytest.raises(TypeError):
        placer(circuit)


def test_subgraph_perfect(star_connectivity):
    connectivity = star_connectivity()
    placer = Subgraph(connectivity=connectivity)
    circuit = star_circuit()
    placer(circuit)
    assert circuit.wire_names[0] == 2
    assert_placement(circuit, connectivity)


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


def test_subgraph_non_perfect(star_connectivity):
    connectivity = star_connectivity()
    placer = Subgraph(connectivity=connectivity)
    circuit = imperfect_circuit()
    placer(circuit)
    assert_placement(circuit, connectivity)


def test_subgraph_error(star_connectivity):
    connectivity = star_connectivity()
    placer = Subgraph(connectivity=connectivity)
    circuit = Circuit(5)
    with pytest.raises(ValueError):
        placer(circuit)


def test_subgraph_restricted(star_connectivity):
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
    placer(circuit)
    assert_placement(circuit, restricted_connectivity)


@pytest.mark.parametrize("reps", [1, 10, 100])
@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_random(reps, names, star_connectivity):
    connectivity = star_connectivity(names)
    placer = Random(connectivity=connectivity, samples=reps)
    circuit = star_circuit(names=names)
    placer(circuit)
    assert_placement(circuit, connectivity)


def test_random_restricted(star_connectivity):
    names = [0, 1, 2, 3, 4]
    circuit = Circuit(4, wire_names=names[:4])
    circuit.add(gates.CNOT(1, 3))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(3, 2))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(3, 1))
    connectivity = star_connectivity(names)
    restricted_connectivity = restrict_connectivity_qubits(connectivity, [0, 2, 3, 4])
    placer = Random(connectivity=restricted_connectivity, samples=100)
    placer(circuit)
    assert_placement(circuit, restricted_connectivity)


@pytest.mark.parametrize("ngates", [None, 5, 13])
@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_reverse_traversal(ngates, names, star_connectivity):
    circuit = star_circuit(names=names)
    connectivity = star_connectivity(names=names)
    routing = ShortestPaths(connectivity=connectivity)
    placer = ReverseTraversal(connectivity, routing, depth=ngates)
    placer(circuit)
    assert_placement(circuit, connectivity)


def test_reverse_traversal_no_gates(star_connectivity):
    connectivity = star_connectivity()
    routing = ShortestPaths(connectivity=connectivity)
    placer = ReverseTraversal(connectivity, routing, depth=10)
    circuit = Circuit(5)
    with pytest.raises(ValueError):
        placer(circuit)


def test_reverse_traversal_restricted(star_connectivity):
    circuit = Circuit(4)
    circuit.add(gates.CNOT(1, 3))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(3, 2))
    circuit.add(gates.CNOT(2, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(3, 1))
    connectivity = star_connectivity()
    restrict_names = [0, 2, 3, 4]
    restricted_connectivity = restrict_connectivity_qubits(connectivity, restrict_names)
    circuit.wire_names = restrict_names
    routing = ShortestPaths(connectivity=restricted_connectivity)
    placer = ReverseTraversal(
        connectivity=restricted_connectivity, routing_algorithm=routing, depth=5
    )
    placer(circuit)
    assert_placement(circuit, restricted_connectivity)


def test_star_connectivity_placer(star_connectivity):
    circ = Circuit(5)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.CZ(0, 2))
    connectivity = star_connectivity()
    placer = StarConnectivityPlacer(connectivity)
    placer(circ)
    assert_placement(circ, connectivity)
    assert circ.wire_names == [0, 2, 1, 3, 4]


@pytest.mark.parametrize("first", [True, False])
def test_star_connectivity_placer_error(first, star_connectivity):
    circ = Circuit(5)
    if first:
        circ.add(gates.CZ(0, 1))
    circ.add(gates.TOFFOLI(0, 1, 2))
    connectivity = star_connectivity()
    placer = StarConnectivityPlacer(connectivity)
    with pytest.raises(PlacementError):
        placer(circ)

    chip = nx.Graph()
    chip.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    with pytest.raises(ValueError):
        StarConnectivityPlacer(chip)
