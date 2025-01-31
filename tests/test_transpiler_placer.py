import networkx as nx
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler._exceptions import ConnectivityError, PlacementError
from qibo.transpiler.asserts import assert_placement
from qibo.transpiler.pipeline import restrict_connectivity_qubits
from qibo.transpiler.placer import (
    Random,
    ReverseTraversal,
    StarConnectivityPlacer,
    Subgraph,
    _find_gates_qubits_pairs,
)
from qibo.transpiler.router import ShortestPaths


def star_circuit(names=[0, 1, 2, 3, 4]):
    circuit = Circuit(5, wire_names=names)
    for i in range(1, 5):
        circuit.add(gates.CNOT(i, 0))
    return circuit


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
    circuit = Circuit(4, wire_names=[0, 2, 3, 4])
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


def test_subgraph_leasttwoqubitgates(star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 3))
    circuit.add(gates.CNOT(1, 2))
    connectivity = star_connectivity()
    placer = Subgraph(connectivity=connectivity)
    placer(circuit)
    assert_placement(circuit, connectivity)


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
    circuit = Circuit(4, wire_names=[0, 2, 3, 4])
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
    placer = ReverseTraversal(routing, connectivity, depth=ngates)
    placer(circuit)
    assert_placement(circuit, connectivity)


def test_reverse_traversal_no_gates(star_connectivity):
    connectivity = star_connectivity()
    routing = ShortestPaths(connectivity=connectivity)
    placer = ReverseTraversal(routing, connectivity, depth=10)
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
    with pytest.raises(ConnectivityError):
        placer = StarConnectivityPlacer(chip)
        placer(circ)


def test_star_connectivity_plus_disconnected_edges(star_connectivity):
    connectivity = star_connectivity()
    connectivity.add_edge(5, 6)
    placer = StarConnectivityPlacer(connectivity=connectivity)
    with pytest.raises(PlacementError):
        placer(Circuit(5))


def test_incorrect_star_connectivity(star_connectivity):
    connectivity = star_connectivity()
    connectivity.add_edge(3, 4)
    placer = StarConnectivityPlacer(connectivity=connectivity)
    error_msg = "This connectivity graph is not a star graph. There is a node with degree different from 1 and 4."
    with pytest.raises(ConnectivityError, match=error_msg):
        placer(Circuit(5))
