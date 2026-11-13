import networkx as nx
import pytest

from qibo import Circuit, gates
from qibo.models.encodings import entangling_layer
from qibo.transpiler._exceptions import ConnectivityError, PlacementError
from qibo.transpiler.asserts import assert_placement, assert_transpiling
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes, restrict_connectivity_qubits
from qibo.transpiler.placer import (
    Dense,
    Random,
    ReverseTraversal,
    StarConnectivityPlacer,
    Subgraph,
    _find_gates_qubits_pairs,
)
from qibo.transpiler.router import Sabre, ShortestPaths
from qibo.transpiler.unroller import NativeGates, Unroller


def star_circuit(names=None):
    if names is None:
        names = [0, 1, 2, 3, 4]
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
        _find_gates_qubits_pairs(circuit)


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
def test_random(backend, names, reps, star_connectivity):
    connectivity = star_connectivity(names)
    placer = Random(connectivity=connectivity, samples=reps)
    circuit = star_circuit(names=names)
    placer(circuit, backend=backend)
    assert_placement(circuit, connectivity)


def test_random_restricted(backend, star_connectivity):
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
    placer(circuit, backend=backend)
    assert_placement(circuit, restricted_connectivity)


@pytest.mark.parametrize("ngates", [None, 5, 13])
@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_reverse_traversal(backend, names, ngates, star_connectivity):
    circuit = star_circuit(names=names)
    connectivity = star_connectivity(names=names)
    routing = ShortestPaths(connectivity=connectivity)
    placer = ReverseTraversal(routing, connectivity, depth=ngates)
    placer(circuit, backend=backend)
    assert_placement(circuit, connectivity)


def test_reverse_traversal_no_gates(backend, star_connectivity):
    connectivity = star_connectivity()
    routing = ShortestPaths(connectivity=connectivity)
    placer = ReverseTraversal(routing, connectivity, depth=10)
    circuit = Circuit(5)
    with pytest.raises(ValueError):
        placer(circuit, backend=backend)


def test_reverse_traversal_restricted(backend, star_connectivity):
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
    placer(circuit, backend=backend)
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


def test_star_connectivity_placer_wrong_node_count(star_connectivity):
    """StarConnectivityPlacer raises when the graph does not have 5 nodes."""
    chip = nx.Graph()
    chip.add_edges_from([(0, 1), (1, 2), (2, 3)])  # 4 nodes
    circ = Circuit(4)
    circ.add(gates.CZ(0, 1))
    placer = StarConnectivityPlacer(chip)
    with pytest.raises(ConnectivityError):
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


@pytest.mark.parametrize("placer_class", [Random, StarConnectivityPlacer, Subgraph])
def test_placer_backend_argument(backend, placer_class, star_connectivity):
    connectivity = star_connectivity()
    circuit = star_circuit()
    placer_class(connectivity=connectivity)(circuit, backend=backend)
    assert_placement(circuit, connectivity)


@pytest.mark.parametrize("placer_class", [StarConnectivityPlacer, Subgraph])
def test_placer_in_passes(backend, placer_class, star_connectivity):
    connectivity = star_connectivity()
    circuit = star_circuit()
    passes = Passes(
        passes=[
            Preprocessing(),
            placer_class(),
            Sabre(seed=0),
            Unroller(native_gates=NativeGates.default(), backend=backend),
        ],
        connectivity=connectivity,
        native_gates=NativeGates.default(),
    )
    transpiled_circuit, final_layout = passes(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


def test_random_zero_samples(star_connectivity):
    connectivity = star_connectivity(["A", "B", "C", "D", "E"])
    circuit = star_circuit(["E", "D", "C", "B", "A"])
    Random(connectivity=connectivity, samples=0)(circuit)
    assert circuit.wire_names == ["A", "B", "C", "D", "E"]
    assert_placement(circuit, connectivity)


def test_gates_qubits_pairs_ignore_barriers():
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.Barrier(0, 1, 2))
    circuit.add(gates.Barrier(3, 4))
    circuit.add(gates.CNOT(1, 2))
    assert _find_gates_qubits_pairs(circuit) == [[0, 1], [1, 2]]


@pytest.mark.parametrize("barrier_qubits", [(0, 1, 2), (0, 1, 2, 3, 4), (0, 4)])
def test_star_connectivity_placer_barrier(barrier_qubits, star_connectivity):
    circ = Circuit(5)
    circ.add(gates.Barrier(*barrier_qubits))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.Barrier(*barrier_qubits))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.CZ(0, 2))
    connectivity = star_connectivity()
    placer = StarConnectivityPlacer(connectivity)
    placer(circ)
    assert_placement(circ, connectivity)
    assert circ.wire_names == [0, 2, 1, 3, 4]


@pytest.mark.parametrize("placer_class", [Random, Subgraph])
def test_placers_with_barrier(placer_class, star_connectivity):
    circ = star_circuit()
    circ.add(gates.Barrier(0, 1, 2, 3))
    connectivity = star_connectivity()
    placer = placer_class(connectivity=connectivity)
    placer(circ)
    assert_placement(circ, connectivity)


@pytest.mark.parametrize(
    "placer_class",
    [
        Random,
        Subgraph,
        StarConnectivityPlacer,
        lambda connectivity=None: ReverseTraversal(Sabre(), connectivity),
    ],
    ids=["Random", "Subgraph", "StarConnectivityPlacer", "ReverseTraversal"],
)
def test_placers_store_undirected_connectivity(placer_class, star_connectivity):
    assert placer_class().connectivity is None
    undirected = star_connectivity()
    assert placer_class(undirected).connectivity is undirected
    directed = nx.DiGraph(undirected.edges)
    placer = placer_class(directed)
    assert not placer.connectivity.is_directed()
    assert set(map(frozenset, placer.connectivity.edges)) == set(
        map(frozenset, undirected.edges)
    )
    assert directed.is_directed()


def count_swaps(circuit, connectivity):
    """Number of SWAP gates that Sabre adds to route the circuit."""
    routed_circuit, _ = Sabre(connectivity, seed=1)(circuit.copy())
    return sum(isinstance(gate, gates.SWAP) for gate in routed_circuit.queue)


def test_dense_no_two_qubit_gates():
    circuit = Circuit(4, wire_names=[3, 2, 1, 0])
    circuit.add(gates.H(0))
    circuit.add(gates.M(1))
    Dense(nx.path_graph(4))(circuit)
    assert circuit.wire_names == [3, 2, 1, 0]


def test_dense_error():
    connectivity = nx.Graph([(0, 1), (2, 3)])
    circuit = Circuit(4)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    with pytest.raises(PlacementError):
        Dense(connectivity)(circuit)


def test_dense_multi_qubit_gate_error():
    circuit = Circuit(4)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ValueError):
        Dense(nx.path_graph(4))(circuit)


def test_reverse_traversal_trials(backend):
    connectivity = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 4))
    circuit = entangling_layer(8, "pyramid")
    swaps = {}
    for trials in (1, 10):
        copy = circuit.copy()
        placer = ReverseTraversal(Sabre(seed=1), connectivity, trials=trials, seed=1)
        placer(copy, backend=backend)
        assert_placement(copy, connectivity)
        swaps[trials] = count_swaps(copy, connectivity)
    assert swaps[10] < swaps[1]


def test_reverse_traversal_trials_error():
    with pytest.raises(ValueError):
        ReverseTraversal(Sabre(), trials=0)
