import networkx as nx
import numpy as np
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler._exceptions import ConnectivityError, TranspilerPipelineError
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import (
    Passes,
    assert_circuit_equivalence,
    assert_transpiling,
    restrict_connectivity_qubits,
)
from qibo.transpiler.placer import Random, ReverseTraversal, Trivial
from qibo.transpiler.router import Sabre, ShortestPaths
from qibo.transpiler.unroller import NativeGates, Unroller


def generate_random_circuit(nqubits, ngates, seed=None):
    """Generate random circuits one-qubit rotations and CZ gates."""
    if seed is not None:  # pragma: no cover
        np.random.seed(seed)

    one_qubit_gates = [gates.RX, gates.RY, gates.RZ, gates.X, gates.Y, gates.Z, gates.H]
    two_qubit_gates = [
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
        gates.iSWAP,
        gates.CRX,
        gates.CRY,
        gates.CRZ,
    ]
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
            circuit.add(gate(*q, theta=theta))
        else:
            circuit.add(gate(*q))
    return circuit


def star_connectivity():
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


def test_restrict_qubits_error_no_subset():
    with pytest.raises(ConnectivityError) as excinfo:
        restrict_connectivity_qubits(star_connectivity(), [1, 2, 6])
    assert "Some qubits are not in the original connectivity." in str(excinfo.value)


def test_restrict_qubits_error_not_connected():
    with pytest.raises(ConnectivityError) as excinfo:
        restrict_connectivity_qubits(star_connectivity(), [1, 3])
    assert "New connectivity graph is not connected." in str(excinfo.value)


def test_restrict_qubits():
    new_connectivity = restrict_connectivity_qubits(star_connectivity(), [1, 2, 3])
    assert list(new_connectivity.nodes) == [1, 2, 3]
    assert list(new_connectivity.edges) == [(1, 2), (2, 3)]


@pytest.mark.parametrize("ngates", [5, 10, 50])
def test_pipeline_default(ngates):
    circ = generate_random_circuit(nqubits=5, ngates=ngates)
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    transpiled_circ, final_layout = default_transpiler(circ)
    initial_layout = default_transpiler.get_initial_layout()
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=star_connectivity(),
        initial_layout=initial_layout,
        final_layout=final_layout,
        native_gates=NativeGates.default(),
        check_circuit_equivalence=False,
    )


def test_assert_circuit_equivalence_equal():
    circ1 = Circuit(2)
    circ2 = Circuit(2)
    circ1.add(gates.X(0))
    circ1.add(gates.CZ(0, 1))
    circ2.add(gates.X(0))
    circ2.add(gates.CZ(0, 1))
    final_map = {"q0": 0, "q1": 1}
    assert_circuit_equivalence(circ1, circ2, final_map=final_map)


def test_assert_circuit_equivalence_swap():
    circ1 = Circuit(2)
    circ2 = Circuit(2)
    circ1.add(gates.X(0))
    circ2.add(gates.SWAP(0, 1))
    circ2.add(gates.X(1))
    final_map = {"q0": 1, "q1": 0}
    assert_circuit_equivalence(circ1, circ2, final_map=final_map)


def test_assert_circuit_equivalence_false():
    circ1 = Circuit(2)
    circ2 = Circuit(2)
    circ1.add(gates.X(0))
    circ2.add(gates.SWAP(0, 1))
    circ2.add(gates.X(1))
    final_map = {"q0": 0, "q1": 1}
    with pytest.raises(TranspilerPipelineError):
        assert_circuit_equivalence(circ1, circ2, final_map=final_map)


def test_int_qubit_names():
    circ = Circuit(2)
    final_map = {i: i for i in range(5)}
    default_transpiler = Passes(
        passes=None, connectivity=star_connectivity(), int_qubit_names=True
    )
    _, final_layout = default_transpiler(circ)
    assert final_map == final_layout


def test_assert_circuit_equivalence_wrong_nqubits():
    circ1 = Circuit(1)
    circ2 = Circuit(2)
    final_map = {"q0": 0, "q1": 1}
    with pytest.raises(ValueError):
        assert_circuit_equivalence(circ1, circ2, final_map=final_map)


def test_error_connectivity():
    with pytest.raises(TranspilerPipelineError):
        default_transpiler = Passes(passes=None, connectivity=None)


@pytest.mark.parametrize("qubits", [3, 5])
def test_is_satisfied(qubits):
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(qubits)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.Z(0))
    assert default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_decomposition():
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.X(0))
    assert not default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_connectivity():
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.Z(0))
    assert not default_transpiler.is_satisfied(circuit)


@pytest.mark.parametrize("qubits", [2, 5])
@pytest.mark.parametrize("gates", [5, 20])
@pytest.mark.parametrize("placer", [Random, Trivial, ReverseTraversal])
@pytest.mark.parametrize("routing", [ShortestPaths, Sabre])
def test_custom_passes(placer, routing, gates, qubits):
    circ = generate_random_circuit(nqubits=qubits, ngates=gates)
    custom_passes = []
    custom_passes.append(Preprocessing(connectivity=star_connectivity()))
    if placer == ReverseTraversal:
        custom_passes.append(
            placer(
                connectivity=star_connectivity(),
                routing_algorithm=routing(connectivity=star_connectivity()),
            )
        )
    else:
        custom_passes.append(placer(connectivity=star_connectivity()))
    custom_passes.append(routing(connectivity=star_connectivity()))
    custom_passes.append(Unroller(native_gates=NativeGates.default()))
    custom_pipeline = Passes(
        custom_passes,
        connectivity=star_connectivity(),
        native_gates=NativeGates.default(),
    )
    transpiled_circ, final_layout = custom_pipeline(circ)
    initial_layout = custom_pipeline.get_initial_layout()
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=star_connectivity(),
        initial_layout=initial_layout,
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )


@pytest.mark.parametrize("gates", [5, 20])
@pytest.mark.parametrize("placer", [Random, Trivial, ReverseTraversal])
@pytest.mark.parametrize("routing", [ShortestPaths, Sabre])
def test_custom_passes_restict(gates, placer, routing):
    circ = generate_random_circuit(nqubits=3, ngates=gates)
    custom_passes = []
    custom_passes.append(Preprocessing(connectivity=star_connectivity()))
    if placer == ReverseTraversal:
        custom_passes.append(
            placer(
                connectivity=star_connectivity(),
                routing_algorithm=routing(connectivity=star_connectivity()),
            )
        )
    else:
        custom_passes.append(placer(connectivity=star_connectivity()))
    custom_passes.append(routing(connectivity=star_connectivity()))
    custom_passes.append(Unroller(native_gates=NativeGates.default()))
    custom_pipeline = Passes(
        custom_passes,
        connectivity=star_connectivity(),
        native_gates=NativeGates.default(),
        on_qubits=[1, 2, 3],
    )
    transpiled_circ, final_layout = custom_pipeline(circ)
    initial_layout = custom_pipeline.get_initial_layout()
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=restrict_connectivity_qubits(star_connectivity(), [1, 2, 3]),
        initial_layout=initial_layout,
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )
    assert transpiled_circ.wire_names == ["q1", "q2", "q3"]


def test_custom_passes_multiple_placer():
    custom_passes = []
    custom_passes.append(Random(connectivity=star_connectivity()))
    custom_passes.append(Trivial(connectivity=star_connectivity()))
    custom_pipeline = Passes(
        custom_passes,
        connectivity=star_connectivity(),
        native_gates=NativeGates.default(),
    )
    circ = generate_random_circuit(nqubits=5, ngates=20)
    with pytest.raises(TranspilerPipelineError):
        transpiled_circ, final_layout = custom_pipeline(circ)


def test_custom_passes_no_placer():
    custom_passes = []
    custom_passes.append(ShortestPaths(connectivity=star_connectivity()))
    custom_pipeline = Passes(
        custom_passes,
        connectivity=star_connectivity(),
        native_gates=NativeGates.default(),
    )
    circ = generate_random_circuit(nqubits=5, ngates=20)
    with pytest.raises(TranspilerPipelineError):
        transpiled_circ, final_layout = custom_pipeline(circ)


def test_custom_passes_wrong_pass():
    custom_passes = [0]
    custom_pipeline = Passes(passes=custom_passes, connectivity=None)
    circ = generate_random_circuit(nqubits=5, ngates=5)
    with pytest.raises(TranspilerPipelineError):
        transpiled_circ, final_layout = custom_pipeline(circ)


def test_int_qubit_names():
    connectivity = star_connectivity()
    transpiler = Passes(
        connectivity=connectivity,
        passes=[
            Preprocessing(connectivity),
            Random(connectivity, seed=0),
            Sabre(connectivity),
            Unroller(NativeGates.default()),
        ],
        int_qubit_names=True,
    )
    circuit = Circuit(1)
    circuit.add(gates.I(0))
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    transpiled_circuit, final_map = transpiler(circuit)
    initial_layout = transpiler.get_initial_layout()
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        initial_layout=initial_layout,
        final_layout=final_map,
        native_gates=NativeGates.default(),
    )
