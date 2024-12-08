import networkx as nx
import numpy as np
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler._exceptions import ConnectivityError, TranspilerPipelineError
from qibo.transpiler.asserts import assert_circuit_equivalence, assert_transpiling
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes, restrict_connectivity_qubits
from qibo.transpiler.placer import Random, ReverseTraversal
from qibo.transpiler.router import Sabre, ShortestPaths
from qibo.transpiler.unroller import NativeGates, Unroller


def generate_random_circuit(nqubits, ngates, names=None, seed=42):
    """Generate a random circuit with RX and CZ gates."""
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
    circuit = Circuit(nqubits, wire_names=names)
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


def test_restrict_qubits_error_no_subset(star_connectivity):
    with pytest.raises(ConnectivityError) as excinfo:
        restrict_connectivity_qubits(star_connectivity(), [0, 1, 5])
    assert "Some qubits are not in the original connectivity." in str(excinfo.value)


def test_restrict_qubits_error_not_connected(star_connectivity):
    with pytest.raises(ConnectivityError) as excinfo:
        restrict_connectivity_qubits(star_connectivity(), [0, 1])
    assert "New connectivity graph is not connected." in str(excinfo.value)


def test_restrict_qubits(star_connectivity):
    new_connectivity = restrict_connectivity_qubits(
        star_connectivity(["A", "B", "C", "D", "E"]), ["A", "B", "C"]
    )
    assert list(new_connectivity.nodes) == ["A", "B", "C"]
    assert list(new_connectivity.edges) == [("A", "C"), ("B", "C")]


def test_assert_circuit_equivalence_wrong_nqubits():
    circ1 = Circuit(1)
    circ2 = Circuit(2)
    final_layout = {0: 0, 1: 1}
    with pytest.raises(ValueError):
        assert_circuit_equivalence(circ1, circ2, final_layout=final_layout)


@pytest.mark.parametrize("qubits", [3, 5])
def test_is_satisfied(qubits, star_connectivity):
    default_transpiler = Passes(
        passes=None, connectivity=star_connectivity(), on_qubits=list(range(qubits))
    )
    circuit = Circuit(qubits, wire_names=list(range(qubits)))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.Z(0))
    assert default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_placement(star_connectivity):
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5, wire_names=["A", "B", "C", "D", "E"])
    assert not default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_decomposition(star_connectivity):
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.X(0))
    assert not default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_connectivity(star_connectivity):
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.Z(0))
    assert not default_transpiler.is_satisfied(circuit)


@pytest.mark.parametrize("nqubits", [2, 3, 5])
@pytest.mark.parametrize("ngates", [5, 20])
@pytest.mark.parametrize("placer", [Random, ReverseTraversal])
@pytest.mark.parametrize("router", [ShortestPaths, Sabre])
def test_custom_passes(placer, router, ngates, nqubits, star_connectivity):
    connectivity = star_connectivity()
    circ = generate_random_circuit(nqubits=nqubits, ngates=ngates)
    custom_passes = []
    custom_passes.append(Preprocessing())
    if placer == ReverseTraversal:
        custom_passes.append(
            placer(
                routing_algorithm=router(),
            )
        )
    else:
        custom_passes.append(placer())
    custom_passes.append(router())
    custom_passes.append(Unroller(native_gates=NativeGates.default()))
    custom_pipeline = Passes(
        passes=custom_passes,
        connectivity=connectivity,
        native_gates=NativeGates.default(),
    )
    transpiled_circ, final_layout = custom_pipeline(circ)
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )


@pytest.mark.parametrize("ngates", [5, 20])
@pytest.mark.parametrize("placer", [Random, ReverseTraversal])
@pytest.mark.parametrize("routing", [ShortestPaths, Sabre])
@pytest.mark.parametrize("restrict_names", [[1, 2, 3], [0, 2, 4], [4, 2, 3]])
def test_custom_passes_restrict(
    ngates, placer, routing, restrict_names, star_connectivity
):
    connectivity = star_connectivity()
    circ = generate_random_circuit(nqubits=3, ngates=ngates, names=restrict_names)
    custom_passes = []
    custom_passes.append(Preprocessing())
    if placer == ReverseTraversal:
        custom_passes.append(
            placer(
                routing_algorithm=routing(),
            )
        )
    else:
        custom_passes.append(placer())
    custom_passes.append(routing())
    custom_passes.append(Unroller(native_gates=NativeGates.default()))
    custom_pipeline = Passes(
        passes=custom_passes,
        connectivity=connectivity,
        native_gates=NativeGates.default(),
        on_qubits=restrict_names,
    )
    transpiled_circ, final_layout = custom_pipeline(circ)
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=restrict_connectivity_qubits(star_connectivity(), restrict_names),
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )
    assert set(transpiled_circ.wire_names) == set(restrict_names)


def test_custom_passes_wrong_pass():
    custom_passes = [0]
    custom_pipeline = Passes(passes=custom_passes, connectivity=None)
    circ = generate_random_circuit(nqubits=5, ngates=5)
    with pytest.raises(TranspilerPipelineError):
        custom_pipeline(circ)


def test_int_qubit_names(star_connectivity):
    names = [980, 123, 45, 9, 210464]
    connectivity = star_connectivity(names)
    transpiler = Passes(
        connectivity=connectivity,
        passes=[
            Preprocessing(),
            Random(seed=0),
            Sabre(),
            Unroller(NativeGates.default()),
        ],
    )
    circuit = Circuit(1, wire_names=[123])
    circuit.add(gates.I(0))
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    transpiled_circuit, final_layout = transpiler(circuit)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )
