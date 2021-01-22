"""Test special features of core gates."""
import pytest
import numpy as np
import qibo
from qibo import K, gates
from qibo.models import Circuit
from qibo.tests_new.test_core_gates import random_state


GATES = [
    ("H", (0,), np.array([[1, 1], [1, -1]]) / np.sqrt(2)),
    ("X", (0,), np.array([[0, 1], [1, 0]])),
    ("Y", (0,), np.array([[0, -1j], [1j, 0]])),
    ("Z", (1,), np.array([[1, 0], [0, -1]])),
    ("CNOT", (0, 1), np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 0, 1], [0, 0, 1, 0]])),
    ("CZ", (1, 3), np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, -1]])),
    ("SWAP", (2, 4), np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                               [0, 1, 0, 0], [0, 0, 0, 1]])),
    ("TOFFOLI", (1, 2, 3), np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 1, 0]]))
]
@pytest.mark.parametrize("gate,qubits,target_matrix", GATES)
def test_construct_unitary(backend, gate, qubits, target_matrix):
    """Check that `construct_unitary` method constructs the proper matrix."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    gate = getattr(gates, gate)(*qubits)
    np.testing.assert_allclose(gate.unitary, target_matrix)
    qibo.set_backend(original_backend)


GATES = [
    ("RX", lambda x: np.array([[np.cos(x / 2.0), -1j * np.sin(x / 2.0)],
                               [-1j * np.sin(x / 2.0), np.cos(x / 2.0)]])),
    ("RY", lambda x: np.array([[np.cos(x / 2.0), -np.sin(x / 2.0)],
                               [np.sin(x / 2.0), np.cos(x / 2.0)]])),
    ("RZ", lambda x: np.diag([np.exp(-1j * x / 2.0), np.exp(1j * x / 2.0)])),
    ("U1", lambda x: np.diag([1, np.exp(1j * x)])),
    ("CU1", lambda x: np.diag([1, 1, 1, np.exp(1j * x)]))
]
@pytest.mark.parametrize("gate,target_matrix", GATES)
def test_construct_unitary_rotations(backend, gate, target_matrix):
    """Check that `construct_unitary` method constructs the proper matrix."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    if gate == "CU1":
        gate = getattr(gates, gate)(0, 1, theta)
    else:
        gate = getattr(gates, gate)(0, theta)
    np.testing.assert_allclose(gate.unitary, target_matrix(theta))
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits,targets", [(5, [2, 4]), (6, [3, 5])])
def test_collapse_gate_distributed(backend, accelerators, nqubits, targets):
    """Check :class:`qibo.core.cgates.Collapse` as part of distributed circuits."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_state = random_state(nqubits)
    c = Circuit(nqubits, accelerators)
    c.add(gates.Collapse(*targets))
    final_state = c(np.copy(initial_state))
    slicer = nqubits * [slice(None)]
    for t in targets:
        slicer[t] = 0
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_rx_parameter_setter(backend):
    """Check the parameter setter of RX gate."""
    def exact_state(theta):
        phase = np.exp(1j * theta / 2.0)
        gate = np.array([[phase.real, -1j * phase.imag],
                        [-1j * phase.imag, phase.real]])
        return gate.dot(np.ones(2)) / np.sqrt(2)

    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    gate = gates.RX(0, theta=theta)
    initial_state = K.cast(np.ones(2) / np.sqrt(2))
    final_state = gate(initial_state)
    target_state = exact_state(theta)
    np.testing.assert_allclose(final_state, target_state)

    theta = 0.4321
    gate.parameters = theta
    initial_state = K.cast(np.ones(2) / np.sqrt(2))
    final_state = gate(initial_state)
    target_state = exact_state(theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


GATES = [
    ("H", (0,)),
    ("X", (0,)),
    ("Y", (0,)),
    ("Z", (0,)),
    ("RX", (0, 0.1)),
    ("RY", (0, 0.2)),
    ("RZ", (0, 0.3)),
    ("U1", (0, 0.1)),
    ("U2", (0, 0.2, 0.3)),
    ("U3", (0, 0.1, 0.2, 0.3)),
    ("CNOT", (0, 1)),
    ("CRX", (0, 1, 0.1)),
    ("CRZ", (0, 1, 0.3)),
    ("CU1", (0, 1, 0.1)),
    ("CU2", (0, 1, 0.2, 0.3)),
    ("CU3", (0, 1, 0.1, 0.2, 0.3)),
    ("fSim", (0, 1, 0.1, 0.2))
]
@pytest.mark.parametrize("gate,args", GATES)
def test_dagger(backend, gate, args):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    gate = getattr(gates, gate)(*args)
    nqubits = len(gate.qubits)
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))
    initial_state = random_state(nqubits)
    final_state = c(np.copy(initial_state))
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


GATES = [
    ("H", (3,)),
    ("X", (3,)),
    ("Y", (3,)),
    ("RX", (3, 0.1)),
    ("U1", (3, 0.1)),
    ("U3", (3, 0.1, 0.2, 0.3))
]
@pytest.mark.parametrize("gate,args", GATES)
def test_controlled_dagger(backend, gate, args):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    gate = getattr(gates, gate)(*args).controlled_by(0, 1, 2)
    c = Circuit(4)
    c.add((gate, gate.dagger()))
    initial_state = random_state(4)
    final_state = c(np.copy(initial_state))
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_dagger(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = np.random.random((2 ** nqubits, 2 ** nqubits))
    gate = gates.Unitary(matrix, *range(nqubits))
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))
    initial_state = random_state(nqubits)
    final_state = c(np.copy(initial_state))
    target_state = np.dot(matrix, initial_state)
    target_state = np.dot(np.conj(matrix).T, target_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_controlled_unitary_dagger(backend):
    from scipy.linalg import expm
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.Unitary(matrix, 0).controlled_by(1, 2, 3, 4)
    c = Circuit(5)
    c.add((gate, gate.dagger()))
    initial_state = random_state(5)
    final_state = c(np.copy(initial_state))
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


def test_generalizedfsim_dagger(backend):
    from scipy.linalg import expm
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = 0.2
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    c = Circuit(2)
    c.add((gate, gate.dagger()))
    initial_state = random_state(2)
    final_state = c(np.copy(initial_state))
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_variational_layer_dagger(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 2 * np.pi * np.random.random((2, nqubits))
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    gate = gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ,
                                  theta[0], theta[1])
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))
    initial_state = random_state(nqubits)
    final_state = c(np.copy(initial_state))
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)
