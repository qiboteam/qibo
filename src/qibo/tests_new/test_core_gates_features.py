"""Test special features of core gates."""
import pytest
import numpy as np
import qibo
from qibo import K, gates
from qibo.models import Circuit
from qibo.tests_new.test_core_gates import random_state


####################### Test `construct_unitary` feature #######################
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


def test_construct_unitary_controlled(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    rotation = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                         [np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    target_matrix = np.eye(4, dtype=rotation.dtype)
    target_matrix[2:, 2:] = rotation
    gate = gates.RY(0, theta).controlled_by(1)
    np.testing.assert_allclose(gate.unitary, target_matrix)

    gate = gates.RY(0, theta).controlled_by(1, 2)
    with pytest.raises(NotImplementedError):
        unitary = gate.unitary
    qibo.set_backend(original_backend)

###############################################################################

########################### Test `Collapse` features ##########################
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


def test_collapse_after_measurement(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    qubits = [0, 2, 3]

    c1 = Circuit(5)
    c1.add((gates.H(i) for i in range(5)))
    c1.add(gates.M(*qubits))
    result = c1(nshots=1)
    c2 = Circuit(5)
    bitstring = result.samples(binary=True)[0]
    c2.add(gates.Collapse(*qubits, result=bitstring))
    c2.add((gates.H(i) for i in range(5)))
    final_state = c2(initial_state=c1.final_state)

    ct = Circuit(5)
    for i, r in zip(qubits, bitstring):
        if r:
            ct.add(gates.X(i))
    ct.add((gates.H(i) for i in qubits))
    target_state = ct()
    np.testing.assert_allclose(final_state, target_state, atol=1e-15)
    qibo.set_backend(original_backend)

###############################################################################

########################## Test gate parameter setter #########################
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

###############################################################################

################################# Test dagger #################################
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

###############################################################################

##################### Test repeated execution with channels ###################
def test_noise_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    thetas = np.random.random(4)
    probs = 0.1 * np.random.random([4, 3]) + 0.2
    gatelist = [gates.X, gates.Y, gates.Z]

    c = Circuit(4)
    c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
    c.add((gates.PauliNoiseChannel(i, px, py, pz, seed=123)
           for i, (px, py, pz) in enumerate(probs)))
    final_state = c(nshots=40)

    np.random.seed(123)
    target_state = []
    for _ in range(40):
        noiseless_c = Circuit(4)
        noiseless_c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
        for i, ps in enumerate(probs):
            for p, gate in zip(ps, gatelist):
                if np.random.random() < p:
                    noiseless_c.add(gate(i))
        target_state.append(noiseless_c())
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_reset_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    initial_state = random_state(5)
    c = Circuit(5)
    c.add(gates.ResetChannel(2, p0=0.3, p1=0.3, seed=123))
    final_state = c(np.copy(initial_state), nshots=30)

    np.random.seed(123)
    target_state = []
    for _ in range(30):
        noiseless_c = Circuit(5)
        if np.random.random() < 0.3:
            noiseless_c.add(gates.Collapse(2))
        if np.random.random() < 0.3:
            noiseless_c.add(gates.Collapse(2))
            noiseless_c.add(gates.X(2))
        target_state.append(noiseless_c(np.copy(initial_state)))
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_thermal_relaxation_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    initial_state = random_state(5)
    c = Circuit(5)
    c.add(gates.ThermalRelaxationChannel(4, t1=1.0, t2=0.6, time=0.8,
                                         excited_population=0.8, seed=123))
    final_state = c(np.copy(initial_state), nshots=30)

    pz, p0, p1 = c.queue[0].calculate_probabilities(1.0, 0.6, 0.8, 0.8)
    np.random.seed(123)
    target_state = []
    for _ in range(30):
        noiseless_c = Circuit(5)
        if np.random.random() < pz:
            noiseless_c.add(gates.Z(4))
        if np.random.random() < p0:
            noiseless_c.add(gates.Collapse(4))
        if np.random.random() < p1:
            noiseless_c.add(gates.Collapse(4))
            noiseless_c.add(gates.X(4))
        target_state.append(noiseless_c(np.copy(initial_state)))
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)

###############################################################################
