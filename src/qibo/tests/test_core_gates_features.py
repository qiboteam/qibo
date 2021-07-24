"""Test special features of core gates."""
import pytest
import numpy as np
from qibo import K, gates
from qibo.models import Circuit
from qibo.tests.utils import random_state


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
    gate = getattr(gates, gate)(*qubits)
    K.assert_allclose(gate.unitary, target_matrix)


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
    theta = 0.1234
    if gate == "CU1":
        gate = getattr(gates, gate)(0, 1, theta)
    else:
        gate = getattr(gates, gate)(0, theta)
    K.assert_allclose(gate.unitary, target_matrix(theta))
    K.assert_allclose(gate.matrix, target_matrix(theta))


def test_construct_unitary_controlled(backend):
    theta = 0.1234
    rotation = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                         [np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    target_matrix = np.eye(4, dtype=rotation.dtype)
    target_matrix[2:, 2:] = rotation
    gate = gates.RY(0, theta).controlled_by(1)
    K.assert_allclose(gate.unitary, target_matrix)

    gate = gates.RY(0, theta).controlled_by(1, 2)
    with pytest.raises(NotImplementedError):
        unitary = gate.unitary

###############################################################################

########################### Test `Collapse` features ##########################
@pytest.mark.parametrize("nqubits,targets", [(5, [2, 4]), (6, [3, 5])])
def test_measurement_collapse_distributed(backend, accelerators, nqubits, targets):
    initial_state = random_state(nqubits)
    c = Circuit(nqubits, accelerators)
    output = c.add(gates.M(*targets, collapse=True))
    result = c(np.copy(initial_state))
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, output.samples()[0]):
        slicer[t] = int(r)
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    K.assert_allclose(result.state(), target_state)


def test_collapse_after_measurement(backend):
    qubits = [0, 2, 3]
    c = Circuit(5)
    c.add((gates.H(i) for i in range(5)))
    output = c.add(gates.M(*qubits, collapse=True))
    c.add((gates.H(i) for i in range(5)))
    result = c()
    bitstring = output.samples()[0]
    final_state = result.state()

    ct = Circuit(5)
    for i, r in zip(qubits, bitstring):
        if r:
            ct.add(gates.X(i))
    ct.add((gates.H(i) for i in qubits))
    target_state = ct()
    K.assert_allclose(final_state, target_state, atol=1e-15)

###############################################################################

########################## Test gate parameter setter #########################
def test_rx_parameter_setter(backend):
    """Check the parameter setter of RX gate."""
    def exact_state(theta):
        phase = np.exp(1j * theta / 2.0)
        gate = np.array([[phase.real, -1j * phase.imag],
                        [-1j * phase.imag, phase.real]])
        return gate.dot(np.ones(2)) / np.sqrt(2)

    theta = 0.1234
    gate = gates.RX(0, theta=theta)
    initial_state = K.cast(np.ones(2) / np.sqrt(2))
    final_state = gate(initial_state)
    target_state = exact_state(theta)
    K.assert_allclose(final_state, target_state)

    theta = 0.4321
    gate.parameters = theta
    initial_state = K.cast(np.ones(2) / np.sqrt(2))
    final_state = gate(initial_state)
    target_state = exact_state(theta)
    K.assert_allclose(final_state, target_state)

###############################################################################

########################### Test gate decomposition ###########################
@pytest.mark.parametrize(("target", "controls", "free"),
                         [(0, (1,), ()), (2, (0, 1), ()),
                          (3, (0, 1, 4), (2, 5)),
                          (7, (0, 1, 2, 3, 4), (5, 6)),
                          (5, (0, 2, 4, 6, 7), (1, 3)),
                          (8, (0, 2, 4, 6, 9), (3, 5, 7))])
@pytest.mark.parametrize("use_toffolis", [True, False])
def test_x_decomposition_execution(backend, target, controls, free, use_toffolis):
    """Check that applying the decomposition is equivalent to applying the multi-control gate."""
    gate = gates.X(target).controlled_by(*controls)
    nqubits = max((target,) + controls + free) + 1
    initial_state = random_state(nqubits)
    targetc = Circuit(nqubits)
    targetc.add(gate)
    target_state = targetc(np.copy(initial_state))
    c = Circuit(nqubits)
    c.add(gate.decompose(*free, use_toffolis=use_toffolis))
    final_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, target_state, atol=1e-6)

###############################################################################

########################### Test gate decomposition ###########################
def test_one_qubit_gate_multiplication(backend):
    gate1 = gates.X(0)
    gate2 = gates.H(0)
    final_gate = gate1 @ gate2
    assert final_gate.__class__.__name__ == "Unitary"
    target_matrix = (np.array([[0, 1], [1, 0]]) @
                     np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    K.assert_allclose(final_gate.unitary, target_matrix)

    final_gate = gate2 @ gate1
    assert final_gate.__class__.__name__ == "Unitary"
    target_matrix = (np.array([[1, 1], [1, -1]]) / np.sqrt(2) @
                     np.array([[0, 1], [1, 0]]))
    K.assert_allclose(final_gate.unitary, target_matrix)

    gate1 = gates.X(1)
    gate2 = gates.X(1)
    assert (gate1 @ gate2).__class__.__name__ == "I"
    assert (gate2 @ gate1).__class__.__name__ == "I"


def test_two_qubit_gate_multiplication(backend):
    theta, phi = 0.1234, 0.5432
    gate1 = gates.fSim(0, 1, theta=theta, phi=phi)
    gate2 = gates.SWAP(0, 1)
    final_gate = gate1 @ gate2
    target_matrix = (np.array([[1, 0, 0, 0],
                               [0, np.cos(theta), -1j * np.sin(theta), 0],
                               [0, -1j * np.sin(theta), np.cos(theta), 0],
                               [0, 0, 0, np.exp(-1j * phi)]]) @
                     np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                               [0, 1, 0, 0], [0, 0, 0, 1]]))
    K.assert_allclose(final_gate.unitary, target_matrix)
    # Check that error is raised when target qubits do not agree
    with pytest.raises(NotImplementedError):
        final_gate = gate1 @ gates.SWAP(0, 2)


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
    gate = getattr(gates, gate)(*args)
    nqubits = len(gate.qubits)
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))
    initial_state = random_state(nqubits)
    final_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, initial_state)


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
    gate = getattr(gates, gate)(*args).controlled_by(0, 1, 2)
    c = Circuit(4)
    c.add((gate, gate.dagger()))
    initial_state = random_state(4)
    final_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, initial_state)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_dagger(backend, nqubits):
    matrix = np.random.random((2 ** nqubits, 2 ** nqubits))
    gate = gates.Unitary(matrix, *range(nqubits))
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))
    initial_state = random_state(nqubits)
    final_state = c(np.copy(initial_state))
    target_state = np.dot(matrix, initial_state)
    target_state = np.dot(np.conj(matrix).T, target_state)
    K.assert_allclose(final_state, target_state)


def test_controlled_unitary_dagger(backend):
    from scipy.linalg import expm
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.Unitary(matrix, 0).controlled_by(1, 2, 3, 4)
    c = Circuit(5)
    c.add((gate, gate.dagger()))
    initial_state = random_state(5)
    final_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, initial_state)


def test_generalizedfsim_dagger(backend):
    from scipy.linalg import expm
    phi = 0.2
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    c = Circuit(2)
    c.add((gate, gate.dagger()))
    initial_state = random_state(2)
    final_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, initial_state)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_variational_layer_dagger(backend, nqubits):
    theta = 2 * np.pi * np.random.random((2, nqubits))
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    gate = gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ,
                                  theta[0], theta[1])
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))
    initial_state = random_state(nqubits)
    final_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, initial_state)

###############################################################################

##################### Test repeated execution with channels ###################
def test_noise_channel_repeated(backend):
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
    K.assert_allclose(final_state, target_state)


def test_reset_channel_repeated(backend):
    initial_state = random_state(5)
    c = Circuit(5)
    c.add(gates.ResetChannel(2, p0=0.3, p1=0.3, seed=123))
    final_state = c(K.cast(np.copy(initial_state)), nshots=30)

    np.random.seed(123)
    target_state = []
    collapse = gates.M(2, collapse=True)
    collapse.nqubits = 5
    xgate = gates.X(2)
    for _ in range(30):
        state = K.cast(np.copy(initial_state))
        if np.random.random() < 0.3:
            state = K.state_vector_collapse(collapse, state, [0])
        if np.random.random() < 0.3:
            state = K.state_vector_collapse(collapse, state, [0])
            state = xgate(state)
        target_state.append(K.copy(state))
    target_state = K.stack(target_state)
    K.assert_allclose(final_state, target_state)


def test_thermal_relaxation_channel_repeated(backend):
    initial_state = random_state(5)
    c = Circuit(5)
    c.add(gates.ThermalRelaxationChannel(4, t1=1.0, t2=0.6, time=0.8,
                                         excited_population=0.8, seed=123))
    final_state = c(K.cast(np.copy(initial_state)), nshots=30)

    pz, p0, p1 = c.queue[0].calculate_probabilities(1.0, 0.6, 0.8, 0.8)
    np.random.seed(123)
    target_state = []
    collapse = gates.M(4, collapse=True)
    collapse.nqubits = 5
    zgate, xgate = gates.Z(4), gates.X(4)
    for _ in range(30):
        state = K.cast(np.copy(initial_state))
        if np.random.random() < pz:
            state = zgate(state)
        if np.random.random() < p0:
            state = K.state_vector_collapse(collapse, state, [0])
        if np.random.random() < p1:
            state = K.state_vector_collapse(collapse, state, [0])
            state = xgate(state)
        target_state.append(K.copy(state))
    target_state = K.stack(target_state)
    K.assert_allclose(final_state, target_state)

###############################################################################
