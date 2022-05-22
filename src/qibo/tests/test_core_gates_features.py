"""Test special features of core gates."""
import pytest
import numpy as np
from qibo import K, gates
from qibo.models import Circuit
from qibo.tests.utils import random_state


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
    K.assert_allclose(final_gate.matrix, target_matrix)

    final_gate = gate2 @ gate1
    assert final_gate.__class__.__name__ == "Unitary"
    target_matrix = (np.array([[1, 1], [1, -1]]) / np.sqrt(2) @
                     np.array([[0, 1], [1, 0]]))
    K.assert_allclose(final_gate.matrix, target_matrix)

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
    K.assert_allclose(final_gate.matrix, target_matrix)
    # Check that error is raised when target qubits do not agree
    with pytest.raises(NotImplementedError):
        final_gate = gate1 @ gates.SWAP(0, 2)


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
