"""Test gates defined in `qibo/core/cgates.py` and `qibo/core/gates.py`."""
import pytest
import numpy as np
import qibo
from qibo import gates
from qibo.config import raise_error


def random_state(nqubits):
    """Generates a random normalized state vector as numpy array."""
    nstates = 2 ** nqubits
    initial_state = np.random.random(nstates) + 1j * np.random.random(nstates)
    return initial_state / np.sqrt((np.abs(initial_state) ** 2).sum())


def random_density_matrix(nqubits):
    """Generates a random normalized density matrix."""
    shape = 2 * (2 ** nqubits,)
    m = np.random.random(shape) + 1j * np.random.random(shape)
    rho = (m + m.T.conj()) / 2.0
    # Normalize
    ids = np.arange(2 ** nqubits)
    rho[ids, ids] = rho[ids, ids] / np.trace(rho)
    return rho


def apply_gates(gatelist, nqubits=None, initial_state=None):
    from qibo import K
    if initial_state is None:
        state = K.qnp.zeros(2 ** nqubits)
        state[0] = 1
    elif isinstance(initial_state, np.ndarray):
        state = np.copy(initial_state)
        if nqubits is None:
            nqubits = int(np.log2(len(state)))
        else: # pragma: no cover
            assert nqubits == int(np.log2(len(state)))
    else: # pragma: no cover
        raise_error(TypeError, "Invalid initial state type {}."
                               "".format(type(initial_state)))

    state = K.cast(state)
    for gate in gatelist:
        state = gate(state)
    return state


def test_control_unitary(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = np.random.random((2, 2))
    gate = gates.Unitary(matrix, 0)
    unitary = np.array(gate.control_unitary(matrix))
    target_unitary = np.eye(4, dtype=unitary.dtype)
    target_unitary[2:, 2:] = matrix
    np.testing.assert_allclose(unitary, target_unitary)
    with pytest.raises(ValueError):
        unitary = gate.control_unitary(np.random.random((16, 16)))
    qibo.set_backend(original_backend)


def test_h(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.H(0), gates.H(1)], nqubits=2)
    target_state = np.ones_like(final_state) / 2
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_x(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.X(0)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_y(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.Y(1)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[1] = 1j
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_z(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.H(0), gates.H(1), gates.Z(0)], nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    target_state[2] *= -1.0
    target_state[3] *= -1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_identity(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    gatelist = [gates.H(0), gates.H(1), gates.I(0), gates.I(1)]
    final_state = apply_gates(gatelist, nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    np.testing.assert_allclose(final_state, target_state)
    gatelist = [gates.H(0), gates.H(1), gates.I(0, 1)]
    final_state = apply_gates(gatelist, nqubits=2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)

# :class:`qibo.core.cgates.M` is tested seperately in `test_measurement_gate.py`

def test_rx(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    final_state = apply_gates([gates.H(0), gates.RX(0, theta=theta)], nqubits=1)
    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -1j * phase.imag],
                    [-1j * phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_ry(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    final_state = apply_gates([gates.H(0), gates.RY(0, theta=theta)], nqubits=1)
    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -phase.imag],
                     [phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("applyx", [True, False])
def test_rz(backend, applyx):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    if applyx:
        gatelist = [gates.X(0)]
    else:
        gatelist = []
    gatelist.append(gates.RZ(0, theta))
    final_state = apply_gates(gatelist, nqubits=1)
    target_state = np.zeros_like(final_state)
    p = int(applyx)
    target_state[p] = np.exp((2 * p - 1) * 1j * theta / 2.0)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_u1(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    final_state = apply_gates([gates.X(0), gates.U1(0, theta)], nqubits=1)
    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_u2(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = 0.1234
    lam = 0.4321
    initial_state = random_state(1)
    final_state = apply_gates([gates.U2(0, phi, lam)], initial_state=initial_state)
    matrix = np.array([[np.exp(-1j * (phi + lam) / 2), -np.exp(-1j * (phi - lam) / 2)],
                       [np.exp(1j * (phi - lam) / 2), np.exp(1j * (phi + lam) / 2)]])
    target_state = matrix.dot(initial_state) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_u3(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1111
    phi = 0.1234
    lam = 0.4321
    initial_state = random_state(1)
    final_state = apply_gates([gates.U3(0, theta, phi, lam)],
                              initial_state=initial_state)
    cost, sint = np.cos(theta / 2), np.sin(theta / 2)
    ep = np.exp(1j * (phi + lam) / 2)
    em = np.exp(1j * (phi - lam) / 2)
    matrix = np.array([[ep.conj() * cost, - em.conj() * sint],
                       [em * sint, ep * cost]])
    target_state = matrix.dot(initial_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_zpow(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    final_state = apply_gates([gates.X(0), gates.ZPow(0, theta)], nqubits=1)
    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("applyx", [False, True])
def test_cnot(backend, applyx):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    if applyx:
        gatelist = [gates.X(0)]
    else:
        gatelist = []
    gatelist.append(gates.CNOT(0, 1))
    final_state = apply_gates(gatelist, nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[3 * int(applyx)] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("controlled_by", [False, True])
def test_cz(backend, controlled_by):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_state = random_state(2)
    matrix = np.eye(4)
    matrix[3, 3] = -1
    target_state = matrix.dot(initial_state)
    if controlled_by:
        gate = gates.Z(1).controlled_by(0)
    else:
        gate = gates.CZ(0, 1)
    final_state = apply_gates([gate], initial_state=initial_state)
    assert gate.name == "cz"
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("name,params",
                         [("CRX", {"theta": 0.1}),
                          ("CRY", {"theta": 0.2}),
                          ("CRZ", {"theta": 0.3}),
                          ("CU1", {"theta": 0.1}),
                          ("CU2", {"phi": 0.1, "lam": 0.2}),
                          ("CU3", {"theta": 0.1, "phi": 0.2, "lam": 0.3})])
def test_cun(backend, name, params):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_state = random_state(2)
    gate = getattr(gates, name)(0, 1, **params)
    final_state = apply_gates([gate], initial_state=initial_state)
    target_state = np.dot(gate.unitary, initial_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_czpow(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    gatelist = [gates.X(0), gates.X(1), gates.CZPow(0, 1, theta)]
    final_state = apply_gates(gatelist, nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[-1] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_swap(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.X(1), gates.SWAP(0, 1)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_multiple_swap(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    gatelist = [gates.X(0), gates.X(2), gates.SWAP(0, 1), gates.SWAP(2, 3)]
    final_state = apply_gates(gatelist, nqubits=4)
    gatelist = [gates.X(1), gates.X(3)]
    target_state = apply_gates(gatelist, nqubits=4)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_fsim(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    phi = 0.4321
    gatelist = [gates.H(0), gates.H(1), gates.fSim(0, 1, theta, phi)]
    final_state = apply_gates(gatelist, nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    rotation = np.array([[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]])
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state = matrix.dot(target_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_generalized_fsim(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = np.random.random()
    rotation = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    gatelist = [gates.H(0), gates.H(1), gates.H(2)]
    gatelist.append(gates.GeneralizedfSim(1, 2, rotation, phi))
    final_state = apply_gates(gatelist, nqubits=3)
    target_state = np.ones_like(final_state) / np.sqrt(8)
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state[:4] = matrix.dot(target_state[:4])
    target_state[4:] = matrix.dot(target_state[4:])
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_generalized_fsim_parameter_setter(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = np.random.random()
    matrix = np.random.random((2, 2))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    np.testing.assert_allclose(gate.parameters[0], matrix)
    assert gate.parameters[1] == phi
    matrix = np.random.random((4, 4))
    with pytest.raises(ValueError):
        gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("applyx", [False, True])
def test_toffoli(backend, applyx):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    if applyx:
        gatelist = [gates.X(0), gates.X(1), gates.TOFFOLI(0, 1, 2)]
    else:
        gatelist = [gates.X(1), gates.TOFFOLI(0, 1, 2)]
    final_state = apply_gates(gatelist, nqubits=3)
    target_state = np.zeros_like(final_state)
    if applyx:
        target_state[-1] = 1
    else:
        target_state[2] = 1
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [2, 3])
def test_unitary(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    matrix = np.random.random(2 * (2 ** (nqubits - 1),))
    target_state = np.kron(np.eye(2), matrix).dot(initial_state)
    gatelist = [gates.H(i) for i in range(nqubits)]
    gatelist.append(gates.Unitary(matrix, *range(1, nqubits), name="random"))
    final_state = apply_gates(gatelist, nqubits=nqubits)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_unitary_initialization(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = np.random.random((4, 4))
    gate = gates.Unitary(matrix, 0, 1)
    np.testing.assert_allclose(gate.parameters, matrix)
    matrix = np.random.random((8, 8))
    with pytest.raises(ValueError):
        gate = gates.Unitary(matrix, 0, 1)
    with pytest.raises(TypeError):
        gate = gates.Unitary("abc", 0, 1)
    if backend == "custom":
        with pytest.raises(NotImplementedError):
            gate = gates.Unitary(matrix, 0, 1, 2)
    qibo.set_backend(original_backend)


def test_unitary_common_gates(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    target_state = apply_gates([gates.X(0), gates.H(1)], nqubits=2)
    gatelist = [gates.Unitary(np.array([[0, 1], [1, 0]]), 0),
                gates.Unitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), 1)]
    final_state = apply_gates(gatelist, nqubits=2)
    np.testing.assert_allclose(final_state, target_state)

    thetax = 0.1234
    thetay = 0.4321
    gatelist = [gates.RX(0, theta=thetax), gates.RY(1, theta=thetay),
                gates.CNOT(0, 1)]
    target_state = apply_gates(gatelist, nqubits=2)

    rx = np.array([[np.cos(thetax / 2), -1j * np.sin(thetax / 2)],
                   [-1j * np.sin(thetax / 2), np.cos(thetax / 2)]])
    ry = np.array([[np.cos(thetay / 2), -np.sin(thetay / 2)],
                   [np.sin(thetay / 2), np.cos(thetay / 2)]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    gatelist = [gates.Unitary(rx, 0), gates.Unitary(ry, 1),
                gates.Unitary(cnot, 0, 1)]
    final_state = apply_gates(gatelist, nqubits=2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [5, 6])
def test_variational_layer(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 2 * np.pi * np.random.random(nqubits)
    gatelist = [gates.RY(i, t) for i, t in enumerate(theta)]
    gatelist.extend(gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2))
    target_state = apply_gates(gatelist, nqubits=nqubits)

    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    gate = gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ,
                                  theta)
    final_state = apply_gates([gate], nqubits=nqubits)
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


def test_variational_layer_construct_unitary(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    pairs = list((i, i + 1) for i in range(0, 5, 2))
    theta = 2 * np.pi * np.random.random(6)
    gate = gates.VariationalLayer(range(6), pairs, gates.RY, gates.CZ, theta)
    with pytest.raises(ValueError):
        gate.construct_unitary()
    qibo.set_backend(original_backend)


def test_flatten(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    target_state = np.ones(4) / 2.0
    final_state = apply_gates([gates.Flatten(target_state)], nqubits=2)
    np.testing.assert_allclose(final_state, target_state)

    target_state = np.ones(4) / 2.0
    gate = gates.Flatten(target_state)
    with pytest.raises(ValueError):
        gate.construct_unitary()
    qibo.set_backend(original_backend)


def test_callback_gate_errors():
    from qibo import callbacks
    entropy = callbacks.EntanglementEntropy([0])
    gate = gates.CallbackGate(entropy)
    with pytest.raises(ValueError):
        gate.construct_unitary()


def test_general_channel(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from qibo import K
    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 0, 1], [0, 0, 1, 0]])
    a1, a2 = K.cast(a1), K.cast(a2)
    initial_rho = random_density_matrix(2)
    gate = gates.KrausChannel([((1,), a1), ((0, 1), a2)])
    assert gate.target_qubits == (0, 1)
    final_rho = gate(np.copy(initial_rho))
    m1 = np.kron(np.eye(2), np.array(a1))
    m2 = np.array(a2)
    target_rho = (m1.dot(initial_rho).dot(m1.conj().T) +
                  m2.dot(initial_rho).dot(m2.conj().T))
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


def test_krauss_channel_errors(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    # bad Kraus matrix shape
    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        gate = gates.KrausChannel([((0, 1), a1)])
    # Using KrausChannel on state vectors
    channel = gates.KrausChannel([((0,), np.eye(2))])
    with pytest.raises(ValueError):
        channel.state_vector_call(np.random.random(4))
    # Attempt to construct unitary for KrausChannel
    with pytest.raises(ValueError):
        channel.construct_unitary()
    qibo.set_backend(original_backend)


def test_controlled_by_channel_error():
    with pytest.raises(ValueError):
        gate = gates.PauliNoiseChannel(0, px=0.5).controlled_by(1)

    a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
    a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1],
                                  [0, 0, 1, 0]])
    config = [((1,), a1), ((0, 1), a2)]
    with pytest.raises(ValueError):
        gates.KrausChannel(config).controlled_by(1)

def test_unitary_channel(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    a1 = np.array([[0, 1], [1, 0]])
    a2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probs = [0.4, 0.3]
    matrices = [((0,), a1), ((2, 3), a2)]
    initial_state = random_density_matrix(4)
    gate = gates.UnitaryChannel(probs, matrices)
    gate.density_matrix = True
    final_state = gate(np.copy(initial_state))

    eye = np.eye(2)
    ma1 = np.kron(np.kron(a1, eye), np.kron(eye, eye))
    ma2 = np.kron(np.kron(eye, eye), a2)
    target_state = (0.3 * initial_state
                    + 0.4 * ma1.dot(initial_state.dot(ma1))
                    + 0.3 * ma2.dot(initial_state.dot(ma2)))
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_unitary_channel_errors():
    """Check errors raised by ``gates.UnitaryChannel``."""
    a1 = np.array([[0, 1], [1, 0]])
    a2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probs = [0.4, 0.3]
    matrices = [((0,), a1), ((2, 3), a2)]
    # Invalid probability length
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel([0.1, 0.3, 0.2], matrices)
    # Probability > 1
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel([1.1, 0.2], matrices)
    # Probability sum = 0
    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel([0.0, 0.0], matrices)


def test_pauli_noise_channel(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_rho = random_density_matrix(2)
    gate = gates.PauliNoiseChannel(1, px=0.3)
    gate.density_matrix = True
    final_rho = gate(np.copy(initial_rho))
    gate = gates.X(1)
    gate.density_matrix = True
    target_rho = 0.3 * gate(np.copy(initial_rho))
    target_rho += 0.7 * initial_rho
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


def test_reset_channel(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_rho = random_density_matrix(3)
    gate = gates.ResetChannel(0, p0=0.2, p1=0.2)
    gate.density_matrix = True
    final_rho = gate(np.copy(initial_rho))

    dtype = initial_rho.dtype
    collapsed_rho = np.copy(initial_rho).reshape(6 * (2,))
    collapsed_rho[0, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
    collapsed_rho[1, :, :, 0, :, :] = np.zeros(4 * (2,), dtype=dtype)
    collapsed_rho[1, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
    collapsed_rho = collapsed_rho.reshape((8, 8))
    collapsed_rho /= np.trace(collapsed_rho)
    mx = np.kron(np.array([[0, 1], [1, 0]]), np.eye(4))
    flipped_rho = mx.dot(collapsed_rho.dot(mx))
    target_rho = 0.6 * initial_rho + 0.2 * (collapsed_rho + flipped_rho)
    np.testing.assert_allclose(final_rho, target_rho)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(0.8, 0.5, 1.0, 0.4), (0.5, 0.8, 1.0, 0.4)])
def test_thermal_relaxation_channel(backend, t1, t2, time, excpop):
    """Check ``gates.ThermalRelaxationChannel`` on a 3-qubit random density matrix."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    initial_rho = random_density_matrix(3)
    gate = gates.ThermalRelaxationChannel(0, t1, t2, time=time,
        excited_population=excpop)
    gate.density_matrix = True
    final_rho = gate(np.copy(initial_rho))

    exp, p0, p1 = gate.calculate_probabilities(t1, t2, time, excpop)
    if t2 > t1:
        matrix = np.diag([1 - p1, p0, p1, 1 - p0])
        matrix[0, -1], matrix[-1, 0] = exp, exp
        matrix = matrix.reshape(4 * (2,))
        # Apply matrix using Eq. (3.28) from arXiv:1111.6950
        target_rho = np.copy(initial_rho).reshape(6 * (2,))
        target_rho = np.einsum("abcd,aJKcjk->bJKdjk", matrix, target_rho)
        target_rho = target_rho.reshape(initial_rho.shape)
    else:
        pz = exp
        pi = 1 - pz - p0 - p1
        dtype = initial_rho.dtype
        collapsed_rho = np.copy(initial_rho).reshape(6 * (2,))
        collapsed_rho[0, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
        collapsed_rho[1, :, :, 0, :, :] = np.zeros(4 * (2,), dtype=dtype)
        collapsed_rho[1, :, :, 1, :, :] = np.zeros(4 * (2,), dtype=dtype)
        collapsed_rho = collapsed_rho.reshape((8, 8))
        collapsed_rho /= np.trace(collapsed_rho)
        mx = np.kron(np.array([[0, 1], [1, 0]]), np.eye(4))
        mz = np.kron(np.array([[1, 0], [0, -1]]), np.eye(4))
        z_rho = mz.dot(initial_rho.dot(mz))
        flipped_rho = mx.dot(collapsed_rho.dot(mx))
        target_rho = (pi * initial_rho + pz * z_rho + p0 * collapsed_rho +
                      p1 * flipped_rho)
    np.testing.assert_allclose(final_rho, target_rho)
    # Try to apply to state vector if t1 < t2
    if t1 < t2:
        with pytest.raises(ValueError):
            gate.state_vector_call(initial_rho) # pylint: disable=no-member
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("t1,t2,time,excpop",
                         [(1.0, 0.5, 1.5, 1.5), (1.0, 0.5, -0.5, 0.5),
                          (1.0, -0.5, 1.5, 0.5), (-1.0, 0.5, 1.5, 0.5),
                          (1.0, 3.0, 1.5, 0.5)])
def test_thermal_relaxation_channel_errors(backend, t1, t2, time, excpop):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    with pytest.raises(ValueError):
        gate = gates.ThermalRelaxationChannel(
            0, t1, t2, time, excited_population=excpop)
    qibo.set_backend(original_backend)
