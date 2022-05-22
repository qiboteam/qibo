"""Test gates defined in `qibo/core/gates.py`."""
import pytest
import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.tests.utils import random_state, random_density_matrix


def apply_gates(backend, gatelist, nqubits=None, initial_state=None):
    if initial_state is None:
        state = backend.zero_state(nqubits)
    elif isinstance(initial_state, np.ndarray):
        state = np.copy(initial_state)
        if nqubits is None:
            nqubits = int(np.log2(len(state)))
        else: # pragma: no cover
            assert nqubits == int(np.log2(len(state)))
    else: # pragma: no cover
        raise_error(TypeError, "Invalid initial state type {}."
                               "".format(type(initial_state)))

    for gate in gatelist:
        state = backend.apply_gate(gate, state, nqubits)
    return state


@pytest.mark.skip
def test__control_unitary(backend):
    # TODO: Move this to backend tests
    matrix = np.random.random((2, 2))
    gate = gates.Unitary(matrix, 0)
    unitary = gate._control_unitary(matrix)
    target_unitary = np.eye(4, dtype=backend.dtype)
    target_unitary[2:, 2:] = backend.to_numpy(matrix)
    backend.assert_allclose(unitary, target_unitary)
    with pytest.raises(ValueError):
        unitary = gate._control_unitary(np.random.random((16, 16)))


def test_h(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1)], nqubits=2)
    target_state = np.ones_like(final_state) / 2
    backend.assert_allclose(final_state, target_state)


def test_x(backend):
    final_state = apply_gates(backend, [gates.X(0)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    backend.assert_allclose(final_state, target_state)


def test_y(backend):
    final_state = apply_gates(backend, [gates.Y(1)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[1] = 1j
    backend.assert_allclose(final_state, target_state)


def test_z(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.Z(0)], nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    target_state[2] *= -1.0
    target_state[3] *= -1.0
    backend.assert_allclose(final_state, target_state)


def test_s(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.S(1)], nqubits=2)
    target_state = np.array([0.5, 0.5j, 0.5, 0.5j])
    backend.assert_allclose(final_state, target_state)


def test_sdg(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.SDG(1)], nqubits=2)
    target_state = np.array([0.5, -0.5j, 0.5, -0.5j])
    backend.assert_allclose(final_state, target_state)


def test_t(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.T(1)], nqubits=2)
    target_state = np.array([0.5, (1 + 1j) / np.sqrt(8),
                             0.5, (1 + 1j) / np.sqrt(8)])
    backend.assert_allclose(final_state, target_state)


def test_tdg(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.H(1), gates.TDG(1)], nqubits=2)
    target_state = np.array([0.5, (1 - 1j) / np.sqrt(8),
                             0.5, (1 - 1j) / np.sqrt(8)])
    backend.assert_allclose(final_state, target_state)


def test_identity(backend):
    gatelist = [gates.H(0), gates.H(1), gates.I(0), gates.I(1)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    backend.assert_allclose(final_state, target_state)
    gatelist = [gates.H(0), gates.H(1), gates.I(0, 1)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.skip
def test_align(backend):
    gate = gates.Align(0, 1)
    gatelist = [gates.H(0), gates.H(1), gate]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    backend.assert_allclose(final_state, target_state)
    gate_matrix = gate._construct_unitary()
    backend.assert_allclose(gate_matrix, np.eye(4))


# :class:`qibo.core.cgates.M` is tested seperately in `test_measurement_gate.py`

def test_rx(backend):
    theta = 0.1234
    final_state = apply_gates(backend, [gates.H(0), gates.RX(0, theta=theta)], nqubits=1)
    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -1j * phase.imag],
                    [-1j * phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    backend.assert_allclose(final_state, target_state)


def test_ry(backend):
    theta = 0.1234
    final_state = apply_gates(backend, [gates.H(0), gates.RY(0, theta=theta)], nqubits=1)
    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -phase.imag],
                     [phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("applyx", [True, False])
def test_rz(backend, applyx):
    theta = 0.1234
    if applyx:
        gatelist = [gates.X(0)]
    else:
        gatelist = []
    gatelist.append(gates.RZ(0, theta))
    final_state = apply_gates(backend, gatelist, nqubits=1)
    target_state = np.zeros_like(final_state)
    p = int(applyx)
    target_state[p] = np.exp((2 * p - 1) * 1j * theta / 2.0)
    backend.assert_allclose(final_state, target_state)


def test_u1(backend):
    theta = 0.1234
    final_state = apply_gates(backend, [gates.X(0), gates.U1(0, theta)], nqubits=1)
    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    backend.assert_allclose(final_state, target_state)


def test_u2(backend):
    phi = 0.1234
    lam = 0.4321
    initial_state = random_state(1)
    final_state = apply_gates(backend, [gates.U2(0, phi, lam)], initial_state=initial_state)
    matrix = np.array([[np.exp(-1j * (phi + lam) / 2), -np.exp(-1j * (phi - lam) / 2)],
                       [np.exp(1j * (phi - lam) / 2), np.exp(1j * (phi + lam) / 2)]])
    target_state = matrix.dot(initial_state) / np.sqrt(2)
    backend.assert_allclose(final_state, target_state)


def test_u3(backend):
    theta = 0.1111
    phi = 0.1234
    lam = 0.4321
    initial_state = random_state(1)
    final_state = apply_gates(backend, [gates.U3(0, theta, phi, lam)],
                              initial_state=initial_state)
    cost, sint = np.cos(theta / 2), np.sin(theta / 2)
    ep = np.exp(1j * (phi + lam) / 2)
    em = np.exp(1j * (phi - lam) / 2)
    matrix = np.array([[ep.conj() * cost, - em.conj() * sint],
                       [em * sint, ep * cost]])
    target_state = matrix.dot(initial_state)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("applyx", [False, True])
def test_cnot(backend, applyx):
    if applyx:
        gatelist = [gates.X(0)]
    else:
        gatelist = []
    gatelist.append(gates.CNOT(0, 1))
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[3 * int(applyx)] = 1.0
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("controlled_by", [False, True])
def test_cz(backend, controlled_by):
    initial_state = random_state(2)
    matrix = np.eye(4)
    matrix[3, 3] = -1
    target_state = matrix.dot(initial_state)
    if controlled_by:
        gate = gates.Z(1).controlled_by(0)
    else:
        gate = gates.CZ(0, 1)
    final_state = apply_gates(backend, [gate], initial_state=initial_state)
    assert gate.name == "cz"
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("name,params",
                         [("CRX", {"theta": 0.1}),
                          ("CRY", {"theta": 0.2}),
                          ("CRZ", {"theta": 0.3}),
                          ("CU1", {"theta": 0.1}),
                          ("CU2", {"phi": 0.1, "lam": 0.2}),
                          ("CU3", {"theta": 0.1, "phi": 0.2, "lam": 0.3})])
def test_cun(backend, name, params):
    initial_state = random_state(2)
    gate = getattr(gates, name)(0, 1, **params)
    final_state = apply_gates(backend, [gate], initial_state=initial_state)
    target_state = np.dot(backend.asmatrix(gate), initial_state)
    backend.assert_allclose(final_state, target_state)


def test_swap(backend):
    final_state = apply_gates(backend, [gates.X(1), gates.SWAP(0, 1)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    backend.assert_allclose(final_state, target_state)


def test_fswap(backend):
    final_state = apply_gates(backend, [gates.H(0), gates.X(1), gates.FSWAP(0, 1)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0 / np.sqrt(2)
    target_state[3] = -1.0 / np.sqrt(2)
    backend.assert_allclose(final_state, target_state)


def test_multiple_swap(backend):
    gatelist = [gates.X(0), gates.X(2), gates.SWAP(0, 1), gates.SWAP(2, 3)]
    final_state = apply_gates(backend, gatelist, nqubits=4)
    gatelist = [gates.X(1), gates.X(3)]
    target_state = apply_gates(backend, gatelist, nqubits=4)
    backend.assert_allclose(final_state, target_state)


def test_fsim(backend):
    theta = 0.1234
    phi = 0.4321
    gatelist = [gates.H(0), gates.H(1), gates.fSim(0, 1, theta, phi)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.ones_like(backend.to_numpy(final_state)) / 2.0
    rotation = np.array([[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]])
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state = matrix.dot(target_state)
    backend.assert_allclose(final_state, target_state)


def test_generalized_fsim(backend):
    phi = np.random.random()
    rotation = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    gatelist = [gates.H(0), gates.H(1), gates.H(2)]
    gatelist.append(gates.GeneralizedfSim(1, 2, rotation, phi))
    final_state = apply_gates(backend, gatelist, nqubits=3)
    target_state = np.ones_like(backend.to_numpy(final_state)) / np.sqrt(8)
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state[:4] = matrix.dot(target_state[:4])
    target_state[4:] = matrix.dot(target_state[4:])
    backend.assert_allclose(final_state, target_state)


def test_generalized_fsim_parameter_setter(backend):
    phi = np.random.random()
    matrix = np.random.random((2, 2))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    backend.assert_allclose(gate.parameters[0], matrix)
    assert gate.parameters[1] == phi
    matrix = np.random.random((4, 4))
    with pytest.raises(ValueError):
        gate = gates.GeneralizedfSim(0, 1, matrix, phi)


@pytest.mark.parametrize("applyx", [False, True])
def test_toffoli(backend, applyx):
    if applyx:
        gatelist = [gates.X(0), gates.X(1), gates.TOFFOLI(0, 1, 2)]
    else:
        gatelist = [gates.X(1), gates.TOFFOLI(0, 1, 2)]
    final_state = apply_gates(backend, gatelist, nqubits=3)
    target_state = np.zeros_like(final_state)
    if applyx:
        target_state[-1] = 1
    else:
        target_state[2] = 1
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("nqubits", [2, 3])
def test_unitary(backend, nqubits):
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    matrix = np.random.random(2 * (2 ** (nqubits - 1),))
    target_state = np.kron(np.eye(2), matrix).dot(initial_state)
    gatelist = [gates.H(i) for i in range(nqubits)]
    gatelist.append(gates.Unitary(matrix, *range(1, nqubits), name="random"))
    final_state = apply_gates(backend, gatelist, nqubits=nqubits)
    backend.assert_allclose(final_state, target_state)


def test_unitary_initialization(backend):
    matrix = np.random.random((4, 4))
    gate = gates.Unitary(matrix, 0, 1)
    backend.assert_allclose(gate.parameters[0], matrix)


def test_unitary_common_gates(backend):
    target_state = apply_gates(backend, [gates.X(0), gates.H(1)], nqubits=2)
    gatelist = [gates.Unitary(np.array([[0, 1], [1, 0]]), 0),
                gates.Unitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), 1)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    backend.assert_allclose(final_state, target_state)

    thetax = 0.1234
    thetay = 0.4321
    gatelist = [gates.RX(0, theta=thetax), gates.RY(1, theta=thetay),
                gates.CNOT(0, 1)]
    target_state = apply_gates(backend, gatelist, nqubits=2)

    rx = np.array([[np.cos(thetax / 2), -1j * np.sin(thetax / 2)],
                   [-1j * np.sin(thetax / 2), np.cos(thetax / 2)]])
    ry = np.array([[np.cos(thetay / 2), -np.sin(thetay / 2)],
                   [np.sin(thetay / 2), np.cos(thetay / 2)]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    gatelist = [gates.Unitary(rx, 0), gates.Unitary(ry, 1),
                gates.Unitary(cnot, 0, 1)]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    backend.assert_allclose(final_state, target_state)


def test_unitary_multiqubit(backend):
    gatelist = [gates.H(i) for i in range(4)]
    gatelist.append(gates.CNOT(0, 1))
    gatelist.append(gates.CNOT(2, 3))
    gatelist.extend(gates.X(i) for i in range(4))

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    x = np.array([[0, 1], [1, 0]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    matrix = np.kron(np.kron(x, x), np.kron(x, x))
    matrix = matrix @ np.kron(cnot, cnot)
    matrix = matrix @ np.kron(np.kron(h, h), np.kron(h, h))
    unitary = gates.Unitary(matrix, 0, 1, 2, 3)
    final_state = apply_gates(backend, [unitary], nqubits=4)
    target_state = apply_gates(backend, gatelist, nqubits=4)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.skip
@pytest.mark.parametrize("nqubits", [5, 6])
def test_variational_layer(backend, nqubits):
    theta = 2 * np.pi * np.random.random(nqubits)
    gatelist = [gates.RY(i, t) for i, t in enumerate(theta)]
    gatelist.extend(gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2))
    target_state = apply_gates(backend, gatelist, nqubits=nqubits)

    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    gate = gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ,
                                  theta)
    final_state = apply_gates(backend, [gate], nqubits=nqubits)
    backend.assert_allclose(target_state, final_state)


@pytest.mark.skip
def test_variational_layer__construct_unitary(backend):
    pairs = list((i, i + 1) for i in range(0, 5, 2))
    theta = 2 * np.pi * np.random.random(6)
    gate = gates.VariationalLayer(range(6), pairs, gates.RY, gates.CZ, theta)
    with pytest.raises(ValueError):
        gate._construct_unitary()


@pytest.mark.skip
def test_flatten(backend):
    target_state = np.ones(4) / 2.0
    final_state = apply_gates(backend, [gates.Flatten(target_state)], nqubits=2)
    backend.assert_allclose(final_state, target_state)

    target_state = np.ones(4) / 2.0
    gate = gates.Flatten(target_state)
    with pytest.raises(ValueError):
        gate._construct_unitary()


@pytest.mark.skip
def test_callback_gate_errors():
    from qibo import callbacks
    entropy = callbacks.EntanglementEntropy([0])
    gate = gates.CallbackGate(entropy)
    with pytest.raises(ValueError):
        gate._construct_unitary()


@pytest.mark.skip
@pytest.mark.parametrize("nqubits", [2, 3])
def test_fused_gate_construct_unitary(backend, nqubits):
    gate = gates.FusedGate(0, 1)
    gate.append(gates.H(0))
    gate.append(gates.H(1))
    gate.append(gates.CZ(0, 1))
    hmatrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    czmatrix = np.diag([1, 1, 1, -1])
    target_matrix = czmatrix @ np.kron(hmatrix, hmatrix)
    if nqubits > 2:
        gate.append(gates.TOFFOLI(0, 1, 2))
        toffoli = np.eye(8)
        toffoli[-2:, -2:] = np.array([[0, 1], [1, 0]])
        target_matrix = toffoli @ np.kron(target_matrix, np.eye(2))
    backend.assert_allclose(backend.asmatrix(gate), target_matrix)
