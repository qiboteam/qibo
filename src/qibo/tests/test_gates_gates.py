"""Test gates defined in `qibo/gates/gates.py`."""
import pytest
import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.tests.utils import random_state


def apply_gates(backend, gatelist, nqubits=None, initial_state=None):
    if initial_state is None:
        state = backend.zero_state(nqubits)
    elif isinstance(initial_state, np.ndarray):
        state = backend.cast(np.copy(initial_state))
        if nqubits is None:
            nqubits = int(np.log2(len(state)))
        else: # pragma: no cover
            assert nqubits == int(np.log2(len(state)))
    else: # pragma: no cover
        raise_error(TypeError, "Invalid initial state type {}."
                               "".format(type(initial_state)))

    for gate in gatelist:
        state = backend.apply_gate(gate, state, nqubits)
    return backend.to_numpy(state)


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


def test_align(backend):
    gate = gates.Align(0, 1)
    gatelist = [gates.H(0), gates.H(1), gate]
    final_state = apply_gates(backend, gatelist, nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    backend.assert_allclose(final_state, target_state)
    gate_matrix = gate.asmatrix(backend)
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
    target_state = np.dot(gate.asmatrix(backend), initial_state)
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
    target_state = np.ones_like(final_state) / 2.0
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
    target_state = np.ones_like(final_state) / np.sqrt(8)
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


############################# Test ``controlled_by`` #############################

def test_controlled_x(backend):
    gatelist = [gates.X(0), gates.X(1), gates.X(2),
                gates.X(3).controlled_by(0, 1, 2),
                gates.X(0), gates.X(2)]
    final_state = apply_gates(backend, gatelist, nqubits=4)
    gatelist = [gates.X(1), gates.X(3)]
    target_state = apply_gates(backend, gatelist, nqubits=4)
    backend.assert_allclose(final_state, target_state)


def test_controlled_x_vs_cnot(backend):
    gatelist = [gates.X(0), gates.X(2).controlled_by(0)]
    final_state = apply_gates(backend, gatelist, nqubits=3)
    gatelist = [gates.X(0), gates.CNOT(0, 2)]
    target_state = apply_gates(backend, gatelist, nqubits=3)
    backend.assert_allclose(final_state, target_state)


def test_controlled_x_vs_toffoli(backend):
    gatelist = [gates.X(0), gates.X(2), gates.X(1).controlled_by(0, 2)]
    final_state = apply_gates(backend, gatelist, nqubits=3)
    gatelist = [gates.X(0), gates.X(2), gates.TOFFOLI(0, 2, 1)]
    target_state = apply_gates(backend, gatelist, nqubits=3)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("applyx", [False, True])
def test_controlled_rx(backend, applyx):
    theta = 0.1234
    gatelist = [gates.X(0)]
    if applyx:
        gatelist.append(gates.X(1))
    gatelist.append(gates.RX(2, theta).controlled_by(0, 1))
    gatelist.append(gates.X(0))
    final_state = apply_gates(backend, gatelist, nqubits=3)

    gatelist = []
    if applyx:
        gatelist.extend([gates.X(1), gates.RX(2, theta)])
    target_state = apply_gates(backend, gatelist, nqubits=3)

    backend.assert_allclose(final_state, target_state)


def test_controlled_u1(backend):
    theta = 0.1234
    gatelist = [gates.X(i) for i in range(3)]
    gatelist.append(gates.U1(2, theta).controlled_by(0, 1))
    gatelist.append(gates.X(0))
    gatelist.append(gates.X(1))
    final_state = apply_gates(backend, gatelist, nqubits=3)
    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    backend.assert_allclose(final_state, target_state)
    gate = gates.U1(0, theta).controlled_by(1)
    assert gate.__class__.__name__ == "CU1"


def test_controlled_u2(backend):
    phi = 0.1234
    lam = 0.4321
    gatelist = [gates.X(0), gates.X(1)]
    gatelist.append(gates.U2(2, phi, lam).controlled_by(0, 1))
    gatelist.extend([gates.X(0), gates.X(1)])
    final_state = apply_gates(backend, gatelist, nqubits=3)
    gatelist = [gates.X(0), gates.X(1), gates.U2(2, phi, lam),
                gates.X(0), gates.X(1)]
    target_state = apply_gates(backend, gatelist, nqubits=3)
    backend.assert_allclose(final_state, target_state)
    # for coverage
    gate = gates.CU2(0, 1, phi, lam)
    assert gate.parameters == (phi, lam)


def test_controlled_u3(backend):
    theta, phi, lam = 0.1, 0.1234, 0.4321
    initial_state = random_state(2)
    final_state = apply_gates(backend, [gates.U3(1, theta, phi, lam).controlled_by(0)], 2, initial_state)
    target_state = apply_gates(backend, [gates.CU3(0, 1, theta, phi, lam)], 2, initial_state)
    backend.assert_allclose(final_state, target_state)
    # for coverage
    gate = gates.U3(0, theta, phi, lam)
    assert gate.parameters == (theta, phi, lam)


@pytest.mark.parametrize("applyx", [False, True])
@pytest.mark.parametrize("free_qubit", [False, True])
def test_controlled_swap(backend, applyx, free_qubit):
    f = int(free_qubit)
    gatelist = []
    if applyx:
        gatelist.append(gates.X(0))
    gatelist.extend([gates.RX(1 + f, theta=0.1234), gates.RY(2 + f, theta=0.4321),
                     gates.SWAP(1 + f, 2 + f).controlled_by(0)])
    final_state = apply_gates(backend, gatelist, 3 + f)
    gatelist = [gates.RX(1 + f, theta=0.1234), gates.RY(2 + f, theta=0.4321)]
    if applyx:
        gatelist.extend([gates.X(0), gates.SWAP(1 + f, 2 + f)])
    target_state = apply_gates(backend, gatelist, 3 + f)
    backend.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("applyx", [False, True])
def test_controlled_swap_double(backend, applyx):
    gatelist = [gates.X(0)]
    if applyx:
        gatelist.append(gates.X(3))
    gatelist.append(gates.RX(1, theta=0.1234))
    gatelist.append(gates.RY(2, theta=0.4321))
    gatelist.append(gates.SWAP(1, 2).controlled_by(0, 3))
    gatelist.append(gates.X(0))
    final_state = apply_gates(backend, gatelist, 4)

    gatelist = [gates.RX(1, theta=0.1234), gates.RY(2, theta=0.4321)]
    if applyx:
        gatelist.extend([gates.X(3), gates.SWAP(1, 2)])
    target_state = apply_gates(backend, gatelist, 4)
    backend.assert_allclose(final_state, target_state)


def test_controlled_fsim(backend):
    theta, phi = 0.1234, 0.4321
    gatelist = [gates.H(i) for i in range(6)]
    gatelist.append(gates.fSim(5, 3, theta, phi).controlled_by(0, 2, 1))
    final_state = apply_gates(backend, gatelist, 6)

    target_state = np.ones_like(final_state) / np.sqrt(2 ** 6)
    rotation = np.array([[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]])
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    ids = [56, 57, 60, 61]
    target_state[ids] = matrix.dot(target_state[ids])
    ids = [58, 59, 62, 63]
    target_state[ids] = matrix.dot(target_state[ids])
    backend.assert_allclose(final_state, target_state)


def test_controlled_unitary(backend):
    matrix = np.random.random((2, 2))
    gatelist = [gates.H(0), gates.H(1),
                gates.Unitary(matrix, 1).controlled_by(0)]
    final_state = apply_gates(backend, gatelist, 2)
    target_state = np.ones_like(final_state) / 2.0
    target_state[2:] = matrix.dot(target_state[2:])
    backend.assert_allclose(final_state, target_state)

    matrix = np.random.random((4, 4))
    gatelist = [gates.H(i) for i in range(4)]
    gatelist.append(gates.Unitary(matrix, 1, 3).controlled_by(0, 2))
    final_state = apply_gates(backend, gatelist, 4)
    target_state = np.ones_like(final_state) / 4.0
    ids = [10, 11, 14, 15]
    target_state[ids] = matrix.dot(target_state[ids])
    backend.assert_allclose(final_state, target_state)


def test_controlled_unitary_matrix(backend):
    initial_state = random_state(2)
    matrix = np.random.random((2, 2))
    gate = gates.Unitary(matrix, 1).controlled_by(0)
    target_state = apply_gates(backend, [gate], 2, initial_state)
    u = backend.control_matrix(gate)
    final_state = np.dot(u, initial_state)
    backend.assert_allclose(final_state, target_state)

###############################################################################

################################# Test dagger #################################
GATES = [
    ("H", (0,)),
    ("X", (0,)),
    ("Y", (0,)),
    ("Z", (0,)),
    ("S", (0,)),
    ("SDG", (0,)),
    ("T", (0,)),
    ("TDG", (0,)),
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
    initial_state = random_state(nqubits)
    final_state = apply_gates(backend, [gate, gate.dagger()], nqubits, initial_state)
    backend.assert_allclose(final_state, initial_state)


GATES = [
    ("H", (3,)),
    ("X", (3,)),
    ("Y", (3,)),
    ("S", (3,)),
    ("SDG", (3,)),
    ("T", (3,)),
    ("TDG", (3,)),
    ("RX", (3, 0.1)),
    ("U1", (3, 0.1)),
    ("U3", (3, 0.1, 0.2, 0.3))
]
@pytest.mark.parametrize("gate,args", GATES)
def test_controlled_dagger(backend, gate, args):
    gate = getattr(gates, gate)(*args).controlled_by(0, 1, 2)
    initial_state = random_state(4)
    final_state = apply_gates(backend, [gate, gate.dagger()], 4, initial_state)
    backend.assert_allclose(final_state, initial_state)


@pytest.mark.parametrize("gate_1,gate_2", [("S", "SDG"), ("T", "TDG")])
@pytest.mark.parametrize("qubit", (0, 2, 4))
def test_dagger_consistency(backend, gate_1, gate_2, qubit):
    gate_1 = getattr(gates, gate_1)(qubit)
    gate_2 = getattr(gates, gate_2)(qubit)
    initial_state = random_state(qubit + 1)
    final_state = apply_gates(backend, [gate_1, gate_2], qubit + 1, initial_state)
    backend.assert_allclose(final_state, initial_state)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_dagger(backend, nqubits):
    matrix = np.random.random((2 ** nqubits, 2 ** nqubits))
    gate = gates.Unitary(matrix, *range(nqubits))
    initial_state = random_state(nqubits)
    final_state = apply_gates(backend, [gate, gate.dagger()], nqubits, initial_state)
    target_state = np.dot(matrix, initial_state)
    target_state = np.dot(np.conj(matrix).T, target_state)
    backend.assert_allclose(final_state, target_state)


def test_controlled_unitary_dagger(backend):
    from scipy.linalg import expm
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.Unitary(matrix, 0).controlled_by(1, 2, 3, 4)
    initial_state = random_state(5)
    final_state = apply_gates(backend, [gate, gate.dagger()], 5, initial_state)
    backend.assert_allclose(final_state, initial_state)


def test_generalizedfsim_dagger(backend):
    from scipy.linalg import expm
    phi = 0.2
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    initial_state = random_state(2)
    final_state = apply_gates(backend, [gate, gate.dagger()], 2, initial_state)
    backend.assert_allclose(final_state, initial_state)

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
    target_state = backend.apply_gate(gate, np.copy(initial_state), nqubits)
    dgates = gate.decompose(*free, use_toffolis=use_toffolis)
    final_state = np.copy(initial_state)
    for gate in dgates:
        final_state = backend.apply_gate(gate, final_state, nqubits)
    backend.assert_allclose(final_state, target_state, atol=1e-6)

###############################################################################
