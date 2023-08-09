"""Test execution of `controlled_by` gates."""
import pytest
import numpy as np
from qibo import gates, K
from qibo.models import Circuit
from qibo.tests.utils import random_state


def test_controlled_x(backend, accelerators):
    c = Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3).controlled_by(0, 1, 2))
    c.add(gates.X(0))
    c.add(gates.X(2))
    target_c = Circuit(4)
    target_c.add(gates.X(1))
    target_c.add(gates.X(3))
    K.assert_allclose(c(), target_c())


def test_controlled_x_vs_cnot(backend):
    c1 = Circuit(3)
    c1.add(gates.X(0))
    c1.add(gates.X(2).controlled_by(0))
    c2 = Circuit(3)
    c2.add(gates.X(0))
    c2.add(gates.CNOT(0, 2))
    K.assert_allclose(c1(), c2())


def test_controlled_x_vs_toffoli(backend):
    c1 = Circuit(3)
    c1.add(gates.X(0))
    c1.add(gates.X(2))
    c1.add(gates.X(1).controlled_by(0, 2))
    c2 = Circuit(3)
    c2.add(gates.X(0))
    c2.add(gates.X(2))
    c2.add(gates.TOFFOLI(0, 2, 1))
    K.assert_allclose(c1(), c2())


@pytest.mark.parametrize("applyx", [False, True])
def test_controlled_rx(backend, applyx):
    theta = 0.1234
    c = Circuit(3)
    c.add(gates.X(0))
    if applyx:
        c.add(gates.X(1))
    c.add(gates.RX(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    final_state = c.execute()
    c = Circuit(3)
    if applyx:
        c.add(gates.X(1))
        c.add(gates.RX(2, theta))
    target_state = c.execute()
    K.assert_allclose(final_state, target_state)


def test_controlled_u1(backend):
    theta = 0.1234
    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.U1(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    c.add(gates.X(1))
    final_state = c.execute()
    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    K.assert_allclose(final_state, target_state)
    gate = gates.U1(0, theta).controlled_by(1)
    assert gate.__class__.__name__ == "CU1"


def test_controlled_u2(backend):
    phi = 0.1234
    lam = 0.4321

    c = Circuit(3)
    c.add([gates.X(0), gates.X(1)])
    c.add(gates.U2(2, phi, lam).controlled_by(0, 1))
    c.add([gates.X(0), gates.X(1)])
    final_state = c()

    c = Circuit(3)
    c.add([gates.X(0), gates.X(1)])
    c.add(gates.U2(2, phi, lam))
    c.add([gates.X(0), gates.X(1)])
    target_state = c()
    K.assert_allclose(final_state, target_state)

    # for coverage
    gate = gates.CU2(0, 1, phi, lam)
    assert gate.parameters == (phi, lam)


def test_controlled_u3(backend):
    theta, phi, lam = 0.1, 0.1234, 0.4321
    initial_state = random_state(2)
    c = Circuit(2)
    c.add(gates.U3(1, theta, phi, lam).controlled_by(0))
    final_state = c(np.copy(initial_state))
    assert c.queue[0].__class__.__name__ == "CU3"

    c = Circuit(2)
    c.add(gates.CU3(0, 1, theta, phi, lam))
    target_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, target_state)

    # for coverage
    gate = gates.U3(0, theta, phi, lam)
    assert gate.parameters == (theta, phi, lam)


@pytest.mark.parametrize("applyx", [False, True])
@pytest.mark.parametrize("free_qubit", [False, True])
def test_controlled_swap(backend, applyx, free_qubit):
    f = int(free_qubit)
    c = Circuit(3 + f)
    if applyx:
        c.add(gates.X(0))
    c.add(gates.RX(1 + f, theta=0.1234))
    c.add(gates.RY(2 + f, theta=0.4321))
    c.add(gates.SWAP(1 + f, 2 + f).controlled_by(0))
    final_state = c.execute()
    c = Circuit(3 + f)
    c.add(gates.RX(1 + f, theta=0.1234))
    c.add(gates.RY(2 + f, theta=0.4321))
    if applyx:
        c.add(gates.X(0))
        c.add(gates.SWAP(1 + f, 2 + f))
    target_state = c.execute()
    K.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("applyx", [False, True])
def test_controlled_swap_double(backend, applyx):
    c = Circuit(4)
    c.add(gates.X(0))
    if applyx:
        c.add(gates.X(3))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0, 3))
    c.add(gates.X(0))
    final_state = c.execute()
    c = Circuit(4)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    if applyx:
        c.add(gates.X(3))
        c.add(gates.SWAP(1, 2))
    target_state = c.execute()
    K.assert_allclose(final_state, target_state)


def test_controlled_fsim(backend, accelerators):
    theta, phi = 0.1234, 0.4321
    c = Circuit(6, accelerators)
    c.add((gates.H(i) for i in range(6)))
    c.add(gates.fSim(5, 3, theta, phi).controlled_by(0, 2, 1))
    final_state = c.execute()

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
    K.assert_allclose(final_state, target_state)


def test_controlled_unitary(backend, accelerators):
    matrix = np.random.random((2, 2))
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.Unitary(matrix, 1).controlled_by(0))
    final_state = c.execute()
    target_state = np.ones_like(final_state) / 2.0
    target_state[2:] = matrix.dot(target_state[2:])
    K.assert_allclose(final_state, target_state)

    matrix = np.random.random((4, 4))
    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    c.add(gates.Unitary(matrix, 1, 3).controlled_by(0, 2))
    final_state = c.execute()
    target_state = np.ones_like(final_state) / 4.0
    ids = [10, 11, 14, 15]
    target_state[ids] = matrix.dot(target_state[ids])
    K.assert_allclose(final_state, target_state)


def test_controlled_unitary_matrix(backend):
    initial_state = random_state(2)
    matrix = np.random.random((2, 2))
    gate = gates.Unitary(matrix, 1).controlled_by(0)
    c = Circuit(2)
    c.add(gate)
    target_state = c(np.copy(initial_state))
    final_state = np.dot(K.to_numpy(gate.matrix), initial_state)
    K.assert_allclose(final_state, target_state)
