"""
Testing Tensorflow gates.
"""
import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit
from qibo.tests import utils

_BACKENDS = ["custom", "defaulteinsum", "matmuleinsum"]
_DEVICE_BACKENDS = [("custom", None), ("matmuleinsum", None),
                    ("custom", {"/GPU:0": 1, "/GPU:1": 1})]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_hadamard(backend):
    """Check Hadamard gate is working properly."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_flatten(backend):
    """Check ``Flatten`` gate works in circuits ."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    target_state = np.ones(4) / 2.0
    c = Circuit(2)
    c.add(gates.Flatten(target_state))
    final_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)
    gate = gates.Flatten(target_state)
    gate(final_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_xgate(backend, accelerators):
    """Check X gate is working properly."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2, accelerators)
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ygate(backend):
    """Check Y gate is working properly."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.Y(1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[1] = 1j
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_zgate(backend, accelerators):
    """Check Z gate is working properly."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.Z(0))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2.0
    target_state[2] *= -1.0
    target_state[3] *= -1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_identity_gate(backend, accelerators):
    """Check identity gate is working properly."""
    qibo.set_backend(backend)
    c = Circuit(2, accelerators)
    c.add((gates.H(i) for i in range(2)))
    c.add(gates.I(0))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2.0
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    c.add(gates.I(0, 1))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 4.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_multicontrol_xgate(backend):
    """Check that fallback method for X works for one or two controls."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c1 = Circuit(3)
    c1.add(gates.X(0))
    c1.add(gates.X(2).controlled_by(0))
    final_state = c1.execute().numpy()
    c2 = Circuit(3)
    c2.add(gates.X(0))
    c2.add(gates.CNOT(0, 2))
    target_state = c2.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c1 = Circuit(3)
    c1.add(gates.X(0))
    c1.add(gates.X(2))
    c1.add(gates.X(1).controlled_by(0, 2))
    final_state = c1.execute().numpy()
    c2 = Circuit(3)
    c2.add(gates.X(0))
    c2.add(gates.X(2))
    c2.add(gates.TOFFOLI(0, 2, 1))
    target_state = c2.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_multicontrol_xgate_more_controls(backend, accelerators):
    """Check that fallback method for X works for more than two controls."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3).controlled_by(0, 1, 2))
    c.add(gates.X(0))
    c.add(gates.X(2))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rz_phase0(backend):
    """Check RZ gate is working properly when qubit is on |0>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.RZ(0, theta))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[0] = np.exp(-1j * theta / 2.0)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rz_phase1(backend):
    """Check RZ gate is working properly when qubit is on |1>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.RZ(0, theta))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[2] = np.exp(1j * theta / 2.0)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_u1(backend):
    """Check U1 gate is working properly when qubit is on |1>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.X(0))
    c.add(gates.U1(0, theta))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_u2(backend):
    """Check U2 gate on random state."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = 0.1234
    lam = 0.4321

    initial_state = utils.random_numpy_state(1)
    c = Circuit(1)
    c.add(gates.U2(0, phi, lam))
    final_state = c(np.copy(initial_state))

    matrix = np.array([[np.exp(-1j * (phi + lam) / 2), -np.exp(-1j * (phi - lam) / 2)],
                       [np.exp(1j * (phi - lam) / 2), np.exp(1j * (phi + lam) / 2)]])
    target_state = matrix.dot(initial_state) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_u3(backend):
    """Check U3 gate on random state."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1111
    phi = 0.1234
    lam = 0.4321

    initial_state = utils.random_numpy_state(1)
    c = Circuit(1)
    c.add(gates.U3(0, theta, phi, lam))
    final_state = c(np.copy(initial_state))

    cost, sint = np.cos(theta / 2), np.sin(theta / 2)
    ep = np.exp(1j * (phi + lam) / 2)
    em = np.exp(1j * (phi - lam) / 2)
    matrix = np.array([[ep.conj() * cost, - em.conj() * sint],
                       [em * sint, ep * cost]])
    target_state = matrix.dot(initial_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_u1(backend):
    """Check controlled U1 and fallback to CU1."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.U1(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    c.add(gates.X(1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)

    gate = gates.U1(0, theta).controlled_by(1)
    assert gate.__class__.__name__ == "CU1"
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_u2(backend):
    """Check controlled by U2."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
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
    np.testing.assert_allclose(final_state, target_state)

    # for coverage
    gate = gates.CU2(0, 1, phi, lam)
    assert gate.parameters == (phi, lam)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_u3(backend):
    """Check controlled U3 fall backs to CU3."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1
    phi = 0.1234
    lam = 0.4321

    initial_state = utils.random_numpy_state(2)
    c = Circuit(2)
    c.add(gates.U3(1, theta, phi, lam).controlled_by(0))
    final_state = c(np.copy(initial_state))
    assert c.queue[0].__class__.__name__ == "CU3"

    c = Circuit(2)
    c.add(gates.CU3(0, 1, theta, phi, lam))
    target_state = c(np.copy(initial_state))
    np.testing.assert_allclose(final_state, target_state)

    # for coverage
    gate = gates.U3(0, theta, phi, lam)
    assert gate.parameters == (theta, phi, lam)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_zpow_gate(backend):
    """Check ZPow and CZPow gate fall back to U1 and CU1 respectively."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.X(0))
    c.add(gates.ZPow(0, theta))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[1] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)
    assert c.queue[1].name == "u1"

    c = Circuit(2)
    c.add([gates.X(0), gates.X(1)])
    c.add(gates.CZPow(0, 1, theta))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[-1] = np.exp(1j * theta)
    np.testing.assert_allclose(final_state, target_state)
    assert c.queue[2].name == "cu1"

    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rx(backend):
    """Check RX gate is working properly."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RX(0, theta=theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -1j * phase.imag],
                    [-1j * phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ry(backend):
    """Check RY gate is working properly."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RY(0, theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -phase.imag],
                     [phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cnot_no_effect(backend):
    """Check CNOT gate is working properly on |00>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.CNOT(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cnot(backend):
    """Check CNOT gate is working properly on |10>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.CNOT(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[3] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cz(backend):
    """Check CZ gate is working properly on random state."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    init_state = utils.random_numpy_state(2)
    matrix = np.eye(4)
    matrix[3, 3] = -1
    target_state = matrix.dot(init_state)
    c = Circuit(2)
    c.add(gates.CZ(0, 1))
    final_state = c.execute(np.copy(init_state)).numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(2)
    c.add(gates.Z(1).controlled_by(0))
    final_state = c.execute(np.copy(init_state)).numpy()
    assert c.queue[0].name == "cz"
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_cu1(backend, accelerators):
    """Check CU1 gate is working properly on |11>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CU1(0, 1, theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta)
    target_state = np.zeros_like(final_state)
    target_state[3] = phase
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fsim(backend):
    """Check fSim gate is working properly on |++>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    phi = 0.4321

    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.fSim(0, 1, theta, phi))
    final_state = c.execute().numpy()

    target_state = np.ones_like(final_state) / 2.0
    rotation = np.array([[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]])
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state = matrix.dot(target_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_controlled_by_fsim(backend, accelerators):
    """Check ``controlled_by`` for fSim gate."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    phi = 0.4321

    c = Circuit(6, accelerators)
    c.add((gates.H(i) for i in range(6)))
    c.add(gates.fSim(5, 3, theta, phi).controlled_by(0, 2, 1))
    final_state = c.execute().numpy()

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
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_generalized_fsim(backend, accelerators):
    """Check GeneralizedfSim gate is working properly on |++>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = np.random.random()
    rotation = utils.random_numpy_complex((2, 2))

    c = Circuit(3, accelerators)
    c.add((gates.H(i) for i in range(3)))
    c.add(gates.GeneralizedfSim(1, 2, rotation, phi))
    final_state = c.execute().numpy()

    target_state = np.ones_like(final_state) / np.sqrt(8)
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state[:4] = matrix.dot(target_state[:4])
    target_state[4:] = matrix.dot(target_state[4:])
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_generalized_fsim_error(backend):
    """Check GenerelizedfSim gate raises error for wrong unitary shape."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    phi = np.random.random()
    rotation = utils.random_numpy_complex((4, 4))
    c = Circuit(2)
    with pytest.raises(ValueError):
        c.add(gates.GeneralizedfSim(0, 1, rotation, phi))
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_doubly_controlled_by_rx_no_effect(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.RX(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0

    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_doubly_controlled_by_rx(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(3, accelerators)
    c.add(gates.RX(2, theta))
    target_state = c.execute().numpy()

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.RX(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    c.add(gates.X(1))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_swap(backend):
    """Check SWAP gate is working properly on |01>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.X(1))
    c.add(gates.SWAP(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_multiple_swap(backend):
    """Check SWAP gate is working properly when called multiple times."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.X(2))
    c.add(gates.SWAP(0, 1))
    c.add(gates.SWAP(2, 3))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_by_swap_small(backend):
    """Check controlled SWAP using controlled by for ``nqubits=3``."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(3)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0))
    final_state = c.execute().numpy()
    c = Circuit(3)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0))
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    c = Circuit(3)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_by_swap(backend):
    """Check controlled SWAP using controlled by for ``nqubits=4``."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4)
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    c.add(gates.SWAP(2, 3).controlled_by(0))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    c.add(gates.SWAP(2, 3).controlled_by(0))
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    c.add(gates.SWAP(2, 3))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_doubly_controlled_by_swap(backend, accelerators):
    """Check controlled SWAP using controlled by two qubits."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0, 3))
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(3))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0, 3))
    c.add(gates.X(0))
    c.add(gates.X(3))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_toffoli_no_effect(backend):
    """Check Toffoli gate is working properly on |010>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(3)
    c.add(gates.X(1))
    c.add(gates.TOFFOLI(0, 1, 2))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_toffoli(backend):
    """Check Toffoli gate is working properly on |110>."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.TOFFOLI(0, 1, 2))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[-1] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_common_gates(backend):
    """Check that `Unitary` gate can create common gates."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.H(1))
    target_state = c.execute().numpy()
    c = Circuit(2)
    c.add(gates.Unitary(np.array([[0, 1], [1, 0]]), 0))
    c.add(gates.Unitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), 1))
    final_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    thetax = 0.1234
    thetay = 0.4321
    c = Circuit(2)
    c.add(gates.RX(0, theta=thetax))
    c.add(gates.RY(1, theta=thetay))
    c.add(gates.CNOT(0, 1))
    target_state = c.execute().numpy()
    c = Circuit(2)
    rx = np.array([[np.cos(thetax / 2), -1j * np.sin(thetax / 2)],
                   [-1j * np.sin(thetax / 2), np.cos(thetax / 2)]])
    ry = np.array([[np.cos(thetay / 2), -np.sin(thetay / 2)],
                   [np.sin(thetay / 2), np.cos(thetay / 2)]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    c.add(gates.Unitary(rx, 0))
    c.add(gates.Unitary(ry, 1))
    c.add(gates.Unitary(cnot, 0, 1))
    final_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_random_onequbit_gate(backend):
    """Check that ``Unitary`` gate can apply random 2x2 matrices."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    init_state = np.ones(4) / 2.0
    matrix = np.random.random([2, 2])
    target_state = np.kron(np.eye(2), matrix).dot(init_state)

    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.Unitary(matrix, 1, name="random"))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_random_twoqubit_gate(backend):
    """Check that ``Unitary`` gate can apply random 4x4 matrices."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    init_state = np.ones(8) / np.sqrt(8)
    matrix = np.random.random([4, 4])
    target_state = np.kron(np.eye(2), matrix).dot(init_state)

    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.H(2))
    c.add(gates.Unitary(matrix, 1, 2, name="random"))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_unitary_controlled_by(backend, accelerators):
    """Check that `controlled_by` works as expected with `Unitary`."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = np.random.random([2, 2])
    c = Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.Unitary(matrix, 1).controlled_by(0))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2.0
    target_state[2:] = matrix.dot(target_state[2:])
    np.testing.assert_allclose(final_state, target_state)

    matrix = np.random.random([4, 4])
    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    c.add(gates.Unitary(matrix, 1, 3).controlled_by(0, 2))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 4.0
    ids = [10, 11, 14, 15]
    target_state[ids] = matrix.dot(target_state[ids])
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_bad_shape(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = np.random.random((8, 8))
    with pytest.raises(ValueError):
        gate = gates.Unitary(matrix, 0, 1)

    if backend == "custom":
        with pytest.raises(NotImplementedError):
            gate = gates.Unitary(matrix, 0, 1, 2)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_various_type_initialization(backend):
    import tensorflow as tf
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    matrix = utils.random_tensorflow_complex((4, 4), dtype=tf.float64)
    gate = gates.Unitary(matrix, 0, 1)
    with pytest.raises(TypeError):
        gate = gates.Unitary("abc", 0, 1)
    qibo.set_backend(original_backend)


def test_control_unitary_error():
    matrix = np.random.random((4, 4))
    gate = gates.Unitary(matrix, 0, 1)
    with pytest.raises(ValueError):
        unitary = gate.control_unitary(np.random.random((16, 16)))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_construct_unitary(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    target_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    np.testing.assert_allclose(gates.H(0).unitary, target_matrix)
    target_matrix = np.array([[0, 1], [1, 0]])
    np.testing.assert_allclose(gates.X(0).unitary, target_matrix)
    target_matrix = np.array([[0, -1j], [1j, 0]])
    np.testing.assert_allclose(gates.Y(0).unitary, target_matrix)
    target_matrix = np.array([[1, 0], [0, -1]])
    np.testing.assert_allclose(gates.Z(1).unitary, target_matrix)

    target_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1], [0, 0, 1, 0]])
    np.testing.assert_allclose(gates.CNOT(0, 1).unitary, target_matrix)
    target_matrix = np.diag([1, 1, 1, -1])
    np.testing.assert_allclose(gates.CZ(1, 3).unitary, target_matrix)
    target_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                              [0, 1, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(gates.SWAP(2, 4).unitary, target_matrix)
    target_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 1, 0]])
    np.testing.assert_allclose(gates.TOFFOLI(1, 2, 3).unitary, target_matrix)

    theta = 0.1234
    target_matrix = np.array([[np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
                              [-1j * np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    np.testing.assert_allclose(gates.RX(0, theta).unitary, target_matrix)
    target_matrix = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                              [np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    np.testing.assert_allclose(gates.RY(0, theta).unitary, target_matrix)
    target_matrix = np.diag([np.exp(-1j * theta / 2.0), np.exp(1j * theta / 2.0)])
    np.testing.assert_allclose(gates.RZ(0, theta).unitary, target_matrix)
    target_matrix = np.diag([1, np.exp(1j * theta)])
    np.testing.assert_allclose(gates.U1(0, theta).unitary, target_matrix)
    target_matrix = np.diag([1, 1, 1, np.exp(1j * theta)])
    np.testing.assert_allclose(gates.CU1(0, 1, theta).unitary, target_matrix)
    from qibo import matrices
    target_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1]])
    np.testing.assert_allclose(matrices.SWAP, target_matrix)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_construct_unitary_controlled_by(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 0.1234
    rotation = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                         [np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    target_matrix = np.eye(4, dtype=rotation.dtype)
    target_matrix[2:, 2:] = rotation
    gate = gates.RY(0, theta).controlled_by(1)
    np.testing.assert_allclose(gate.unitary.numpy(), target_matrix)

    gate = gates.RY(0, theta).controlled_by(1, 2)
    with pytest.raises(NotImplementedError):
        unitary = gate.unitary
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_construct_unitary_errors(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    gate = gates.M(0)
    with pytest.raises(ValueError):
        matrix = gate.unitary

    pairs = list((i, i + 1) for i in range(0, 5, 2))
    theta = 2 * np.pi * np.random.random(6)
    gate = gates.VariationalLayer(range(6), pairs, gates.RY, gates.CZ, theta)
    with pytest.raises(ValueError):
        matrix = gate.unitary
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_by_unitary_action(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    init_state = utils.random_numpy_state(2)
    matrix = utils.random_numpy_complex((2, 2))
    gate = gates.Unitary(matrix, 1).controlled_by(0)
    c = Circuit(2)
    c.add(gate)
    target_state = c(np.copy(init_state)).numpy()
    final_state = gate.unitary.numpy().dot(init_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("name,params",
                         [("CRX", {"theta": 0.1}),
                          ("CRY", {"theta": 0.2}),
                          ("CRZ", {"theta": 0.3}),
                          ("CU1", {"theta": 0.1}),
                          ("CU2", {"phi": 0.1, "lam": 0.2}),
                          ("CU3", {"theta": 0.1, "phi": 0.2, "lam": 0.3})])
@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_rotations_from_un(backend, name, params):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    init_state = utils.random_numpy_state(2)
    gate = getattr(gates, name)(0, 1, **params)
    c = Circuit(2)
    c.add(gate)
    target_state = c(np.copy(init_state)).numpy()
    final_state = gate.unitary.numpy().dot(init_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [5, 6])
def test_variational_layer_call(nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend("custom")
    theta = 2 * np.pi * np.random.random(nqubits)
    c = Circuit(nqubits)
    c.add((gates.RY(i, t) for i, t in enumerate(theta)))
    c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    target_state = c().numpy()

    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    gate = gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ,
                                  theta)
    final_state = gate(c._default_initial_state()).numpy()
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
@pytest.mark.parametrize("nqubits", [4, 5, 6, 7, 10])
def test_variational_one_layer(backend, accelerators, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 2 * np.pi * np.random.random(nqubits)
    c = Circuit(nqubits)
    c.add((gates.RY(i, t) for i, t in enumerate(theta)))
    c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    target_state = c().numpy()

    c = Circuit(nqubits, accelerators)
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                 gates.RY, gates.CZ,
                                 theta))
    final_state = c().numpy()
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
@pytest.mark.parametrize("nqubits", [4, 5, 6, 7, 10])
def test_variational_two_layers(backend, accelerators, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    theta = 2 * np.pi * np.random.random(2 * nqubits)
    theta_iter = iter(theta)
    c = Circuit(nqubits)
    c.add((gates.RY(i, next(theta_iter)) for i in range(nqubits)))
    c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    c.add((gates.RY(i, next(theta_iter)) for i in range(nqubits)))
    c.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
    c.add(gates.CZ(0, nqubits - 1))
    target_state = c().numpy()

    c = Circuit(nqubits, accelerators)
    theta = theta.reshape((2, nqubits))
    pairs1 = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    pairs2 = list((i, i + 1) for i in range(1, nqubits - 2, 2))
    pairs2.append((0, nqubits - 1))
    c.add(gates.VariationalLayer(range(nqubits), pairs1,
                                 gates.RY, gates.CZ, theta[0]))
    c.add(gates.VariationalLayer(range(nqubits), pairs2,
                                 gates.RY, gates.CZ, theta[1]))
    final_state = c().numpy()
    np.testing.assert_allclose(target_state, final_state)

    c = Circuit(nqubits, accelerators)
    theta = theta.reshape((2, nqubits))
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                 gates.RY, gates.CZ,
                                 theta[0], theta[1]))
    c.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
    c.add(gates.CZ(0, nqubits - 1))
    final_state = c().numpy()
    np.testing.assert_allclose(target_state, final_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_variational_layer_errors(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = Circuit(6)
    pairs = list((i, i + 1) for i in range(0, 5, 2))
    with pytest.raises(ValueError):
        c.add(gates.VariationalLayer(range(6), pairs,
                                     gates.RY, gates.CZ,
                                     np.zeros(6), np.zeros(7)))
    with pytest.raises(ValueError):
        c.add(gates.VariationalLayer(range(7), pairs,
                                     gates.RY, gates.CZ,
                                     np.zeros(7), np.zeros(7)))
    with pytest.raises(ValueError):
        c.add(gates.VariationalLayer(range(10), pairs,
                                     gates.RY, gates.CZ,
                                     np.zeros(10), np.zeros(10)))

    gate = gates.VariationalLayer(range(6), pairs, gates.RY, gates.CZ,
                                  np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(gate.parameters, np.zeros(12))
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("gate,params",
                         [("H", {}), ("X", {}), ("Z", {}),
                          ("RX", {"theta": 0.1}),
                          ("RY", {"theta": 0.2}),
                          ("RZ", {"theta": 0.3}),
                          ("U1", {"theta": 0.1}),
                          ("U2", {"phi": 0.2, "lam": 0.3}),
                          ("U3", {"theta": 0.1, "phi": 0.2, "lam": 0.3})])
def test_dagger_one_qubit(backend, gate, params):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    c = Circuit(1)
    gate = getattr(gates, gate)(0, **params)
    c.add((gate, gate.dagger()))

    initial_state = utils.random_numpy_state(1)
    final_state = c(np.copy(initial_state)).numpy()
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("gate,params",
                         [("CNOT", {}),
                          ("CRX", {"theta": 0.1}),
                          ("CRZ", {"theta": 0.3}),
                          ("CU1", {"theta": 0.1}),
                          ("CU2", {"phi": 0.2, "lam": 0.3}),
                          ("CU3", {"theta": 0.1, "phi": 0.2, "lam": 0.3}),
                          ("fSim", {"theta": 0.1, "phi": 0.2})])
def test_dagger_two_qubit(backend, gate, params):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    c = Circuit(2)
    gate = getattr(gates, gate)(0, 1, **params)
    c.add((gate, gate.dagger()))

    initial_state = utils.random_numpy_state(2)
    final_state = c(np.copy(initial_state)).numpy()
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("gate,params",
                         [("H", {}), ("X", {}),
                          ("RX", {"theta": 0.1}),
                          ("RZ", {"theta": 0.2}),
                          ("U2", {"phi": 0.2, "lam": 0.3}),
                          ("U3", {"theta": 0.1, "phi": 0.2, "lam": 0.3})])
def test_dagger_controlled_by(backend, gate, params):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    c = Circuit(4)
    gate = getattr(gates, gate)(3, **params).controlled_by(0, 1, 2)
    c.add((gate, gate.dagger()))

    initial_state = utils.random_numpy_state(4)
    final_state = c(np.copy(initial_state)).numpy()
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("nqubits", [1, 2])
@pytest.mark.parametrize("tfmatrix", [False, True])
def test_unitary_dagger(backend, nqubits, tfmatrix):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    matrix = np.random.random((2 ** nqubits, 2 ** nqubits))
    if tfmatrix:
        import tensorflow as tf
        from qibo.config import DTYPES
        matrix = tf.cast(matrix, dtype=DTYPES.get('DTYPECPX'))
    gate = gates.Unitary(matrix, *range(nqubits))
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))

    initial_state = utils.random_numpy_state(nqubits)
    final_state = c(np.copy(initial_state)).numpy()
    if tfmatrix:
        matrix = matrix.numpy()
    target_state = matrix.dot(initial_state)
    target_state = matrix.conj().T.dot(target_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_controlled_by_dagger(backend):
    from scipy.linalg import expm
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    gate = gates.Unitary(matrix, 0).controlled_by(1, 2, 3, 4)
    c = Circuit(5)
    c.add((gate, gate.dagger()))

    initial_state = utils.random_numpy_state(5)
    final_state = c(np.copy(initial_state)).numpy()
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("tfmatrix", [False, True])
def test_generalizedfsim_dagger(backend, tfmatrix):
    from scipy.linalg import expm
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    phi = 0.2
    matrix = np.random.random((2, 2))
    matrix = expm(1j * (matrix + matrix.T))
    if tfmatrix:
        import tensorflow as tf
        from qibo.config import DTYPES
        matrix = tf.cast(matrix, dtype=DTYPES.get('DTYPECPX'))
    gate = gates.GeneralizedfSim(0, 1, matrix, phi)
    c = Circuit(2)
    c.add((gate, gate.dagger()))

    initial_state = utils.random_numpy_state(2)
    final_state = c(np.copy(initial_state)).numpy()
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
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

    initial_state = utils.random_numpy_state(nqubits)
    final_state = c(np.copy(initial_state)).numpy()
    np.testing.assert_allclose(final_state, initial_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("nqubits,targets,results,oncircuit",
                         [(2, [1], [0], False),
                          (3, [1], 0, True),
                          (4, [1, 3], [0, 1], True),
                          (5, [0, 3, 4], [1, 1, 0], False),
                          (6, [1, 3], np.ones(2, dtype=np.int), True),
                          (4, [0, 2], np.zeros(2, dtype=np.int32)[0], True)])
def test_collapse_gate(backend, nqubits, targets, results, oncircuit):
    from qibo.config import NUMERIC_TYPES
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    initial_state = utils.random_numpy_state(nqubits)
    if oncircuit:
        c = Circuit(nqubits)
        c.add(gates.Collapse(*targets, result=results))
        final_state = c(np.copy(initial_state)).numpy()
    else:
        collapse = gates.Collapse(*targets, result=results)
        if backend == "custom":
            final_state = collapse(np.copy(initial_state)).numpy()
        else:
            original_shape = initial_state.shape
            new_shape = nqubits * (2,)
            final_state = collapse(np.copy(initial_state).reshape(new_shape))
            final_state = final_state.numpy().reshape(original_shape)

    if isinstance(results, int) or isinstance(results, NUMERIC_TYPES):
        results = nqubits * [results]
    slicer = nqubits * [slice(None)]
    for t, r in zip(targets, results):
        slicer[t] = r
    slicer = tuple(slicer)
    initial_state = initial_state.reshape(nqubits * (2,))
    target_state = np.zeros_like(initial_state)
    target_state[slicer] = initial_state[slicer]
    norm = (np.abs(target_state) ** 2).sum()
    target_state = target_state.ravel() / np.sqrt(norm)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("accelerators",
                         [None, {"/GPU:0": 1, "/GPU:1": 1},
                          {"GPU:0": 2, "/GPU:1": 1, "/GPU:2": 1}])
@pytest.mark.parametrize("nqubits,targets", [(5, [2, 4]), (6, [3, 5])])
def test_collapse_gate_distributed(accelerators, nqubits, targets):
    initial_state = utils.random_numpy_state(nqubits)
    c = Circuit(nqubits, accelerators)
    c.add(gates.Collapse(*targets))
    final_state = c(np.copy(initial_state)).numpy()

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


@pytest.mark.parametrize("backend", _BACKENDS)
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
    for i, r in zip(qubits, bitstring.numpy()):
        if r:
            ct.add(gates.X(i))
    ct.add((gates.H(i) for i in qubits))
    target_state = ct()
    np.testing.assert_allclose(final_state, target_state, atol=1e-15)
    qibo.set_backend(original_backend)


def test_collapse_gate_errors():
    # pass wrong result length
    with pytest.raises(ValueError):
        gate = gates.Collapse(0, 1, result=[0, 1, 0])
    # pass wrong result values
    with pytest.raises(ValueError):
        gate = gates.Collapse(0, 1, result=[0, 2])
    # change result after creation
    gate = gates.Collapse(2, 0, result=[0, 0])
    gate.nqubits = 4
    gate.result = np.ones(2, dtype=np.int)


@pytest.mark.parametrize("backend", _BACKENDS)
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
    final_state = c(nshots=40).numpy()

    np.random.seed(123)
    target_state = []
    for _ in range(40):
        noiseless_c = Circuit(4)
        noiseless_c.add((gates.RY(i, t) for i, t in enumerate(thetas)))
        for i, ps in enumerate(probs):
            for p, gate in zip(ps, gatelist):
                if np.random.random() < p:
                    noiseless_c.add(gate(i))
        target_state.append(noiseless_c().numpy())
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_reset_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    initial_state = utils.random_numpy_state(5)
    c = Circuit(5)
    c.add(gates.ResetChannel(2, p0=0.3, p1=0.3, seed=123))
    final_state = c(np.copy(initial_state), nshots=30).numpy()

    np.random.seed(123)
    target_state = []
    for _ in range(30):
        noiseless_c = Circuit(5)
        if np.random.random() < 0.3:
            noiseless_c.add(gates.Collapse(2))
        if np.random.random() < 0.3:
            noiseless_c.add(gates.Collapse(2))
            noiseless_c.add(gates.X(2))
        target_state.append(noiseless_c(np.copy(initial_state)).numpy())
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_thermal_relaxation_channel_repeated(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    initial_state = utils.random_numpy_state(5)
    c = Circuit(5)
    c.add(gates.ThermalRelaxationChannel(4, t1=1.0, t2=0.6, time=0.8,
                                         excited_population=0.8, seed=123))
    final_state = c(np.copy(initial_state), nshots=30).numpy()

    pz, p0, p1 = gates.ThermalRelaxationChannel._calculate_probs(
        1.0, 0.6, 0.8, 0.8)
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
        target_state.append(noiseless_c(np.copy(initial_state)).numpy())
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)
