"""
Testing Tensorflow gates.
"""
import numpy as np
import qibo
from qibo import K, gates

try:
    import tensorflow as tf
    BACKENDS = ["custom", "defaulteinsum", "matmuleinsum",
                "numpy_defaulteinsum", "numpy_matmuleinsum"]
except ModuleNotFoundError: # pragma: no cover
    BACKENDS = ["numpy_defaulteinsum", "numpy_matmuleinsum"]


def apply_gates(gatelist, nqubits=None, initial_state=None):
    if initial_state is None:
        state = K.qnp.zeros(2 ** nqubits)
        state[0] = 1
        if qibo.get_backend() != "custom":
            state = K.qnp.reshape(state, nqubits * (2,))
    else: # pragma: no cover
        state = K.np.copy(initial_state)

    for gate in gatelist:
        state = gate(state)

    if qibo.get_backend() != "custom":
        state = np.array(state).ravel()
    return state


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


def test_rx_parameter_setter(backend):
    """Check that the parameter setter of RX gate is working properly."""
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
