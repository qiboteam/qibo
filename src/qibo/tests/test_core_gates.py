"""
Testing Tensorflow gates.
"""
import pytest
import numpy as np
import qibo
from qibo import K, gates

try:
    import tensorflow as tf
    BACKENDS = ["custom", "defaulteinsum", "matmuleinsum",
                "numpy_defaulteinsum", "numpy_matmuleinsum"]
except ModuleNotFoundError:
    BACKENDS = ["defaulteinsum", "matmuleinsum"]


def apply_gates(gatelist, nqubits=None, initial_state=None):
    if initial_state is None:
        state = K.qnp.zeros(2 ** nqubits)
        state[0] = 1
        if qibo.get_backend() != "custom":
            state = K.qnp.reshape(state, nqubits * (2,))
    for gate in gatelist:
        state = gate(state)
    if qibo.get_backend() != "custom":
        state = np.array(state).ravel()
    return state


@pytest.mark.parametrize("backend", BACKENDS)
def test_h(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.H(0), gates.H(1)], nqubits=2)
    target_state = np.ones_like(final_state) / 2
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_x(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.X(0)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_y(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.Y(1)], nqubits=2)
    target_state = np.zeros_like(final_state)
    target_state[1] = 1j
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_z(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    final_state = apply_gates([gates.H(0), gates.H(1), gates.Z(0)], nqubits=2)
    target_state = np.ones_like(final_state) / 2.0
    target_state[2] *= -1.0
    target_state[3] *= -1.0
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)
