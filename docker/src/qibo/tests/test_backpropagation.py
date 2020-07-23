import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit

_BACKENDS = ["defaulteinsum", "matmuleinsum"]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_variable_backpropagation(backend):
    """Check that backpropagation works when using `tf.Variable` parameters."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    import tensorflow as tf
    from qibo.config import DTYPES
    theta = tf.Variable(0.1234, dtype=DTYPES.get('DTYPE'))

    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with tf.GradientTape() as tape:
        c = Circuit(1)
        c.add(gates.X(0))
        c.add(gates.RZ(0, theta))
        loss = tf.math.real(c()[-1])
    grad = tape.gradient(loss, theta)

    target_loss = np.cos(theta.numpy() / 2.0)
    np.testing.assert_allclose(loss.numpy(), target_loss)

    target_grad = - np.sin(theta.numpy() / 2.0) / 2.0
    np.testing.assert_allclose(grad.numpy(), target_grad)
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_two_variables_backpropagation(backend):
    """Check that backpropagation works when using `tf.Variable` parameters."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    import tensorflow as tf
    from qibo.config import DTYPES
    theta = tf.Variable([0.1234, 0.4321], dtype=DTYPES.get('DTYPE'))

    # TODO: Fix parametrized gates so that `Circuit` can be defined outside
    # of the gradient tape
    with tf.GradientTape() as tape:
        c = Circuit(2)
        c.add(gates.RX(0, theta[0]))
        c.add(gates.RY(1, theta[1]))
        loss = tf.math.real(c()[0])
    grad = tape.gradient(loss, theta)

    t = np.array([0.1234, 0.4321]) / 2.0
    target_loss = np.cos(t[0]) * np.cos(t[1])
    np.testing.assert_allclose(loss.numpy(), target_loss)

    target_grad1 = - np.sin(t[0]) * np.cos(t[1])
    target_grad2 = - np.cos(t[0]) * np.sin(t[1])
    target_grad = np.array([target_grad1, target_grad2]) / 2.0
    np.testing.assert_allclose(grad.numpy(), target_grad)
    qibo.set_backend(original_backend)
