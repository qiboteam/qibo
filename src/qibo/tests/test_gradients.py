import pytest
import numpy as np
import tensorflow as tf
from qibo.tensorflow import custom_operators


@pytest.mark.skip("tf.tensor_scatter_nd_update bug on GPU (tensorflow#42581)")
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("compile", [False, True])
def test_initial_state_gradient(dtype, compile): # pragma: no cover
    """Check that initial_state works."""
    # Test skipped due to ``tf.tensor_scatter_nd_update`` bug on GPU
    def grad_default(var):
        update = tf.constant([1], dtype=dtype)
        with tf.GradientTape() as tape:
            loss = tf.tensor_scatter_nd_update(var, [[0]], update)
        return tape.gradient(loss, var)

    def grad_custom(var):
        with tf.GradientTape() as tape:
            loss = custom_operators.initial_state(var)
        return tape.gradient(loss, var)

    if compile:
        grad_default = tf.function(grad_default)
        grad_custom = tf.function(grad_custom)

    zeros = tf.Variable(tf.zeros(10, dtype=dtype))
    grad_reference = grad_default(zeros)
    grad_custom_op = grad_custom(zeros)
    np.testing.assert_allclose(grad_reference, grad_custom_op)
