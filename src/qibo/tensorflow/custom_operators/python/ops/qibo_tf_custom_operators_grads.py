import tensorflow as tf
from tensorflow.python.framework import ops # pylint: disable=no-name-in-module


@ops.RegisterGradient("InitialState")
def _initial_state_grad(op, grad): # pragma: no cover
    """The gradients for `initial_state`.

    Args:
        op: The `initial_state` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `initial_state` op.

    Returns:
        Gradients with respect to the input of `initial_state`.
    """
    # Not tested currently due to ``tf.tensor_scatter_nd_update`` bug on GPU
    to_initial_state = tf.concat([[0], grad[1:]], axis=0)
    return [to_initial_state]
