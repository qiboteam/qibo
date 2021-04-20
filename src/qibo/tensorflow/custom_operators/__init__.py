"""TensorFlow custom operator for Tensor initial state."""
from qibo.config import log

_custom_operators_loaded = False
try:
    import tensorflow as tf
    try:
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import initial_state
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import transpose_state
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import swap_pieces
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_gate
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_x
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_y
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_z
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_z_pow
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_two_qubit_gate
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_fsim
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import apply_swap
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import collapse_state
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators import measure_frequencies
        # Import gradients
        from qibo.tensorflow.custom_operators.python.ops.qibo_tf_custom_operators_grads import _initial_state_grad
        _custom_operators_loaded = True
    except tf.errors.NotFoundError: # pragma: no cover
        log.warning("Custom operators not found, skipping custom operators load.")

except ModuleNotFoundError: # pragma: no cover
    # case not tested because CI has tf installed
    log.warning("Tensorflow is not installed. Skipping custom operators load.")
