"""
Define the backend and constants, header style.
"""
# Logging level from 0 (all) to 3 (errors)
LOG_LEVEL = 3

# Select the default backend engine
BACKEND_NAME = 'tensorflow'

# Choose the least significant qubit
LEAST_SIGNIFICANT_QUBIT = 0

if LEAST_SIGNIFICANT_QUBIT != 0:
    raise NotImplementedError("The least significant qubit should be 0.")

# Load backend specifics
if BACKEND_NAME == "tensorflow":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(LOG_LEVEL)
    import tensorflow as tf
    from qibo.backends import tensorflow

    # Default types
    DTYPE = tf.float64
    DTYPEINT = tf.int32
    DTYPECPX = tf.complex128

    def new_backend():
        return tensorflow.TensorflowBackend()
else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")