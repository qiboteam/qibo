# Logging level
LOG_LEVEL = 3

# Default values
BACKEND_NAME = "tensorflow"
LEAST_SIGNIFICANT_QUBIT = 0


if LEAST_SIGNIFICANT_QUBIT != 0:
    raise NotImplementedError("The least significant qubit should be 0.")


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