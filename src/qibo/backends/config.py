# Logging level
LOG_LEVEL = 3

# Default backend
BACKEND_NAME = "tensorflow"

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
    raise NotImplementedError