"""
Define the default circuit, constants and types.
"""
# Logging level from 0 (all) to 3 (errors)
LOG_LEVEL = 3

# Select the default backend engine
BACKEND_NAME = "tensorflow"

# Choose the least significant qubit
LEAST_SIGNIFICANT_QUBIT = 0

if LEAST_SIGNIFICANT_QUBIT != 0:
    raise NotImplementedError("The least significant qubit should be 0.")

# Load backend specifics
if BACKEND_NAME == "tensorflow":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(LOG_LEVEL)
    import tensorflow as tf
    # Backend access
    K = tf

    # Einsum backend switcher according to device
    if tf.config.list_physical_devices("GPU"):
        # If GPU is available use `tf.einsum`
        from qibo.tensorflow.einsum import DefaultEinsum
        einsum = DefaultEinsum()
    else:
        # If only CPU is available then fall back to `tf.matmul`
        from qibo.tensorflow.einsum import MatmulEinsum
        einsum = MatmulEinsum()

    # Default types
    DTYPE = tf.float64
    DTYPEINT = tf.int32
    DTYPECPX = tf.complex128

    # Set memory cut-off for using GPU when sampling
    GPU_MEASUREMENT_CUTOFF = 1300000000

    # Find available CPUs as they may be needed for sampling
    _available_cpus = tf.config.list_logical_devices("CPU")
    if _available_cpus:
        CPU_NAME = _available_cpus[0].name
    else:
        CPU_NAME = None

    from qibo.tensorflow import matrices as tensorflow_matrices
    matrices = tensorflow_matrices.GateMatrices(DTYPECPX)

else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")
