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

    # characters used in einsum strings
    EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Entanglement entropy eigenvalue cut-off
    # Eigenvalues smaller than this cut-off are ignored in entropy calculation
    EIGVAL_CUTOFF = 1e-14

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
    DTYPES = {
        'DTYPEINT': tf.int64,
        'DTYPE': tf.float64,
        'DTYPECPX': tf.complex128
    }

    # Set memory cut-off for using GPU when sampling
    GPU_MEASUREMENT_CUTOFF = 1300000000

    # Find available CPUs as they may be needed for sampling
    _available_cpus = tf.config.list_logical_devices("CPU")
    if _available_cpus:
        CPU_NAME = _available_cpus[0].name
    else:
        CPU_NAME = None

    from qibo.tensorflow import matrices as tensorflow_matrices
    matrices = tensorflow_matrices.GateMatrices()

    def set_precision(dtype='double'):
        """Set precision for states and gates simulation.

        Args:
            dtype (str): possible options are 'single' for single precision
                (complex64) and 'double' for double precision (complex128).
        """
        if dtype == 'single':
            DTYPES['DTYPE'] = tf.float32
            DTYPES['DTYPECPX'] = tf.complex64
        elif dtype == 'double':
            DTYPES['DTYPE'] = tf.float64
            DTYPES['DTYPECPX'] = tf.complex128
        else:
            raise RuntimeError(f'dtype {dtype} not supported.')
        matrices.allocate_gates()

else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")
