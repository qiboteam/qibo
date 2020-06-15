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

    # Default types
    DTYPES = {
        'DTYPEINT': tf.int64,
        'DTYPE': tf.float64,
        'DTYPECPX': tf.complex128
    }

    # Gate backends
    BACKEND = {'GATES': 'custom', 'EINSUM': None}

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


    def set_backend(backend='custom'):
        """Sets backend used to implement gates.

        Args:
            backend (str): possible options are 'custom' for the gates that use
                custom tensorflow operator and 'defaulteinsum' or 'matmuleinsum'
                for the gates that use tensorflow primitives (``tf.einsum`` or
                ``tf.matmul`` respectively).
        """
        if backend == 'custom':
            BACKEND['GATES'] = 'custom'
            BACKEND['EINSUM'] = None
        elif backend == 'defaulteinsum':
            from qibo.tensorflow import einsum
            BACKEND['GATES'] = 'native'
            BACKEND['EINSUM'] = einsum.DefaultEinsum()
        elif backend == 'matmuleinsum':
            from qibo.tensorflow import einsum
            BACKEND['GATES'] = 'native'
            BACKEND['EINSUM'] = einsum.MatmulEinsum()
        else:
            raise RuntimeError(f"Gate backend '{backend}' not supported.")


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
            raise RuntimeError(f"dtype '{dtype}' not supported.")
        matrices.allocate_gates()

else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")
