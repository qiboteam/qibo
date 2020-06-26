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

    # Set devices recognized by tensorflow
    DEVICES = {
        'CPU': tf.config.list_logical_devices("CPU"),
        'GPU': tf.config.list_physical_devices("GPU"),
        'MEASUREMENT_CUTOFF': 1300000000
    }
    #DEVICE['NAMES'] = set(d.name for d in DEVICES['CPU'] + DEVICES['GPU'])
    if DEVICES['GPU']: # set default device to GPU if it exists
        DEVICES['DEFAULT'] = DEVICES['GPU'][0].name
    elif DEVICES['CPU']:
        DEVICES['DEFAULT'] = DEVICES['CPU'][0].name
    else:
        raise RuntimeError("Unable to find Tensorflow devices.")

    # Define numpy and tensorflow matrices
    # numpy matrices are exposed to user via ``from qibo import matrices``
    # tensorflow matrices are used by native gates (``/tensorflow/gates.py``)
    from qibo.tensorflow import matrices as _matrices
    matrices = _matrices.NumpyMatrices()
    tfmatrices = _matrices.TensorflowMatrices()


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
            raise RuntimeError(f'dtype {dtype} not supported.')
        matrices.allocate_matrices()
        tfmatrices.allocate_matrices()


    def set_device(device_name: str):
        DEVICES['DEFAULT'] = device_name
        with tf.devices(device_name):
            tfmatrices.allocate_matrices()


else:
    raise NotImplementedError("Only Tensorflow backend is implemented.")
