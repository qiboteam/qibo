"""
Define the default circuit, constants and types.
"""
import os
import blessings
import logging
import warnings

# Logging level from 0 (all) to 3 (errors)
LOG_LEVEL = 3

# Select the default backend engine
BACKEND_NAME = "tensorflow"

# Choose the least significant qubit
LEAST_SIGNIFICANT_QUBIT = 0

if LEAST_SIGNIFICANT_QUBIT != 0: # pragma: no cover
    # case not tested because least significant qubit is preset to 0
    raise_error(NotImplementedError, "The least significant qubit should be 0.")

# characters used in einsum strings
EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Entanglement entropy eigenvalue cut-off
# Eigenvalues smaller than this cut-off are ignored in entropy calculation
EIGVAL_CUTOFF = 1e-14

# Flag for raising warning in ``set_precision`` and ``set_backend``
ALLOW_SWITCHERS = True


def raise_error(exception, message=None, args=None):
    """Raise exception with logging error.

    Args:
        exception (Exception): python exception.
        message (str): the error message.
    """
    log.error(message)
    if args:
        raise exception(message, args)
    else:
        raise exception(message)


# Set the number of threads from the environment variable
OMP_NUM_THREADS = None
if "OMP_NUM_THREADS" not in os.environ:
    import psutil
    # using physical cores by default
    cores = psutil.cpu_count(logical=False)
    OMP_NUM_THREADS = cores
else: # pragma: no cover
    OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS"))

def get_threads():
    """Returns number of threads."""
    return OMP_NUM_THREADS

def set_threads(num_threads):
    """Set number of OpenMP threads.

    Args:
        num_threads (int): number of threads.
    """
    if not isinstance(num_threads, int): # pragma: no cover
        raise_error(RuntimeError, "Number of threads must be integer.")
    if num_threads < 1: # pragma: no cover
        raise_error(RuntimeError, "Number of threads must be positive.")
    global OMP_NUM_THREADS
    OMP_NUM_THREADS = num_threads


# Backend access
from qibo import backend as K
def set_computation_backend(backend="tensorflow"):
    # TODO: Rename this to ``set_backend`` and change the
    # defaulteinsum/matmuleinsum setter name
    bk = K.factory.get(backend)()
    for method in K.function_names:
        setattr(K, method, getattr(bk, method))

set_computation_backend()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(LOG_LEVEL)
import numpy as np
import tensorflow as tf

ARRAY_TYPES = (np.ndarray, tf.Tensor)
NUMERIC_TYPES = (np.int, np.float, np.complex,
                 np.int32, np.int64, np.float32,
                 np.float64, np.complex64, np.complex128)


# Define numpy and tensorflow matrices
# numpy matrices are exposed to user via ``from qibo import matrices``
# tensorflow matrices are used by native gates (``/tensorflow/gates.py``)
from qibo.backend import matrices as _matrices
matrices = _matrices.NumpyMatrices(np.complex128)
tfmatrices = _matrices.TensorflowMatrices(tf.complex128)


def set_precision(dtype='double'):
    """Set precision for states and gates simulation.

    Args:
        dtype (str): possible options are 'single' for single precision
            (complex64) and 'double' for double precision (complex128).
    """
    K.set_precision(dtype)
    matrices.allocate_matrices()
    tfmatrices.allocate_matrices()

def get_precision():
    """Get precision for states and gates simulation.

    Returns:
        A string with the precision name ('single', 'double').
    """
    return K.dtypes('STRING')


def set_device(name):
    K.set_device(name)
    with tf.device(name):
        tfmatrices.allocate_matrices()

def get_device():
    return K.active_device


# Load backend specifics
if BACKEND_NAME == "tensorflow":
    # Gate backends
    BACKEND = {'GATES': 'custom', 'EINSUM': None, 'STRING': 'custom'}

    def set_backend(backend='custom'):
        """Sets backend used to implement gates.

        Args:
            backend (str): possible options are 'custom' for the gates that use
                custom tensorflow operator and 'defaulteinsum' or 'matmuleinsum'
                for the gates that use tensorflow primitives (``tf.einsum`` or
                ``tf.matmul`` respectively).
        """
        if not ALLOW_SWITCHERS and backend != BACKEND['GATES']:
            warnings.warn("Backend should not be changed after allocating gates.",
                          category=RuntimeWarning)
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
            raise_error(RuntimeError, f"Gate backend '{backend}' not supported.")
        BACKEND['STRING'] = backend

    def get_backend():
        """Get backend used to implement gates.

        Returns:
            A string with the backend name.
        """
        return BACKEND['STRING']

else: # pragma: no cover
    # case not tested because the backend is preset to TensorFlow
    raise_error(NotImplementedError, "Only Tensorflow backend is implemented.")


# Configuration for logging mechanism
t = blessings.Terminal()


class CustomColorHandler(logging.StreamHandler):
    """Custom color handler for logging algorithm."""

    colors = {
        logging.DEBUG: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold},
        logging.INFO: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_green},
        logging.WARNING: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_yellow},
        logging.ERROR: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_red, '%(message)s': t.bold},
        logging.CRITICAL: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_white_on_red, '%(message)s': t.bold},
    }

    def format(self, record):
        """Format the record with specific color."""
        levelcolors = self.colors[record.levelno]
        fmt = '[Qibo|%(levelname)s|%(asctime)s]: %(message)s'
        for s, subs in levelcolors.items():
            fmt = fmt.replace(s, subs(s))
        return logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S').format(record)


# allocate logger object
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(CustomColorHandler())
