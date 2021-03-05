"""
Define the default circuit, constants and types.
"""
import os
import logging
import warnings

# Logging level from 0 (all) to 3 (errors)
LOG_LEVEL = 3

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

# Batch size for sampling shots in measurement frequencies calculation
SHOT_BATCH_SIZE = 2 ** 18

# Threshold size for sampling shots in measurements frequencies with custom operator
SHOT_CUSTOM_OP_THREASHOLD = 100000

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

def get_batch_size():
    """Returns batch size used for sampling measurement shots."""
    return SHOT_BATCH_SIZE

def set_batch_size(batch_size):
    if not isinstance(batch_size, int):
        raise_error(TypeError, "Shot batch size must be integer.")
    elif batch_size < 1:
        raise_error(ValueError, "Shot batch size must be a positive integer.")
    elif batch_size > 2 ** 31:
         raise_error(ValueError, "Shot batch size cannot be greater than 2^31.")
    global SHOT_BATCH_SIZE
    SHOT_BATCH_SIZE = batch_size


# Configuration for logging mechanism
class CustomHandler(logging.StreamHandler):
    """Custom handler for logging algorithm."""
    def format(self, record):
        """Format the record with specific format."""
        fmt = '[Qibo|%(levelname)s|%(asctime)s]: %(message)s'
        return logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S').format(record)


# allocate logger object
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(CustomHandler())
