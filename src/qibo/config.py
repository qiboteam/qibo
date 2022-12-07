"""
Define the default circuit, constants and types.
"""
import logging
import os

# Logging level from 0 (all) to 4 (errors) (see https://docs.python.org/3/library/logging.html#logging-levels)
QIBO_LOG_LEVEL = 1
if "QIBO_LOG_LEVEL" in os.environ:  # pragma: no cover
    QIBO_LOG_LEVEL = 10 * int(os.environ.get("QIBO_LOG_LEVEL"))

# Logging level from 0 (all) to 3 (errors) for TensorFlow
TF_LOG_LEVEL = 3
if "TF_LOG_LEVEL" in os.environ:  # pragma: no cover
    TF_LOG_LEVEL = int(os.environ.get("TF_LOG_LEVEL"))

# characters used in einsum strings
EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Entanglement entropy eigenvalue cut-off
# Eigenvalues smaller than this cut-off are ignored in entropy calculation
EIGVAL_CUTOFF = 1e-14

# Tolerance for the probability sum check in the unitary channel
PRECISION_TOL = 1e-8

# Batch size for sampling shots in measurement frequencies calculation
SHOT_BATCH_SIZE = 2**18

# Threshold size for sampling shots in measurements frequencies with custom operator
SHOT_METROPOLIS_THRESHOLD = 100000

# Max iterations for normalizing bistochastic matrices
MAX_ITERATIONS = 50


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


def get_batch_size():
    """Returns batch size used for sampling measurement shots."""
    return SHOT_BATCH_SIZE


def set_batch_size(batch_size):
    """Sets batch size used for sampling measurement shots."""
    if not isinstance(batch_size, int):
        raise_error(TypeError, "Shot batch size must be integer.")
    elif batch_size < 1:
        raise_error(ValueError, "Shot batch size must be a positive integer.")
    elif batch_size > 2**31:
        raise_error(ValueError, "Shot batch size cannot be greater than 2^31.")
    global SHOT_BATCH_SIZE
    SHOT_BATCH_SIZE = batch_size


def get_metropolis_threshold():
    """Returns threshold for using Metropolis algorithm for sampling measurement shots."""
    return SHOT_METROPOLIS_THRESHOLD


def set_metropolis_threshold(threshold):
    """Sets threshold for using Metropolis algorithm for sampling measurement shots."""
    if not isinstance(threshold, int):
        raise_error(TypeError, "Shot threshold must be integer.")
    elif threshold < 1:
        raise_error(ValueError, "Shot threshold be a positive integer.")
    global SHOT_METROPOLIS_THRESHOLD
    SHOT_METROPOLIS_THRESHOLD = threshold


# Configuration for logging mechanism
class CustomHandler(logging.StreamHandler):
    """Custom handler for logging algorithm."""

    def format(self, record):
        """Format the record with specific format."""
        from qibo import __version__

        fmt = f"[Qibo {__version__}|%(levelname)s|%(asctime)s]: %(message)s"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


# allocate logger object
log = logging.getLogger(__name__)
log.setLevel(QIBO_LOG_LEVEL)
log.addHandler(CustomHandler())
