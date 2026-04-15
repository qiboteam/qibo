"""
Define the default circuit, constants and types.
"""

import logging
import os

# Logging level from 0 (all) to 4 (errors)
# (see https://docs.python.org/3/library/logging.html#logging-levels)
QIBO_LOG_LEVEL = 1
if "QIBO_LOG_LEVEL" in os.environ:  # pragma: no cover
    QIBO_LOG_LEVEL = 10 * int(os.environ.get("QIBO_LOG_LEVEL"))

# Logging level from 0 (all) to 3 (errors) for TensorFlow
TF_LOG_LEVEL = 3
if "TF_LOG_LEVEL" in os.environ:  # pragma: no cover
    TF_LOG_LEVEL = int(os.environ.get("TF_LOG_LEVEL"))

# Maximum number of qubits allowed for state allocation.
# These limits prevent uncontrolled memory consumption (CWE-400).
# Memory scales as 2^n * 16 bytes (complex128) for state vectors
# and 4^n * 16 bytes for density matrices.
#
# Default is -1 (unlimited). To enable protection, set to a positive
# integer via environment variables or the corresponding setter
# functions (``qibo.set_max_qubits``, ``qibo.set_max_qubits_dm``).
MAX_QUBITS = -1
if "QIBO_MAX_QUBITS" in os.environ:  # pragma: no cover
    MAX_QUBITS = int(os.environ.get("QIBO_MAX_QUBITS"))

MAX_QUBITS_DM = -1
if "QIBO_MAX_QUBITS_DM" in os.environ:  # pragma: no cover
    MAX_QUBITS_DM = int(os.environ.get("QIBO_MAX_QUBITS_DM"))

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


def get_max_qubits():
    """Returns maximum number of qubits allowed for state vector allocation."""
    return MAX_QUBITS


def set_max_qubits(max_qubits):
    """Sets maximum number of qubits allowed for state vector allocation.

    This limit prevents uncontrolled memory consumption when allocating
    state vectors. Memory scales as ``2^n * 16`` bytes (complex128).

    Args:
        max_qubits (int): Maximum number of qubits. Must be a positive
            integer, or ``-1`` to disable the limit.
    """
    if not isinstance(max_qubits, int):
        raise_error(TypeError, "Maximum number of qubits must be an integer.")
    elif max_qubits < 1 and max_qubits != -1:
        raise_error(
            ValueError,
            "Maximum number of qubits must be a positive integer or -1 (unlimited).",
        )
    global MAX_QUBITS
    MAX_QUBITS = max_qubits


def get_max_qubits_dm():
    """Returns maximum number of qubits allowed for density matrix allocation."""
    return MAX_QUBITS_DM


def set_max_qubits_dm(max_qubits_dm):
    """Sets maximum number of qubits allowed for density matrix allocation.

    This limit prevents uncontrolled memory consumption when allocating
    density matrices. Memory scales as ``4^n * 16`` bytes (complex128).

    Args:
        max_qubits_dm (int): Maximum number of qubits. Must be a positive
            integer, or ``-1`` to disable the limit.
    """
    if not isinstance(max_qubits_dm, int):
        raise_error(TypeError, "Maximum number of qubits must be an integer.")
    elif max_qubits_dm < 1 and max_qubits_dm != -1:
        raise_error(
            ValueError,
            "Maximum number of qubits must be a positive integer or -1 (unlimited).",
        )
    global MAX_QUBITS_DM
    MAX_QUBITS_DM = max_qubits_dm


def raise_error(exception, message=None):
    """Raise exception with logging error.

    Args:
        exception (Exception): python exception.
        message (str): the error message.
    """
    log.error(message)
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
        from qibo import __version__  # pylint: disable=C0415

        fmt = f"[Qibo {__version__}|%(levelname)s|%(asctime)s]: %(message)s"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


# allocate logger object
log = logging.getLogger(__name__)
log.setLevel(QIBO_LOG_LEVEL)
log.addHandler(CustomHandler())
