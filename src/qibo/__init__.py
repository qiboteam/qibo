__version__ = "0.1.5"
from qibo.config import set_threads, get_threads, set_batch_size, get_batch_size
from qibo.backends import set_precision, set_backend, set_device
from qibo.backends import get_backend, get_precision, get_device
from qibo.backends import numpy_matrices as matrices
from qibo.backends import K
from qibo import callbacks, gates, hamiltonians, models
from qibo import parallel, optimizers, solvers
