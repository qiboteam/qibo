__version__ = "0.1.3-dev"
from qibo.config import set_threads, get_threads
from qibo.backends import set_precision, set_backend, set_device
from qibo.backends import get_backend, get_precision, get_device
from qibo.backends import numpy_matrices as matrices
from qibo.backends import K
from qibo.numpy import tomography
from qibo import callbacks, evolution, gates, hamiltonians, models
from qibo import parallel, optimizers, solvers
