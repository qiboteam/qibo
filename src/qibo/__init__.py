__version__ = "0.1.8rc0"
from qibo.config import set_batch_size, get_batch_size
from qibo.config import set_metropolis_threshold, get_metropolis_threshold
from qibo.backends import matrices
from qibo.backends import set_backend, get_backend
from qibo.backends import set_precision, get_precision
from qibo.backends import set_device, get_device
from qibo.backends import set_threads, get_threads
from qibo.backends import set_threads, get_threads
from qibo import gates, models
from qibo import callbacks, hamiltonians
from qibo import parallel, optimizers, solvers
