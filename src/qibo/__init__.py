import importlib.metadata as im

__version__ = im.version(__package__)

from qibo import callbacks, gates, hamiltonians, models, optimizers, parallel, solvers
from qibo.backends import (
    get_backend,
    get_device,
    get_precision,
    get_threads,
    matrices,
    set_backend,
    set_device,
    set_precision,
    set_threads,
)
from qibo.config import (
    get_batch_size,
    get_metropolis_threshold,
    set_batch_size,
    set_metropolis_threshold,
)
from qibo.models.circuit import Circuit
