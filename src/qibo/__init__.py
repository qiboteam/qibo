__version__ = "0.1.10"
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
