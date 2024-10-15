import importlib.metadata as im

__version__ = im.version(__package__)

from qibo import (
    callbacks,
    gates,
    hamiltonians,
    models,
    optimizers,
    parallel,
    parameter,
    result,
    solvers,
)
from qibo.backends import (
    construct_backend,
    get_backend,
    get_device,
    get_precision,
    get_threads,
    get_transpiler,
    get_transpiler_name,
    list_available_backends,
    matrices,
    set_backend,
    set_device,
    set_precision,
    set_threads,
    set_transpiler,
)
from qibo.config import (
    get_batch_size,
    get_metropolis_threshold,
    set_batch_size,
    set_metropolis_threshold,
)
from qibo.models.circuit import Circuit
