__version__ = "0.1.2-dev-cqt"
from qibo.config import BACKEND_NAME, raise_error

if BACKEND_NAME == "cqt":
    from qibo.config import matrices, K
    from qibo.numpy import tomography

elif BACKEND_NAME == "tensorflow":
    from qibo.config import set_precision, set_backend, set_device, get_backend, get_precision, get_device, matrices, K
    from qibo import callbacks, evolution, models, gates, hamiltonians, optimizers, solvers
    from qibo.numpy import tomography

else:
    raise_error(NotImplementedError)
