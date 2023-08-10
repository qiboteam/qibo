import os

from qibo.backends.abstract import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.backends.tensorflow import TensorflowBackend
from qibo.config import log, raise_error


def construct_backend(backend, platform=None, runcard=None):
    if backend == "qibojit":
        from qibojit.backends import CupyBackend, CuQuantumBackend, NumbaBackend

        if platform == "cupy":  # pragma: no cover
            return CupyBackend()
        elif platform == "cuquantum":  # pragma: no cover
            return CuQuantumBackend()
        elif platform == "numba":
            return NumbaBackend()
        else:  # pragma: no cover
            try:
                return CupyBackend()
            except (ModuleNotFoundError, ImportError):
                return NumbaBackend()

    elif backend == "tensorflow":
        return TensorflowBackend()

    elif backend == "numpy":
        return NumpyBackend()

    elif backend == "qibolab":  # pragma: no cover
        from qibolab.backends import QibolabBackend  # pylint: disable=E0401

        return QibolabBackend(platform, runcard)

    else:  # pragma: no cover
        raise_error(ValueError, f"Backend {backend} is not available.")


class GlobalBackend(NumpyBackend):
    """The global backend will be used as default by ``circuit.execute()``."""

    _instance = None
    _dtypes = {"double": "complex128", "single": "complex64"}
    _default_order = [
        {"backend": "qibojit", "platform": "cupy"},
        {"backend": "qibojit", "platform": "numba"},
        {"backend": "tensorflow"},
        {"backend": "numpy"},
    ]

    def __new__(cls):
        if cls._instance is not None:
            return cls._instance

        backend = os.environ.get("QIBO_BACKEND")
        if backend:  # pragma: no cover
            # Create backend specified by user
            platform = os.environ.get("QIBO_PLATFORM")
            cls._instance = construct_backend(backend, platform)
        else:
            # Create backend according to default order
            for kwargs in cls._default_order:
                try:
                    cls._instance = construct_backend(**kwargs)
                    break
                except (ModuleNotFoundError, ImportError):
                    pass

        if cls._instance is None:  # pragma: no cover
            raise_error(RuntimeError, "No backends available.")

        log.info(f"Using {cls._instance} backend on {cls._instance.device}")
        return cls._instance

    @classmethod
    def set_backend(cls, backend, platform=None, runcard=None):  # pragma: no cover
        if (
            cls._instance is None
            or cls._instance.name != backend
            or cls._instance.platform != platform
        ):
            cls._instance = construct_backend(backend, platform, runcard)
        log.info(f"Using {cls._instance} backend on {cls._instance.device}")


class QiboMatrices:
    def __init__(self, dtype="complex128"):
        self.create(dtype)

    def create(self, dtype):
        self.matrices = NumpyMatrices(dtype)
        self.I = self.matrices.I(2)
        self.X = self.matrices.X
        self.Y = self.matrices.Y
        self.Z = self.matrices.Z
        self.SX = self.matrices.SX
        self.H = self.matrices.H
        self.S = self.matrices.S
        self.SDG = self.matrices.SDG
        self.CNOT = self.matrices.CNOT
        self.CZ = self.matrices.CZ


matrices = QiboMatrices()


def get_backend():
    return str(GlobalBackend())


def set_backend(backend, platform=None, runcard=None):
    GlobalBackend.set_backend(backend, platform, runcard)


def get_precision():
    return GlobalBackend().precision


def set_precision(precision):
    GlobalBackend().set_precision(precision)
    matrices.create(GlobalBackend().dtype)


def get_device():
    return GlobalBackend().device


def set_device(device):
    parts = device[1:].split(":")
    if device[0] != "/" or len(parts) < 2 or len(parts) > 3:
        raise_error(
            ValueError,
            "Device name should follow the pattern: " "/{device type}:{device number}.",
        )
    backend = GlobalBackend()
    backend.set_device(device)
    log.info(f"Using {backend} backend on {backend.device}")


def get_threads():
    return GlobalBackend().nthreads


def set_threads(nthreads):
    if not isinstance(nthreads, int):
        raise_error(TypeError, "Number of threads must be integer.")
    if nthreads < 1:
        raise_error(ValueError, "Number of threads must be positive.")
    GlobalBackend().set_threads(nthreads)
