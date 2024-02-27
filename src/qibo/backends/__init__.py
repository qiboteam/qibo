import os
from importlib import import_module

from qibo.backends.abstract import Backend
from qibo.backends.clifford import CliffordBackend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.backends.tensorflow import TensorflowBackend
from qibo.config import log, raise_error

QIBO_NATIVE_BACKENDS = ("numpy", "tensorflow")
QIBO_NON_NATIVE_BACKENDS = ("qibojit", "qibolab", "qibocloud")


class MetaBackend:
    """Meta-backend class which takes care of loading the qibo backends."""

    @staticmethod
    def load(backend: str, **kwargs) -> Backend:
        """Loads the backend.

        Args:
            backend (str): Name of the backend to load.
            kwargs (dict): Additional arguments for the non-native qibo backends.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if backend == "numpy":
            return NumpyBackend()
        elif backend == "tensorflow":
            return TensorflowBackend()
        elif backend == "clifford":
            return CliffordBackend(**kwargs)
        elif backend in QIBO_NON_NATIVE_BACKENDS:
            module = import_module(backend)
            return getattr(module, "MetaBackend").load(**kwargs)
        else:
            raise_error(
                ValueError,
                f"Backend {backend} is not available. To check which  backend is installed use `qibo.list_available_backends()`.",
            )

    def list_available(self) -> dict:
        """Lists all the available qibo backends."""
        available_backends = {}
        for backend in QIBO_NATIVE_BACKENDS:
            try:
                MetaBackend.load(backend)
                available = True
            except:
                available = False
            available_backends[backend] = available
        for backend in QIBO_NON_NATIVE_BACKENDS:
            try:
                module = import_module(backend)
                available = getattr(module, "MetaBackend")().list_available()
            except:
                available = False
            available_backends.update({backend: available})
        return available_backends


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
            cls._instance = construct_backend(backend, platform=platform)
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
    def set_backend(cls, backend, **kwargs):  # pragma: no cover
        if (
            cls._instance is None
            or cls._instance.name != backend
            or cls._instance.platform != kwargs.get("platform")
        ):
            cls._instance = construct_backend(backend, **kwargs)
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
        self.CY = self.matrices.CY
        self.CZ = self.matrices.CZ
        self.CSX = self.matrices.CSX
        self.CSXDG = self.matrices.CSXDG
        self.SWAP = self.matrices.SWAP
        self.iSWAP = self.matrices.iSWAP
        self.SiSWAP = self.matrices.SiSWAP
        self.SiSWAPDG = self.matrices.SiSWAPDG
        self.FSWAP = self.matrices.FSWAP
        self.ECR = self.matrices.ECR
        self.SYC = self.matrices.SYC
        self.TOFFOLI = self.matrices.TOFFOLI


matrices = QiboMatrices()


def get_backend():
    return str(GlobalBackend())


def set_backend(backend, **kwargs):
    GlobalBackend.set_backend(backend, **kwargs)


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
            "Device name should follow the pattern: /{device type}:{device number}.",
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


def _check_backend(backend):
    if backend is None:
        return GlobalBackend()

    return backend
