import os
from importlib import import_module

import numpy as np

from qibo.backends.abstract import Backend
from qibo.backends.clifford import CliffordBackend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.backends.pytorch import PyTorchBackend
from qibo.backends.tensorflow import TensorflowBackend
from qibo.config import log, raise_error

QIBO_NATIVE_BACKENDS = ("numpy", "tensorflow", "pytorch")
QIBO_NON_NATIVE_BACKENDS = ("qibojit", "qibolab", "qibo-cloud-backends", "qibotn")


class MetaBackend:
    """Meta-backend class which takes care of loading the qibo backends."""

    @staticmethod
    def load(backend: str, **kwargs) -> Backend:
        """Loads the native qibo backend.

        Args:
            backend (str): Name of the backend to load.
            kwargs (dict): Additional arguments for the qibo backend.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if backend == "numpy":
            return NumpyBackend()
        elif backend == "tensorflow":
            return TensorflowBackend()
        elif backend == "pytorch":
            return PyTorchBackend()
        elif backend == "clifford":
            engine = kwargs.pop("platform", None)
            kwargs["engine"] = engine
            return CliffordBackend(**kwargs)
        else:
            raise_error(
                ValueError,
                f"Backend {backend} is not available. The native qibo backends are {QIBO_NATIVE_BACKENDS}.",
            )

    def list_available(self) -> dict:
        """Lists all the available native qibo backends."""
        available_backends = {}
        for backend in QIBO_NATIVE_BACKENDS:
            try:
                MetaBackend.load(backend)
                available = True
            except:  # pragma: no cover
                available = False
            available_backends[backend] = available
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
        {"backend": "pytorch"},
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
        self.CCZ = self.matrices.CCZ


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


def list_available_backends() -> dict:
    """Lists all the backends that are available."""
    available_backends = MetaBackend().list_available()
    for backend in QIBO_NON_NATIVE_BACKENDS:
        try:
            module = import_module(backend.replace("-", "_"))
            available = getattr(module, "MetaBackend")().list_available()
        except:
            available = False
        available_backends.update({backend: available})
    return available_backends


def construct_backend(backend, **kwargs) -> Backend:
    """Construct a generic native or non-native qibo backend.
    Args:
        backend (str): Name of the backend to load.
        kwargs (dict): Additional arguments for constructing the backend.
    Returns:
        qibo.backends.abstract.Backend: The loaded backend.

    """
    if backend in QIBO_NATIVE_BACKENDS + ("clifford",):
        return MetaBackend.load(backend, **kwargs)
    elif backend in QIBO_NON_NATIVE_BACKENDS:
        module = import_module(backend.replace("-", "_"))
        return getattr(module, "MetaBackend").load(**kwargs)
    else:
        raise_error(
            ValueError,
            f"Backend {backend} is not available. To check which backends are installed use `qibo.list_available_backends()`.",
        )


def _check_backend_and_local_state(seed, backend):
    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, "seed must be either type int or numpy.random.Generator."
        )

    backend = _check_backend(backend)

    if seed is None or isinstance(seed, int):
        if backend.__class__.__name__ in [
            "CupyBackend",
            "CuQuantumBackend",
        ]:  # pragma: no cover
            local_state = backend.np.random.default_rng(seed)
        else:
            local_state = np.random.default_rng(seed)
    else:
        local_state = seed

    return backend, local_state
