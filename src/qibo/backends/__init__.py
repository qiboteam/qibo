import os
from importlib import import_module

import networkx as nx
import numpy as np

from qibo.backends.abstract import Backend
from qibo.backends.clifford import CliffordBackend
from qibo.backends.hamming_weight import HammingWeightBackend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.config import log, raise_error

QIBO_NATIVE_BACKENDS = ("numpy", "qulacs")


class MissingBackend(ValueError):
    """Impossible to locate backend provider package."""


class MetaBackend:
    """Meta-backend class which takes care of loading the qibo backends."""

    @staticmethod
    def load(backend: str, **kwargs) -> Backend:
        """Loads the native qibo backend.

        Args:
            backend (str): Name of the backend to load.
            kwargs (dict): Additional arguments for the ``qibo`` backend.

        Returns:
            :class:`qibo.backends.abstract.Backend`: Loaded backend.
        """

        if backend not in QIBO_NATIVE_BACKENDS + ("clifford", "hamming_weight"):
            raise_error(
                ValueError,
                f"Backend {backend} is not available. "
                + f"The native qibo backends are {QIBO_NATIVE_BACKENDS + ('clifford', 'hamming_weight')}",
            )

        if backend in ["clifford", "hamming_weight"]:
            backend_class = (
                CliffordBackend if backend == "clifford" else HammingWeightBackend
            )

            engine = kwargs.pop("platform", None)
            kwargs["engine"] = engine

            return backend_class(**kwargs)

        dtype = kwargs.pop("dtype", "complex128")

        if backend == "qulacs":
            from qibo.backends.qulacs import QulacsBackend  # pylint: disable=C0415

            backend_obj = QulacsBackend()
            backend_obj.set_dtype(dtype=dtype)

            return backend_obj

        backend_obj = NumpyBackend()
        backend_obj.set_dtype(dtype=dtype)

        return backend_obj

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


class _Global:
    _backend = None
    _transpiler = None
    # TODO: resolve circular import with qibo.transpiler.pipeline.Passes

    _default_order = [
        {"backend": "qibojit", "platform": "cupy"},
        {"backend": "qibojit", "platform": "numba"},
        {"backend": "numpy"},
        {"backend": "qiboml", "platform": "tensorflow"},
        {"backend": "qiboml", "platform": "pytorch"},
    ]

    @classmethod
    def backend(cls):
        """Get the current backend. If no backend is set, it will create one."""
        if cls._backend is not None:
            return cls._backend
        cls._backend = cls._create_backend()
        log.info(f"Using {cls._backend} backend on {cls._backend.device}")
        return cls._backend

    @classmethod
    def _create_backend(cls):
        backend_env = os.environ.get("QIBO_BACKEND")
        if backend_env:  # pragma: no cover
            # Create backend specified by user
            platform = os.environ.get("QIBO_PLATFORM")
            backend = construct_backend(backend_env, platform=platform)
        else:
            # Create backend according to default order
            for kwargs in cls._default_order:
                try:
                    backend = construct_backend(**kwargs)
                    break
                except (ImportError, MissingBackend):
                    pass

        if backend is None:  # pragma: no cover
            raise_error(RuntimeError, "No backends available.")
        return backend

    @classmethod
    def set_backend(cls, backend, **kwargs):
        cls._backend = construct_backend(backend, **kwargs)
        cls._transpiler = None
        log.info(f"Using {cls._backend} backend on {cls._backend.device}")

    @classmethod
    def transpiler(cls):
        """Get the current transpiler. If no transpiler is set, it will create one."""
        if cls._transpiler is not None:
            return cls._transpiler

        cls._transpiler = cls._default_transpiler()
        return cls._transpiler

    @classmethod
    def set_transpiler(cls, transpiler):
        cls._transpiler = transpiler
        # TODO: check if transpiler is valid on the backend

    @classmethod
    def _default_transpiler(cls):
        from qibo.transpiler.optimizer import Preprocessing
        from qibo.transpiler.pipeline import Passes
        from qibo.transpiler.router import Sabre
        from qibo.transpiler.unroller import NativeGates, Unroller

        qubits = cls._backend.qubits
        natives = cls._backend.natives
        connectivity_edges = cls._backend.connectivity
        if qubits is not None and natives is not None:
            connectivity = (
                nx.Graph(connectivity_edges)
                if connectivity_edges is not None
                else nx.Graph()
            )
            connectivity.add_nodes_from(qubits)

            return Passes(
                connectivity=connectivity,
                passes=[
                    Preprocessing(),
                    Sabre(),
                    Unroller(NativeGates[natives]),
                ],
            )
        return Passes(passes=[])


class QiboMatrices:
    def __init__(self, dtype="complex128"):
        self.create(dtype)

    def create(self, dtype):
        self.matrices = NumpyMatrices(dtype)
        self.I = self.matrices.I(2)
        self.X = self.matrices.X
        self.Z = self.matrices.Z
        self.H = self.matrices.H
        self.CNOT = self.matrices.CNOT
        self.CZ = self.matrices.CZ
        self.SWAP = self.matrices.SWAP
        self.FSWAP = self.matrices.FSWAP
        self.TOFFOLI = self.matrices.TOFFOLI
        self.CCZ = self.matrices.CCZ

        if dtype in ("complex64", "complex128"):
            self.Y = self.matrices.Y
            self.SX = self.matrices.SX
            self.S = self.matrices.S
            self.T = self.matrices.T
            self.SDG = self.matrices.SDG
            self.CY = self.matrices.CY
            self.CSX = self.matrices.CSX
            self.CSXDG = self.matrices.CSXDG
            self.iSWAP = self.matrices.iSWAP
            self.SiSWAP = self.matrices.SiSWAP
            self.SiSWAPDG = self.matrices.SiSWAPDG
            self.SYC = self.matrices.SYC
            self.ECR = self.matrices.ECR


matrices = QiboMatrices()


def get_backend():
    """Get the current backend."""
    return _Global.backend()


def set_backend(backend, **kwargs):
    """Set the current backend.

    Args:
        backend (str): Name of the backend to use.
        kwargs (dict): Additional arguments for the backend.
    """
    _Global.set_backend(backend, **kwargs)


def get_transpiler():
    """Get the current transpiler."""
    return _Global.transpiler()


def get_transpiler_name():
    """Get the name of the current transpiler as a string."""
    return str(_Global.transpiler())


def set_transpiler(transpiler):
    """Set the current transpiler.

    Args:
        transpiler (Passes): The transpiler to use.
    """
    _Global.set_transpiler(transpiler)


def get_dtype():
    """Get the data type of the backend."""
    return get_backend().dtype


def set_dtype(dtype):
    """Set the data type of the backend.

    Args:
        dtype (str): data type to use. Options are the following: ``"complex128"``,
            ``"complex64"``, ``"float64"``, and ``"float32"``.
    """
    get_backend().set_dtype(dtype)
    matrices.create(get_backend().dtype)


def get_device():
    """Get the device of the backend."""
    return get_backend().device


def set_device(device):
    """Set the device of the backend.

    Args:
        device (str): Device to use.
    """
    parts = device[1:].split(":")
    if device[0] != "/" or len(parts) < 2 or len(parts) > 3:
        raise_error(
            ValueError,
            "Device name should follow the pattern: /{device type}:{device number}.",
        )
    backend = get_backend()
    backend.set_device(device)
    log.info(f"Using {backend} backend on {backend.device}")


def get_threads():
    """Get the number of threads used by the backend."""
    return get_backend().nthreads


def set_threads(nthreads):
    """Set the number of threads used by the backend.

    Args:
        nthreads (int): Number of threads to use.
    """
    if not isinstance(nthreads, int):
        raise_error(TypeError, "Number of threads must be integer.")
    if nthreads < 1:
        raise_error(ValueError, "Number of threads must be positive.")
    get_backend().set_threads(nthreads)


def _check_backend(backend):
    if backend is None:
        return get_backend()

    return backend


def _get_engine_name(backend):
    return backend.platform if backend.platform is not None else backend.name


def list_available_backends(*providers: str) -> dict:
    """Lists all the backends that are available."""
    available_backends = MetaBackend().list_available()
    for backend in providers:
        try:
            module = import_module(backend.replace("-", "_"))
            available = getattr(module, "MetaBackend")().list_available()
        except:
            available = False
        available_backends.update({backend: available})
    return available_backends


def construct_backend(backend, **kwargs) -> Backend:  # pylint: disable=R1710
    """Construct a generic native or non-native qibo backend.

    Args:
        backend (str): Name of the backend to load.
        kwargs (dict): Additional arguments for constructing the backend.
    Returns:
        qibo.backends.abstract.Backend: The loaded backend.
    """
    if backend in QIBO_NATIVE_BACKENDS + ("clifford", "hamming_weight"):
        return MetaBackend.load(backend, **kwargs)

    if backend not in QIBO_NATIVE_BACKENDS + ("clifford",):
        provider = backend.replace("-", "_")
        try:
            module = import_module(provider)
            return getattr(module, "MetaBackend").load(**kwargs)
        except ImportError as e:
            # pylint: disable=unsupported-membership-test
            if provider not in e.msg:
                raise e
            raise MissingBackend(
                f"The '{backend}' backends' provider is not available. Check that a Python "
                + f"package named '{provider}' is installed, and it is exposing valid Qibo "
                + "backends.",
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
