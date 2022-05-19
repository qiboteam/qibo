from qibo import config
from qibo.config import log, raise_error
from qibo.engines.numpy import NumpyEngine
from qibo.engines.tensorflow import TensorflowEngine


def construct_backend(backend, platform=None, show_error=False):
    def _initialize_backend(backend_cls, show_error=False):
        try:
            return backend_cls()
        except (ModuleNotFoundError, ImportError):
            if show_error:
                name = backend_cls.__name__
                raise_error(ModuleNotFoundError, f"{name} is not installed.")
            return None

    if backend == "qibojit":
        from qibojit.engines import CupyEngine, NumbaEngine
        if platform == "cupy":
            return _initialize_backend(CupyEngine, show_error)
        elif platform == "numba":
            return _initialize_backend(NumbaEngine, show_error)
        else:
            bk = _initialize_backend(CupyEngine)
            if bk is None:
                bk = _initialize_backend(NumbaEngine, show_error)
            return bk
        
    elif backend == "tensorflow":
        return _initialize_backend(TensorflowEngine, show_error)

    elif backend == "numpy":
        return _initialize_backend(NumpyEngine, show_error)

    else:
        # TODO: Fix errors to their previous format
        raise_error(ValueError, f"Backend {backend} is not available.")


class GlobalBackend:
    """The global backend will be used as default by ``circuit.execute()``."""

    _instance = None
    _dtypes = {"double": "complex128", "single": "complex64"}
    _default_order = [
        {"backend": "qibojit", "platform": "cupy"},
        {"backend": "qibojit", "platform": "numba"},
        {"backend": "tensorflow"},
        {"backend": "numpy"}
    ]

    def __new__(cls):
        if cls._instance is not None:
            return cls._instance
        
        # Create default backend
        for kwargs in cls._default_order:
            cls._instance = construct_backend(**kwargs)
            if cls._instance is not None:
                log.info(f"Using {cls._instance} backend on {cls._instance.device}")
                return cls._instance
        
        raise_error(RuntimeError, "No backends available.")

    @classmethod
    def set_backend(cls, backend, platform):
        if backend != cls._instance.name or platform != cls._instance.platform:
            if not config.ALLOW_SWITCHERS:
                log.warning("Backend should not be changed after allocating gates.")
            cls._instance = construct_backend(backend, platform, show_error=True)
        log.info(f"Using {cls._instance} backend on {cls._instance.device}")


def set_backend(backend="qibojit", platform=None):
    GlobalBackend.set_backend(backend, platform)


def get_backend():
    return GlobalBackend().name


def set_precision(precision):
    GlobalBackend().set_precision(precision)


def get_precision():
    GlobalBackend().get_precision()


def get_device():
    return GlobalBackend().device
