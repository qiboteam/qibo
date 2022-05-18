from qibo import config
from qibo.config import log, raise_error
from qibo.engines.numpy import NumpyEngine
from qibo.engines.tensorflow import TensorflowEngine


class GlobalSimulator:
    """The global simulator will be used as default by ``circuit.execute()``."""
    # TODO: Implement precision setters
    # TODO: Implement device setters

    _instance = NumpyEngine()
    _dtypes = {"double": "complex128", "single": "complex64"}

    def __new__(cls):
        """Creates singleton instance."""
        return cls._instance

    @classmethod
    def set_backend(cls, backend, platform):
        if not config.ALLOW_SWITCHERS and backend != cls._instance.name:
            log.warning("Backend should not be changed after allocating gates.")
        # TODO: Use ``profiles.yml`` here
        if backend != cls._instance.name:
            if backend == "numpy":
                cls._instance = NumpyEngine()
            elif backend == "tensorflow":
                cls._instance = TensorflowEngine()
            # TODO: Decide if cupy/numba will be treated as sepearate platforms
            # or completely different backends
            elif backend == "qibojit" and platform == "cupy":
                from qibojit import CupyEngine
                cls._instance = CupyEngine()
            elif backend == "qibojit" and platform == "cuquantum":
                raise_error(NotImplementedError)
            elif backend == "qibojit": # any other platform leads to numba
                from qibojit import NumbaEngine
                cls._instance = NumbaEngine()
            else:
                raise_error(NotImplementedError)

    @classmethod
    def set_precision(cls, precision):
        dtype = cls._dtypes.get(precision)
        if dtype != cls._instance.dtype:
            cls._instance = cls._instance.__class__(dtype)


def set_backend(backend="qibojit", platform=None):
    GlobalSimulator.set_backend(backend, platform)
    sim = GlobalSimulator()
    log.info(f"Using {sim} backend on ...")


def get_backend():
    return str(GlobalSimulator())


def set_precision(precision):
    GlobalSimulator.set_precision(precision)


def get_precision():
    dtype = GlobalSimulator().dtype
    if dtype == "complex64":
        return "single"
    else:
        return "double"


def get_device():
    return GlobalSimulator().device