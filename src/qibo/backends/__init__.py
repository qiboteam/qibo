import os
from qibo import config
from qibo.config import raise_error, log, warnings
from qibo.backends.numpy import NumpyBackend, IcarusQBackend
from qibo.backends.tensorflow import TensorflowBackend


_CONSTRUCTED_BACKENDS = {}
def _construct_backend(name):
    if name not in _CONSTRUCTED_BACKENDS:
        if name == "numpy":
            _CONSTRUCTED_BACKENDS["numpy"] = NumpyBackend()
        elif name == "tensorflow":
            _CONSTRUCTED_BACKENDS["tensorflow"] = TensorflowBackend()
        elif name == "icarusq":
            _CONSTRUCTED_BACKENDS["icarusq"] = IcarusQBackend()
        else:
            raise_error(ValueError, "Unknown backend name {}.".format(name))
    return _CONSTRUCTED_BACKENDS.get(name)

numpy_backend = _construct_backend("numpy")
numpy_matrices = numpy_backend.matrices

AVAILABLE_BACKENDS = ["custom", "defaulteinsum", "matmuleinsum",
                      "tensorflow_defaulteinsum", "tensorflow_matmuleinsum",
                      "numpy_defaulteinsum", "numpy_matmuleinsum", "icarusq"]


# Select the default backend engine
if "QIBO_BACKEND" in os.environ: # pragma: no cover
    _BACKEND_NAME = os.environ.get("QIBO_BACKEND")
    if _BACKEND_NAME == "tensorflow":
        K = TensorflowBackend()
    elif _BACKEND_NAME == "numpy": # pragma: no cover
        # CI uses tensorflow as default backend
        K = NumpyBackend()
    else: # pragma: no cover
        raise_error(ValueError, "Environment variable `QIBO_BACKEND` has "
                                "unknown value {}. Please select either "
                                "`tensorflow` or `numpy`."
                                "".format(_BACKEND_NAME))
else:
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(config.LOG_LEVEL)
        import tensorflow as tf
        import qibo.tensorflow.custom_operators as op
        _CUSTOM_OPERATORS_LOADED = op._custom_operators_loaded
        if not _CUSTOM_OPERATORS_LOADED: # pragma: no cover
            log.warning("Removing custom operators from available backends.")
            AVAILABLE_BACKENDS.remove("custom")
        K = TensorflowBackend()
    except ModuleNotFoundError: # pragma: no cover
        # case not tested because CI has tf installed
        log.warning("Tensorflow is not installed. Falling back to numpy.")
        K = NumpyBackend()
        AVAILABLE_BACKENDS = [b for b in AVAILABLE_BACKENDS
                              if "tensorflow" not in b]
        AVAILABLE_BACKENDS.remove("custom")


K.qnp = numpy_backend
_BACKEND_NAME = K.name
def set_backend(backend="custom"):
    """Sets backend used for mathematical operations and applying gates.

    The following backends are available:
    'custom': Tensorflow backend with custom operators for applying gates,
    'defaulteinsum': Tensorflow backend that applies gates using ``tf.einsum``,
    'matmuleinsum': Tensorflow backend that applies gates using ``tf.matmul``,
    'numpy_defaulteinsum': Numpy backend that applies gates using ``np.einsum``,
    'numpy_matmuleinsum': Numpy backend that applies gates using ``np.matmul``,

    Args:
        backend (str): A backend from the above options.
    """
    if backend not in AVAILABLE_BACKENDS:
        available = ", ".join(AVAILABLE_BACKENDS)
        raise_error(ValueError, "Unknown backend {}. Please select one of the "
                                "available backends: {}."
                                "".format(backend, available))
    if not config.ALLOW_SWITCHERS and backend != K.gates:
        warnings.warn("Backend should not be changed after allocating gates.",
                      category=RuntimeWarning)

    gate_backend = backend.split("_")
    if gate_backend == ["icarusq"]:
        calc_backend, gate_backend = "icarusq", "defaulteinsum"
    elif len(gate_backend) == 1:
        calc_backend, gate_backend = _BACKEND_NAME, gate_backend[0]
    elif len(gate_backend) == 2:
        calc_backend, gate_backend = gate_backend
    if gate_backend == "custom":
        calc_backend = "tensorflow"
    bk = _construct_backend(calc_backend)
    K.assign(bk)
    K.qnp = numpy_backend
    K.set_gates(gate_backend)


def get_backend():
    """Get backend used to implement gates.

    Returns:
        A string with the backend name.
    """
    if K.name == "tensorflow":
        return K.gates
    else:
        return "_".join([K.name, K.gates])


if _BACKEND_NAME != "tensorflow": # pragma: no cover
    # CI uses tensorflow as default backend
    log.warning("{} does not support Qibo custom operators and GPU. "
                "Einsum will be used to apply gates on CPU."
                "".format(_BACKEND_NAME))
    set_backend("defaulteinsum")


if _BACKEND_NAME == "tensorflow" and not _CUSTOM_OPERATORS_LOADED: # pragma: no cover
    log.warning("Einsum will be used to apply gates with tensorflow.")
    set_backend("defaulteinsum")


def set_precision(dtype='double'):
    """Set precision for states and gates simulation.

    Args:
        dtype (str): possible options are 'single' for single precision
            (complex64) and 'double' for double precision (complex128).
    """
    if not config.ALLOW_SWITCHERS and dtype != K.precision:
        warnings.warn("Precision should not be changed after allocating gates.",
                      category=RuntimeWarning)
    K.set_precision(dtype)
    for bk in _CONSTRUCTED_BACKENDS.values():
        bk.set_precision(dtype)
        bk.matrices.allocate_matrices()


def get_precision():
    """Get precision for states and gates simulation.

    Returns:
        A string with the precision name ('single', 'double').
    """
    return K.precision


def set_device(name):
    """Set default execution device.

    Args:
        name (str): Device name. Should follow the pattern
            '/{device type}:{device number}' where device type is one of
            CPU or GPU.
    """
    if not config.ALLOW_SWITCHERS and name != K.default_device:
        warnings.warn("Device should not be changed after allocating gates.",
                      category=RuntimeWarning)
    K.set_device(name)
    for bk in _CONSTRUCTED_BACKENDS.values():
        if bk.default_device is not None:
            bk.set_device(name)
            with bk.device(bk.default_device):
                bk.matrices.allocate_matrices()


def get_device():
    return K.default_device
