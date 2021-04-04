import os
from qibo import config
from qibo.config import raise_error, log, warnings
from qibo.backends.numpy import NumpyDefaultEinsumBackend, NumpyMatmulEinsumBackend, IcarusQBackend
from qibo.backends.tensorflow import TensorflowCustomBackend, TensorflowDefaultEinsumBackend, TensorflowMatmulEinsumBackend


AVAILABLE_BACKENDS = {
    "custom": TensorflowCustomBackend,
    "tensorflow": TensorflowCustomBackend,
    "defaulteinsum": TensorflowDefaultEinsumBackend,
    "matmuleinsum": TensorflowMatmulEinsumBackend,
    "tensorflow_defaulteinsum": TensorflowDefaultEinsumBackend,
    "tensorflow_matmuleinsum": TensorflowMatmulEinsumBackend,
    "numpy": NumpyDefaultEinsumBackend,
    "numpy_defaulteinsum": NumpyDefaultEinsumBackend,
    "numpy_matmuleinsum": NumpyMatmulEinsumBackend,
    "icarusq": IcarusQBackend
}

_CONSTRUCTED_BACKENDS = {}
def _construct_backend(name):
    if name not in _CONSTRUCTED_BACKENDS:
        if name not in AVAILABLE_BACKENDS:
            available = ", ".join(list(AVAILABLE_BACKENDS.keys()))
            raise_error(ValueError, "Unknown backend {}. Please select one of "
                                    "the available backends: {}."
                                    "".format(name, available))
        _CONSTRUCTED_BACKENDS[name] = AVAILABLE_BACKENDS.get(name)()
    return _CONSTRUCTED_BACKENDS.get(name)

numpy_backend = _construct_backend("numpy_defaulteinsum")
numpy_matrices = numpy_backend.matrices


# Select the default backend engine
if "QIBO_BACKEND" in os.environ: # pragma: no cover
    _BACKEND_NAME = os.environ.get("QIBO_BACKEND")
    if _BACKEND_NAME not in AVAILABLE_BACKENDS: # pragma: no cover
        _available_names = ", ".join(list(AVAILABLE_BACKENDS.keys()))
        raise_error(ValueError, "Environment variable `QIBO_BACKEND` has "
                                "unknown value {}. Please select one of {}."
                                "".format(_BACKEND_NAME, _available_names))
    K = AVAILABLE_BACKENDS.get(_BACKEND_NAME)()
else:
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(config.LOG_LEVEL)
        import tensorflow as tf
        import qibo.tensorflow.custom_operators as op
        if not op._custom_operators_loaded: # pragma: no cover
            log.warning("Einsum will be used to apply gates with Tensorflow. "
                        "Removing custom operators from available backends.")
            AVAILABLE_BACKENDS.pop("custom")
            AVAILABLE_BACKENDS["tensorflow"] = TensorflowDefaultEinsumBackend
        K = AVAILABLE_BACKENDS.get("tensorflow")()
    except ModuleNotFoundError: # pragma: no cover
        # case not tested because CI has tf installed
        log.warning("Tensorflow is not installed. Falling back to numpy. "
                    "Numpy does not support Qibo custom operators and GPU. "
                    "Einsum will be used to apply gates on CPU.")
        AVAILABLE_BACKENDS = {k: v for k, v in AVAILABLE_BACKENDS.items()
                              if "numpy" in k}
        K = AVAILABLE_BACKENDS.get("numpy")()

    try:
        from qibo.hardware.experiments import IcarusQ
    except ModuleNotFoundError: # pragma: no cover
        AVAILABLE_BACKENDS.remove("icarusq")


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
    bk = _construct_backend(backend)
    if not config.ALLOW_SWITCHERS and backend != K.name:
        warnings.warn("Backend should not be changed after allocating gates.",
                      category=RuntimeWarning)
    K.assign(bk)
    K.qnp = numpy_backend


def get_backend():
    """Get backend used to implement gates.

    Returns:
        A string with the backend name.
    """
    return K.name


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
