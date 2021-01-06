import os
from qibo import config
from qibo.config import raise_error, log, warnings
from qibo.backends.abstract import _AVAILABLE_BACKENDS
from qibo.backends.numpy import NumpyBackend
from qibo.backends.tensorflow import TensorflowBackend


_CONSTRUCTED_BACKENDS = {}
def _construct_backend(name):
    if name not in _CONSTRUCTED_BACKENDS:
        if name == "numpy":
            _CONSTRUCTED_BACKENDS["numpy"] = NumpyBackend()
        elif name == "tensorflow":
            _CONSTRUCTED_BACKENDS["tensorflow"] = TensorflowBackend()
        else: # pragma: no cover
            raise_error(ValueError, "Unknown backend name {}.".format(name))
    return _CONSTRUCTED_BACKENDS.get(name)

numpy_backend = _construct_backend("numpy")
numpy_matrices = numpy_backend.matrices


# Select the default backend engine
if "QIBO_BACKEND" in os.environ: # pragma: no cover
    if os.environ.get("QIBO_BACKEND") == "tensorflow":
        K = TensorflowBackend()
    else: # pragma: no cover
        # CI uses tensorflow as default backend
        K = NumpyBackend()
else:
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(config.LOG_LEVEL)
        import tensorflow as tf
        K = TensorflowBackend()
    except ModuleNotFoundError: # pragma: no cover
        # case not tested because CI has tf installed
        log.warning("Tensorflow is not installed. Falling back to numpy.")
        log.warning("Numpy does not support custom operators and GPU.")
        K = NumpyBackend()

_BACKEND_NAME = K.name
if _BACKEND_NAME != "tensorflow": # pragma: no cover
    # CI uses tensorflow as default backend
    log.warning("Numpy does not support Qibo custom operators. "
                "Einsum will be used to apply gates.")
    set_backend("defaulteinsum")


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
    if backend not in _AVAILABLE_BACKENDS:
        available = ", ".join(_AVAILABLE_BACKENDS)
        raise_error(ValueError, "Unknown backend {}. Please select one of the "
                                "available backends: {}."
                                "".format(backend, available))
    if not config.ALLOW_SWITCHERS and backend != K.gates:
        warnings.warn("Backend should not be changed after allocating gates.",
                      category=RuntimeWarning)

    gate_backend = backend.split("_")
    if len(gate_backend) == 1:
        calc_backend, gate_backend = _BACKEND_NAME, gate_backend[0]
    elif len(gate_backend) == 2:
        calc_backend, gate_backend = gate_backend
    bk = _construct_backend(calc_backend)
    K.assign(bk)
    if K.name != "tensorflow" and gate_backend == "custom": # pragma: no cover
        raise_error(ValueError, "Custom gates cannot be used with {} backend."
                                "".format(K.name))
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
    if not config.ALLOW_SWITCHERS and name != K.default_device: # pragma: no cover
        # no testing is implemented for warnings
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
