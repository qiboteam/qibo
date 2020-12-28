from qibo import config
from qibo.config import raise_error, warnings, BACKEND_NAME
from qibo.backends.numpy import NumpyBackend
from qibo.backends.tensorflow import TensorflowBackend


CONSTRUCTED_BACKENDS = {}
def construct_backend(name):
    if name not in CONSTRUCTED_BACKENDS:
        if name == "numpy":
            CONSTRUCTED_BACKENDS["numpy"] = NumpyBackend()
        elif name == "tensorflow":
            CONSTRUCTED_BACKENDS["tensorflow"] = TensorflowBackend()
        else:
            raise_error(ValueError, "Unknown backend name {}.".format(name))
    return CONSTRUCTED_BACKENDS.get(name)

numpy_backend = construct_backend("numpy")
numpy_matrices = numpy_backend.matrices

if BACKEND_NAME == "tensorflow":
    K = TensorflowBackend()
else:
    K = NumpyBackend()

def set_backend(backend="custom"):
    """Sets backend used to implement gates.

    Args:
        backend (str): possible options are 'custom' for the gates that use
            custom tensorflow operator and 'defaulteinsum' or 'matmuleinsum'
            for the gates that use tensorflow primitives (``tf.einsum`` or
            ``tf.matmul`` respectively).
    """
    if not config.ALLOW_SWITCHERS and backend != K.gates:
        warnings.warn("Backend should not be changed after allocating gates.",
                      category=RuntimeWarning)

    bk = construct_backend("tensorflow")
    K.assign(bk)
    K.set_gates(backend)


def get_backend():
    """Get backend used to implement gates.

    Returns:
        A string with the backend name.
    """
    return K.gates


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
    for bk in CONSTRUCTED_BACKENDS.values():
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
    for bk in CONSTRUCTED_BACKENDS.values():
        if bk.default_device is not None:
            bk.set_device(name)
            with bk.device(bk.default_device):
                bk.matrices.allocate_matrices()


def get_device():
    return K.default_device
