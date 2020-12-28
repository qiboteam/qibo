from qibo import config
from qibo.config import raise_error, warnings


class Backend:

    base_methods = {"assign", "set_gates", "dtypes",
                    "set_precision", "set_device"}

    def __init__(self):
        self.gates = None
        self.custom_gates = True
        self.custom_einsum = None

        self.precision = 'double'
        self._dtypes = {'DTYPEINT': 'int64', 'DTYPE': 'float64',
                        'DTYPECPX': 'complex128'}

        self.cpu_devices = []
        self.gpu_devices = []
        self.default_device = None

        self.matrices = None

    def assign(self, backend):
        """Assigns backend's methods."""
        for method in dir(backend):
            if method[:2] != "__" and method not in self.base_methods:
                setattr(self, method, getattr(backend, method))
        self.matrices = backend.matrices

    def set_gates(self, name):
        if name == 'custom':
            self.custom_gates = True
            self.custom_einsum = None
        elif name == 'defaulteinsum':
            from qibo.tensorflow import einsum
            self.custom_gates = False
            self.custom_einsum = einsum.DefaultEinsum()
        elif name == 'matmuleinsum':
            from qibo.tensorflow import einsum
            self.custom_gates = False
            self.custom_einsum = einsum.MatmulEinsum()
        else:
            raise_error(RuntimeError, f"Gate backend '{name}' not supported.")
        self.gates = name

    def dtypes(self, name):
        if name in self._dtypes:
            dtype = self._dtypes.get(name)
        else:
            dtype = name
        return getattr(self.backend, dtype)

    def set_precision(self, dtype):
        if dtype == 'single':
            self._dtypes['DTYPE'] = 'float32'
            self._dtypes['DTYPECPX'] = 'complex64'
        elif dtype == 'double':
            self._dtypes['DTYPE'] = 'float64'
            self._dtypes['DTYPECPX'] = 'complex128'
        else:
            raise_error(RuntimeError, f'dtype {dtype} not supported.')
        self.precision = dtype
        if self.matrices is not None:
            self.matrices.dtype = self.dtypes('DTYPECPX')

    def set_device(self, name):
        parts = name[1:].split(":")
        if name[0] != "/" or len(parts) < 2 or len(parts) > 3:
            raise_error(ValueError, "Device name should follow the pattern: "
                             "/{device type}:{device number}.")
        device_type, device_number = parts[-2], int(parts[-1])
        if device_type == "CPU":
            ndevices = len(self.cpu_devices)
        elif device_type == "GPU":
            ndevices = len(self.gpu_devices)
        else:
            raise_error(ValueError, f"Unknown device type {device_type}.")
        if device_number >= ndevices:
            raise_error(ValueError, f"Device {name} does not exist.")
        self.default_device = name


CONSTRUCTED_BACKENDS = {}
def construct_backend(name):
    if name not in CONSTRUCTED_BACKENDS:
        if name == "numpy":
            from qibo.backends import numpy
            CONSTRUCTED_BACKENDS["numpy"] = numpy.NumpyBackend()
        elif name == "tensorflow":
            from qibo.backends import tensorflow
            CONSTRUCTED_BACKENDS["tensorflow"] = tensorflow.TensorflowBackend()
        else:
            raise_error(ValueError, "Unknown backend name {}.".format(name))
    return CONSTRUCTED_BACKENDS.get(name)

numpy_backend = construct_backend("numpy")
numpy_matrices = numpy_backend.matrices

K = Backend()
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

set_backend()


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
