from abc import ABC, abstractmethod
from qibo.config import raise_error


class BaseBackend(ABC):

    def __init__(self):
        self.backend = None
        self.name = "base"
        self._dtypes = {'STRING': 'double', 'DTYPEINT': 'int64',
                        'DTYPE': 'float64', 'DTYPECPX': 'complex128'}

        self.cpu_devices = None
        self.gpu_devices = None
        self._active_device = None

    def dtypes(self, name):
        if name == 'STRING':
            return self._dtypes.get('STRING')
        if name in self._dtypes:
            dtype = self._dtypes.get(name)
        else:
            dtype = name
        return getattr(self.backend, dtype)

    @property
    def active_device(self):
        return self._active_device

    def set_precision(self, dtype='double'):
        from qibo.config import ALLOW_SWITCHERS, warnings
        if not ALLOW_SWITCHERS and dtype != self._dtypes['STRING']:
            warnings.warn("Precision should not be changed after allocating gates.",
                          category=RuntimeWarning)
        if dtype == 'single':
            self._dtypes['DTYPE'] = 'float32'
            self._dtypes['DTYPECPX'] = 'complex64'
        elif dtype == 'double':
            self._dtypes['DTYPE'] = 'float64'
            self._dtypes['DTYPECPX'] = 'complex128'
        else:
            raise_error(RuntimeError, f'dtype {dtype} not supported.')
        self._dtypes['STRING'] = dtype

    def set_device(self, name):
        """Set default execution device.

        Args:
            name (str): Device name. Should follow the pattern
                '/{device type}:{device number}' where device type is one of
                CPU or GPU.
        """
        from qibo.config import ALLOW_SWITCHERS, warnings
        if not ALLOW_SWITCHERS and name != self._active_device:  # pragma: no cover
            # no testing is implemented for warnings
            warnings.warn("Device should not be changed after allocating gates.",
                          category=RuntimeWarning)
        parts = name[1:].split(":")
        if name[0] != "/" or len(parts) < 2 or len(parts) > 3:
            raise_error(ValueError, "Device name should follow the pattern: "
                             "/{device type}:{device number}.")
        device_type, device_number = parts[-2], int(parts[-1])
        if device_type not in {"CPU", "GPU"}:
            raise_error(ValueError, f"Unknown device type {device_type}.")
        if device_number >= len(self._devices[device_type]):
            raise_error(ValueError, f"Device {name} does not exist.")
        self._active_device = name

    @property
    @abstractmethod
    def numeric_types(self):
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def tensor_types(self):
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def Tensor(self):
        """Type of tensor object that is compatible to the backend."""
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def random(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def cast(self, x, dtype='DTYPECPX'):
        """Casts tensor to the given dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def diag(self, x, dtype='DTYPECPX'):
        raise_error(NotImplementedError)

    @abstractmethod
    def reshape(self, x, shape):
        """Reshapes tensor in the given shape."""
        raise_error(NotImplementedError)

    @abstractmethod
    def stack(self, x, axis=None):
        """Stacks a list of tensors to a single tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def concatenate(self, x, axis=None):
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def newaxis(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def copy(self, x):
        """Creates a copy of the tensor in memory."""
        raise_error(NotImplementedError)

    @abstractmethod
    def range(start, finish, step, dtype=None):
        raise_error(NotImplementedError)

    @abstractmethod
    def eye(self, dim, dtype='DTYPECPX'):
        raise_error(NotImplementedError)

    @abstractmethod
    def zeros(self, shape, dtype='DTYPECPX'):
        """Creates tensor of zeros with the given shape and dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def ones(self, shape, dtype='DTYPECPX'):
        raise_error(NotImplementedError)

    @abstractmethod
    def zeros_like(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def ones_like(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def real(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def imag(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def conj(self, x):
        """Elementwise complex conjugate of a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def mod(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def right_shift(self, x, y):
        raise_error(NotImplementedError)

    @abstractmethod
    def exp(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def sin(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def cos(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def pow(self, base, exponent):
        raise_error(NotImplementedError)

    @abstractmethod
    def square(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def sqrt(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def log(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def abs(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def trace(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def sum(self, x, axis=None):
        """Sum of tensor elements."""
        raise_error(NotImplementedError)

    @abstractmethod
    def matmul(self, x, y):
        """Matrix multiplication of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def outer(self, x, y):
        """Outer (Kronecker) product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def kron(self, x, y):
        raise_error(NotImplementedError)

    @abstractmethod
    def einsum(self, *args):
        """Generic tensor operation based on Einstein's summation convention."""
        raise_error(NotImplementedError)

    @abstractmethod
    def tensordot(self, x, y, axes=None):
        """Generalized tensor product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def transpose(self, x, axes=None):
        raise_error(NotImplementedError)

    @abstractmethod
    def inv(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def eigh(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def eigvalsh(self, x):
        raise_error(NotImplementedError)

    @abstractmethod
    def unique(self, x, return_counts=False):
        raise_error(NotImplementedError)

    @abstractmethod
    def gather(self, x, indices=None, condition=None, axis=0):
        raise_error(NotImplementedError)

    @abstractmethod
    def compile(self, func):
        raise_error(NotImplementedError)

    @abstractmethod
    def device(self, device_name):
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def oom_error(self):
        raise_error(NotImplementedError)
