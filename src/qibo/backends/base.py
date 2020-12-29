from abc import ABC, abstractmethod
from qibo.config import raise_error


class BaseBackend(ABC):

    base_methods = {"assign", "set_gates", "dtypes",
                    "set_precision", "set_device"}

    def __init__(self):
        self.backend = None
        self.name = "base"

        self.gates = "custom"
        self.custom_gates = True
        self.custom_einsum = None

        self.precision = 'double'
        self._dtypes = {'DTYPEINT': 'int64', 'DTYPE': 'float64',
                        'DTYPECPX': 'complex128'}

        self.cpu_devices = []
        self.gpu_devices = []
        self.default_device = None

        self.matrices = None
        self.numeric_types = None
        self.tensor_types = None
        self.Tensor = None
        self.random = None
        self.newaxis = None
        self.oom_error = None
        self.optimization = None

    def assign(self, backend):
        """Assigns backend's methods."""
        for method in dir(backend):
            if method[:2] != "__" and method not in self.base_methods:
                setattr(self, method, getattr(backend, method))
        self.matrices = backend.matrices
        self.numeric_types = backend.numeric_types
        self.tensor_types = backend.tensor_types
        self.Tensor = backend.Tensor
        self.random = backend.random
        self.newaxis = backend.newaxis
        self.oom_error = backend.oom_error
        self.optimization = backend.optimization

    def set_gates(self, name):
        if name == 'custom':
            self.custom_gates = True
            self.custom_einsum = None
        elif name == 'defaulteinsum':
            self.custom_gates = False
            self.custom_einsum = "DefaultEinsum"
        elif name == 'matmuleinsum':
            self.custom_gates = False
            self.custom_einsum = "MatmulEinsum"
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

    @abstractmethod
    def expand_dims(self, x, axis):
        raise_error(NotImplementedError)

    @abstractmethod
    def copy(self, x):
        """Creates a copy of the tensor in memory."""
        raise_error(NotImplementedError)

    @abstractmethod
    def range(self, start, finish, step, dtype=None):
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
    def expm(self, x):
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
    def array_equal(self, x, y):
        """Used in :meth:`qibo.tensorflow.hamiltonians.TrotterHamiltonian.construct_terms`."""
        raise_error(NotImplementedError)

    @abstractmethod
    def gather(self, x, indices=None, condition=None, axis=0):
        raise_error(NotImplementedError)

    @abstractmethod
    def gather_nd(self, x, indices):
        raise_error(NotImplementedError)

    @abstractmethod
    def sample_measurements(self, probs, nshots):
        raise_error(NotImplementedError)

    @abstractmethod
    def compile(self, func):
        raise_error(NotImplementedError)

    @abstractmethod
    def device(self, device_name):
        raise_error(NotImplementedError)

    def executing_eagerly(self):
        return True
