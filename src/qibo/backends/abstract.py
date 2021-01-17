from abc import ABC, abstractmethod
from qibo.config import raise_error

_AVAILABLE_BACKENDS = ["custom", "defaulteinsum", "matmuleinsum",
                       "numpy_defaulteinsum", "numpy_matmuleinsum"]


class AbstractBackend(ABC):

    base_methods = {"assign", "set_gates", "dtypes",
                    "set_precision"}

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
        self.op = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}Backend".format(self.name.capitalize())

    def assign(self, backend):
        """Assigns backend's methods."""
        for method in dir(backend):
            if method[:2] != "__" and method not in self.base_methods:
                setattr(self, method, getattr(backend, method))
        self.name = backend.name
        self.matrices = backend.matrices
        self.numeric_types = backend.numeric_types
        self.tensor_types = backend.tensor_types
        self.Tensor = backend.Tensor
        self.random = backend.random
        self.newaxis = backend.newaxis
        self.oom_error = backend.oom_error
        self.optimization = backend.optimization
        self.op = backend.op

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
            raise_error(ValueError, f"Gate backend '{name}' not supported.")
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
            raise_error(ValueError, f'dtype {dtype} not supported.')
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
    def cast(self, x, dtype='DTYPECPX'): # pragma: no cover
        """Casts tensor to the given dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def diag(self, x, dtype='DTYPECPX'): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def reshape(self, x, shape): # pragma: no cover
        """Reshapes tensor in the given shape."""
        raise_error(NotImplementedError)

    @abstractmethod
    def stack(self, x, axis=None): # pragma: no cover
        """Stacks a list of tensors to a single tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def concatenate(self, x, axis=None): # pragma: no cover
        """Concatenates a list of tensor along a given axis."""
        raise_error(NotImplementedError)

    @abstractmethod
    def expand_dims(self, x, axis): # pragma: no cover
        """Creates a new axis of dimension one."""
        raise_error(NotImplementedError)

    @abstractmethod
    def copy(self, x): # pragma: no cover
        """Creates a copy of the tensor in memory."""
        raise_error(NotImplementedError)

    @abstractmethod
    def range(self, start, finish, step, dtype=None): # pragma: no cover
        """Creates a tensor of integers from start to finish."""
        raise_error(NotImplementedError)

    @abstractmethod
    def eye(self, dim, dtype='DTYPECPX'): # pragma: no cover
        """Creates the identity matrix as a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def zeros(self, shape, dtype='DTYPECPX'): # pragma: no cover
        """Creates tensor of zeros with the given shape and dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def ones(self, shape, dtype='DTYPECPX'): # pragma: no cover
        """Creates tensor of ones with the given shape and dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def zeros_like(self, x): # pragma: no cover
        """Creates tensor of zeros with shape and dtype of the given tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def ones_like(self, x): # pragma: no cover
        """Creates tensor of ones with shape and dtype of the given tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def real(self, x): # pragma: no cover
        """Real part of a given complex tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def imag(self, x): # pragma: no cover
        """Imaginary part of a given complex tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def conj(self, x): # pragma: no cover
        """Elementwise complex conjugate of a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def mod(self, x): # pragma: no cover
        """Elementwise mod operation."""
        raise_error(NotImplementedError)

    @abstractmethod
    def right_shift(self, x, y): # pragma: no cover
        """Elementwise bitwise right shift."""
        raise_error(NotImplementedError)

    @abstractmethod
    def exp(self, x): # pragma: no cover
        """Elementwise exponential."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sin(self, x): # pragma: no cover
        """Elementwise sin."""
        raise_error(NotImplementedError)

    @abstractmethod
    def cos(self, x): # pragma: no cover
        """Elementwise cos."""
        raise_error(NotImplementedError)

    @abstractmethod
    def pow(self, base, exponent): # pragma: no cover
        """Elementwise power."""
        raise_error(NotImplementedError)

    @abstractmethod
    def square(self, x): # pragma: no cover
        """Elementwise square."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sqrt(self, x): # pragma: no cover
        """Elementwise square root."""
        raise_error(NotImplementedError)

    @abstractmethod
    def log(self, x): # pragma: no cover
        """Elementwise natural logarithm."""
        raise_error(NotImplementedError)

    @abstractmethod
    def abs(self, x): # pragma: no cover
        """Elementwise absolute value."""
        raise_error(NotImplementedError)

    @abstractmethod
    def expm(self, x): # pragma: no cover
        """Matrix exponential."""
        raise_error(NotImplementedError)

    @abstractmethod
    def trace(self, x): # pragma: no cover
        """Matrix trace."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sum(self, x, axis=None): # pragma: no cover
        """Sum of tensor elements."""
        raise_error(NotImplementedError)

    @abstractmethod
    def matmul(self, x, y): # pragma: no cover
        """Matrix multiplication of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def outer(self, x, y): # pragma: no cover
        """Outer (Kronecker) product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def kron(self, x, y): # pragma: no cover
        """Outer (Kronecker) product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def einsum(self, *args): # pragma: no cover
        """Generic tensor operation based on Einstein's summation convention."""
        raise_error(NotImplementedError)

    @abstractmethod
    def tensordot(self, x, y, axes=None): # pragma: no cover
        """Generalized tensor product of two tensors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def transpose(self, x, axes=None): # pragma: no cover
        """Tensor transpose."""
        raise_error(NotImplementedError)

    @abstractmethod
    def inv(self, x): # pragma: no cover
        """Matrix inversion."""
        raise_error(NotImplementedError)

    @abstractmethod
    def eigh(self, x): # pragma: no cover
        """Hermitian matrix eigenvalues and eigenvectors."""
        raise_error(NotImplementedError)

    @abstractmethod
    def eigvalsh(self, x): # pragma: no cover
        """Hermitian matrix eigenvalues."""
        raise_error(NotImplementedError)

    @abstractmethod
    def unique(self, x, return_counts=False): # pragma: no cover
        """Identifies unique elements in a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def array_equal(self, x, y): # pragma: no cover
        """Checks if two arrays are equal elementwise.

        Used in :meth:`qibo.tensorflow.hamiltonians.TrotterHamiltonian.construct_terms`.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def gather(self, x, indices=None, condition=None, axis=0): # pragma: no cover
        """Indexing of one-dimensional tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def gather_nd(self, x, indices): # pragma: no cover
        """Indexing of multi-dimensional tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def initial_state(self, nqubits, is_matrix=False): # pragma: no cover
        """Creates the default initial state |00...0> as a tensor."""
        raise_error(NotImplementedError)

    @abstractmethod
    def sample_measurements(self, probs, nshots): # pragma: no cover
        """Samples measurements from a given probability distribution.

        Measurements are returned in decimal as a tensor of shape ``(nshots,)``.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def compile(self, func): # pragma: no cover
        """Compiles the graph of a given function.

        Relevant for Tensorflow, not numpy.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def device(self, device_name): # pragma: no cover
        """Used to execute code in specific device if supported by backend."""
        raise_error(NotImplementedError)

    def executing_eagerly(self):
        return True

    @abstractmethod
    def set_seed(self, seed): # pragma: no cover
        raise_error(NotImplementedError)

    @abstractmethod
    def assert_allclose(self, actual, desired, atol=0): # pragma: no cover
        raise_error(NotImplementedError)
