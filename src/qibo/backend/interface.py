from abc import ABC, abstractmethod
from qibo.config import raise_error
# TODO: Implement precision setter in backends


class BaseBackend(ABC):

    def __init__(self):
        self.backend = None
        self.name = "base"
        self._dtypes = {}

    def dtypes(self, name):
        if name in self._dtypes:
            dtype = self._dtypes.get(name)
        else:
            dtype = name
        return getattr(self.backend, dtype)

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


class NumpyBackend(BaseBackend):

    def __init__(self):
        import numpy as np
        self.backend = np
        self.name = "numpy"
        self.np = np
        self._dtypes = {'DTYPEINT': 'int64', 'DTYPE': 'float64',
                        'DTYPECPX': 'complex128'}

    @property
    def numeric_types(self):
        return (self.np.int, self.np.float, self.np.complex,
                self.np.int32, self.np.int64, self.np.float32,
                self.np.float64, self.np.complex64, self.np.complex128)

    @property
    def tensor_types(self):
        return (self.backend.ndarray,)

    @property
    def Tensor(self):
        return self.backend.ndarray

    @property
    def random(self):
        return self.backend.random

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        if isinstance(x, self.backend.ndarray):
            return x.astype(dtype)
        return self.backend.array(x, dtype=dtype)

    def diag(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.diag(x).astype(dtype)

    def reshape(self, x, shape):
        return self.backend.reshape(x, shape)

    def stack(self, x, axis=0):
        return self.backend.stack(x, axis=axis)

    def concatenate(self, x, axis=0):
        return self.backend.concatenate(x, axis=axis)

    @property
    def newaxis(self):
        return self.backend.newaxis

    def copy(self, x):
        return self.backend.copy(x)

    def range(self, start, stop, step, dtype=None):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.arange(start, stop, step, dtype=dtype)

    def eye(self, dim, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.eye(dim, dtype=dtype)

    def zeros(self, shape, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.ones(shape, dtype=dtype)

    def zeros_like(self, x):
        return self.backend.zeros_like(x)

    def ones_like(self, x):
        return self.backend.ones_like(x)

    def real(self, x):
        return self.backend.real(x)

    def imag(self, x):
        return self.backend.imag(x)

    def conj(self, x):
        return self.backend.conj(x)

    def mod(self, x, y):
        return self.backend.mod(x, y)

    def right_shift(self, x, y):
        return self.backend.right_shift(x, y)

    def exp(self, x):
        return self.backend.exp(x)

    def sin(self, x):
        return self.backend.sin(x)

    def cos(self, x):
        return self.backend.cos(x)

    def pow(self, base, exponent):
        return base ** exponent

    def square(self, x):
        return x ** 2

    def sqrt(self, x):
        return self.backend.sqrt(x)

    def log(self, x):
        return self.backend.log(x)

    def abs(self, x):
        return self.backend.abs(x)

    def trace(self, x):
        return self.backend.trace(x)

    def sum(self, x, axis=None):
        return self.backend.sum(x, axis=axis)

    def matmul(self, x, y):
        return self.backend.matmul(x, y)

    def outer(self, x, y):
        return self.backend.outer(x, y)

    def kron(self, x, y):
        return self.backend.kron(x, y)

    def einsum(self, *args):
        return self.backend.einsum(*args)

    def tensordot(self, x, y, axes=None):
        return self.backend.tensordot(x, y, axes=axes)

    def transpose(self, x, axes=None):
        return self.backend.transpose(x, axes)

    def inv(self, x):
        return self.backend.linalg.inv(x)

    def eigh(self, x):
        return self.backend.linalg.eigh(x)

    def eigvalsh(self, x):
        return self.backend.linalg.eigvalsh(x)

    def unique(self, x, return_counts=False):
        # Uses numpy backend always (even on Tensorflow)
        return self.np.unique(x, return_counts=return_counts)

    def gather(self, x, indices=None, condition=None, axis=0):
        if indices is None:
            if condition is None:
                raise_error(ValueError, "Gather call requires either indices "
                                        "or condition.")
            indices = condition
        if axis < 0:
            axis += len(x.shape)
        idx = axis * (slice(None),) + (indices,)
        return x[idx]

    def compile(self, func):
        return func

    def device(self, device_name):
        raise_error(NotImplementedError, "Device functionality is not "
                                         "available in the numpy backend.")

    @property
    def oom_error(self):
        raise_error(NotImplementedError)


class TensorflowBackend(NumpyBackend):

    def __init__(self):
        super().__init__()
        import tensorflow as tf
        self.backend = tf
        self.name = "tensorflow"

    @property
    def tensor_types(self):
        return (self.np.ndarray, self.backend.Tensor, self.backend.Variable)

    @property
    def Tensor(self):
        return self.backend.Tensor

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        if isinstance(x, self.np.ndarray):
            dtypestr = dtype.__repr__().split(".")[1]
            x = x.astype(getattr(self.np, dtypestr))
        return self.backend.cast(x, dtype=dtype)

    def diag(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.cast(self.backend.linalg.diag(x), dtype=dtype)

    def concatenate(self, x, axis=None):
        return self.backend.concat(x, axis=axis)

    def copy(self, x):
        return self.backend.identity(x)

    def range(self, start, stop, step, dtype=None):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.range(start, stop, step, dtype=dtype)

    def real(self, x):
        return self.backend.math.real(x)

    def imag(self, x):
        return self.backend.math.imag(x)

    def conj(self, x):
        return self.backend.math.conj(x)

    def mod(self, x, y):
        return self.backend.math.mod(x, y)

    def right_shift(self, x, y):
        return self.backend.bitwise.right_shift(x, y)

    def pow(self, base, exponent):
        return self.backend.math.pow(base, exponent)

    def square(self, x):
        return self.backend.square(x)

    def sqrt(self, x):
        return self.backend.math.sqrt(x)

    def log(self, x):
        return self.backend.math.log(x)

    def trace(self, x):
        return self.backend.linalg.trace(x)

    def sum(self, x, axis=None):
        return self.backend.reduce_sum(x, axis=axis)

    def outer(self, x, y):
        return self.tensordot(x, y, axes=0)

    def kron(self, x, y):
        raise_error(NotImplementedError)

    def inv(self, x):
        raise_error(NotImplementedError)

    def gather(self, x, indices=None, condition=None, axis=0):
        if indices is None:
            if condition is None:
                raise_error(ValueError, "Gather call is missing indices or "
                                        "condition.")
            indices = self.backend.where(condition)
        return self.backend.gather(x, indices, axis=axis)

    def compile(self, func):
        return self.backend.function(func)

    def device(self, device_name):
        return self.backend.device(device_name)

    @property
    def oom_error(self):
        return self.backend.python.framework.errors_impl.ResourceExhaustedError


function_names = [m for m in dir(BaseBackend) if m[:2] != "__"]

factory = {
    'numpy': NumpyBackend,
    'tensorflow': TensorflowBackend
}
