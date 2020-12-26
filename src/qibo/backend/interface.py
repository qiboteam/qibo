from abc import ABC, abstractmethod
from qibo.config import raise_error
# TODO: Implement precision setter in backends


class BaseBackend(ABC):

    def __init__(self):
        self.backend = None
        self.name = "base"
        self._dtypes = {}

    def dtypes(self, name):
        return getattr(self.backend, self._dtypes.get(name))

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
    def tensortype(self):
        """Type of tensor object that is compatible to the backend."""
        raise_error(NotImplementedError)

    @abstractmethod
    def cast(self, x, dtype=None):
        """Casts tensor to the given dtype."""
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
    def zeros(self, shape, dtype=None):
        """Creates tensor of zeros with the given shape and dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def ones(self, shape, dtype=None):
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
    def einsum(self, *args):
        """Generic tensor operation based on Einstein's summation convention."""
        raise_error(NotImplementedError)

    @abstractmethod
    def tensordot(self, x, y, axes=None):
        """Generalized tensor product of two tensors."""
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
    def tensortype(self):
        return self.backend.ndarray

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return x.astype(dtype)

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

    def zeros(self, shape, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.ones(shape, dtype=dtype)

    def ones_like(self, x):
        return self.backend.ones_like(x)

    def real(self, x):
        return self.backend.real(x)

    def imag(self, x):
        return self.backend.imag(x)

    def conj(self, x):
        return self.backend.conj(x)

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

    def einsum(self, *args):
        return self.backend.einsum(*args)

    def tensordot(self, x, y, axes=None):
        return self.backend.tensordot(x, y, axes=axes)

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
    def tensortype(self):
        return self.backend.Tensor

    def cast(self, x, dtype='DTYPECPX'):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        if isinstance(x, self.np.ndarray):
            dtypestr = dtype.__repr__().split(".")[1]
            x = x.astype(getattr(self.np, dtypestr))
        return self.backend.cast(x, dtype=dtype)

    def concatenate(self, x, axis=None):
        return self.backend.concat(x, axis=axis)

    def copy(self, x):
        return self.backend.identity(x)

    def real(self, x):
        return self.backend.math.real(x)

    def imag(self, x):
        return self.backend.math.imag(x)

    def conj(self, x):
        return self.backend.math.conj(x)

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
