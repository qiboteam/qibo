from abc import ABC, abstractmethod
from qibo.config import raise_error


class BaseBackend(ABC):

    def __init__(self):
        self.backend = None
        self.name = "base"

        self.NUMERIC_TYPES = None
        # TODO: Consider removing ``ARRAY_TYPES``
        self.ARRAY_TYPES = None
        self.TENSOR_TYPES = None
        self.DTYPES = {'string': "double"}

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
    def copy(self, x):
        """Creates a copy of the tensor in memory."""
        raise_error(NotImplementedError)

    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Creates tensor of zeros with the given shape and dtype."""
        raise_error(NotImplementedError)

    @abstractmethod
    def conj(self, x):
        """Elementwise complex conjugate of a tensor."""
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


class NumpyBackend(BaseBackend):

    def __init__(self):
        import numpy as np
        self.backend = np
        self.name = "numpy"

        self.NUMERIC_TYPES = (np.int, np.float, np.complex,
                              np.int32, np.int64, np.float32,
                              np.float64, np.complex64, np.complex128)
        # TODO: Consider removing ``ARRAY_TYPES``
        self.ARRAY_TYPES = (np.ndarray,)
        self.TENSOR_TYPES = (np.ndarray,)
        self.DTYPES = {
            'STRING': 'double',
            'DTYPEINT': np.int64,
            'DTYPE': np.float64,
            'DTYPECPX': np.complex128
            }

    @property
    def tensortype(self):
        return self.backend.ndarray

    def cast(self, x, dtype=None):
        if dtype is None:
            dtype = self.DTYPES.get('DTYPECPX')
        return x.astype(dtype)

    def reshape(self, x, shape):
        return self.backend.reshape(x, shape)

    def stack(self, x, axis=None):
        return self.backend.stack(x, axis=axis)

    def copy(self, x):
        return self.backend.copy(x)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.DTYPES.get('DTYPECPX')
        return self.backend.zeros(shape, dtype=dtype)

    def conj(self, x):
        return self.backend.conj(x)

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


class TensorflowBackend(NumpyBackend):

    def __init__(self):
        import numpy as np
        import tensorflow as tf
        self.backend = tf
        self.name = "tensorflow"
        self.np = np

        self.NUMERIC_TYPES = (np.int, np.float, np.complex,
                              np.int32, np.int64, np.float32,
                              np.float64, np.complex64, np.complex128)
        self.ARRAY_TYPES = (np.ndarray, tf.Tensor)
        self.TENSOR_TYPES = (np.ndarray, tf.Tensor, tf.Variable)
        self.DTYPES = {
            'STRING': 'double',
            'DTYPEINT': tf.int64,
            'DTYPE': tf.float64,
            'DTYPECPX': tf.complex128
            }

    @property
    def tensortype(self):
        return self.backend.Tensor

    def cast(self, x, dtype=None):
        if dtype is None:
            dtype = self.DTYPES.get('DTYPECPX')
        if isinstance(x, self.np.ndarray):
            dtypestr = dtype.__repr__().split(".")[1]
            x = x.astype(getattr(self.np, dtypestr))
        return self.backend.cast(x, dtype=dtype)

    def copy(self, x):
        return self.backend.identity(x)

    def conj(self, x):
        return self.backend.math.conj(x)

    def sum(self, x, axis=None):
        return self.backend.reduce_sum(x, axis=axis)

    def outer(self, x, y):
        return self.tensordot(x, y, axes=0)

    def compile(self, func):
        return self.backend.function(func)

    def device(self, device_name):
        return self.backend.device(device_name)


function_names = [m for m in dir(BaseBackend) if m[:2] != "__"]

factory = {
    'numpy': NumpyBackend,
    'tensorflow': TensorflowBackend
}
