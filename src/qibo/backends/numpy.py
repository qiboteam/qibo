from qibo.backends import Backend
from qibo.backends import base
from qibo.config import raise_error


class NumpyBackend(Backend, base.BaseBackend):

    def __init__(self):
        Backend.__init__(self)
        base.BaseBackend.__init__(self)
        import numpy as np
        self.backend = np
        self.name = "numpy"
        self.np = np
        from qibo.backends import matrices
        self.matrices = matrices.NumpyMatrices(self.dtypes('DTYPECPX'))

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
