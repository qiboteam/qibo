from abc import ABC, abstractmethod
from qibo.config import raise_error


class BaseBackend(ABC):

    def __init__(self):
        self.backend = None
        self.name = "base"

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
    def compile(self, func):
        raise_error(NotImplementedError)

    @abstractmethod
    def device(self, device_name):
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def oom_error(self):
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def optimization(self):
        """Module with attributes useful for optimization."""
        raise_error(NotImplementedError)
