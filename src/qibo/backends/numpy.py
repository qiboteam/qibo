from qibo.backends import abstract
from qibo.config import raise_error, log


class NumpyBackend(abstract.AbstractBackend):

    def __init__(self):
        super().__init__()
        import numpy as np
        self.backend = np
        self.name = "numpy"
        self.np = np

        from qibo.backends import matrices
        self.matrices = matrices.NumpyMatrices(self.dtypes('DTYPECPX'))
        self.numeric_types = (np.int, np.float, np.complex, np.int32,
                              np.int64, np.float32, np.float64,
                              np.complex64, np.complex128)
        self.tensor_types = (np.ndarray,)
        self.Tensor = np.ndarray
        self.random = np.random
        self.newaxis = np.newaxis
        self.oom_error = MemoryError
        self.optimization = None
        self.op = None

    def set_device(self, name):
        log.warning("Numpy does not support device placement. "
                    "Aborting device change.")

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

    def expand_dims(self, x, axis):
        return self.backend.expand_dims(x, axis)

    def copy(self, x):
        return self.backend.copy(x)

    def range(self, start, finish, step, dtype=None):
        if isinstance(dtype, str):
            dtype = self.dtypes(dtype)
        return self.backend.arange(start, finish, step, dtype=dtype)

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

    def round(self, x):
        return self.backend.round(x)

    def sign(self, x):
        return self.backend.sign(x)

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

    def expm(self, x):
        from scipy import linalg
        return linalg.expm(x)

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

    def array_equal(self, x, y):
        return self.np.array_equal(x, y)

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

    def gather_nd(self, x, indices):
        return x[tuple(indices)]

    def initial_state(self, nqubits, is_matrix=False):
        if is_matrix:
            state = self.zeros(2 * (2 ** nqubits,))
            state[0, 0] = 1
        else:
            state = self.zeros((2 ** nqubits,))
            state[0] = 1
        return state

    def transpose_state(self, pieces, state, nqubits, order): # pragma: no cover
        raise_error(NotImplementedError)

    def random_uniform(self, shape, dtype='DTYPE'):
        return self.backend.random.random(shape).astype(self.dtypes(dtype))

    def shuffle(self, x):
        self.random.shuffle(x)
        return x

    def sample_frequencies(self, probs, nshots):
        frequencies = self.round(nshots * probs)
        frequencies = self.cast(frequencies, dtype='DTYPEINT')

        num_ones = nshots - self.sum(frequencies)
        sign = self.sign(num_ones)
        num_ones = sign * num_ones
        num_zeros = tuple(probs.shape)[0] - num_ones

        ones = sign * self.ones(num_ones, dtype='DTYPEINT')
        zeros = self.zeros(num_zeros, dtype='DTYPEINT')
        fixer = self.concatenate([ones, zeros], axis=0)
        fixer = self.shuffle(fixer)
        return frequencies + fixer

    def sample_shots(self, probs, nshots):
        return self.np.random.choice(range(len(probs)), size=nshots, p=probs)

    def compile(self, func):
        return func

    def device(self, device_name):
        class DummyModule:

            def __enter__(self, *args):
                pass

            def __exit__(self, *args):
                pass

        return DummyModule()

    def set_seed(self, seed):
        self.backend.random.seed(seed)
