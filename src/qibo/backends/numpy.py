from functools import cache
from typing import Union

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.linalg import block_diag, fractional_matrix_power

from qibo import __version__
from qibo.backends import einsum_utils
from qibo.backends.abstract import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import log, raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


class NumpyBackend(Backend):
    def __init__(self):
        super().__init__()
        self.name = "numpy"
        self.matrices = NumpyMatrices(self.dtype)
        self.tensor_types = np.ndarray
        self.versions = {"qibo": __version__, "numpy": np.__version__}
        self.numeric_types = (
            int,
            float,
            complex,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )

    @property
    def qubits(self):
        return None

    @property
    def connectivity(self):
        return None

    @property
    def natives(self):
        return None

    def set_precision(self, precision):
        if precision != self.precision:
            if precision == "single":
                self.precision = precision
                self.dtype = "complex64"
            elif precision == "double":
                self.precision = precision
                self.dtype = "complex128"
            else:
                raise_error(ValueError, f"Unknown precision {precision}.")
            if self.matrices:
                self.matrices = self.matrices.__class__(self.dtype)

    def set_device(self, device):
        if device != "/CPU:0":
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )

    def set_threads(self, nthreads):
        if nthreads > 1:
            raise_error(ValueError, "numpy does not support more than one thread.")

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        if isinstance(x, self.tensor_types):
            return x.astype(dtype, copy=copy)
        elif self.is_sparse(x):
            return x.astype(dtype, copy=copy)
        return np.array(x, dtype=dtype, copy=copy)

    def is_sparse(self, x):
        from scipy import sparse

        return sparse.issparse(x)

    def to_numpy(self, x):
        if self.is_sparse(x):
            return x.toarray()
        return x

    def compile(self, func):
        return func

    def matrix_fused(self, fgate):
        rank = len(fgate.target_qubits)
        matrix = sparse.eye(2**rank)

        for gate in fgate.gates:
            # transfer gate matrix to numpy as it is more efficient for
            # small tensor calculations
            # explicit to_numpy see https://github.com/qiboteam/qibo/issues/928
            gmatrix = self.to_numpy(gate.matrix(self))
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = block_diag(
                    self.eye(2 ** len(gate.qubits) - len(gmatrix)), gmatrix
                )
            # Kronecker product with identity is needed to make the
            # original matrix have shape (2**rank x 2**rank)
            eye = self.eye(2 ** (rank - len(gate.qubits)))
            gmatrix = self.kron(gmatrix, eye)
            # Transpose the new matrix indices so that it targets the
            # target qubits of the original gate
            original_shape = gmatrix.shape
            gmatrix = self.reshape(gmatrix, 2 * rank * (2,))
            qubits = list(gate.qubits)
            indices = qubits + [q for q in fgate.target_qubits if q not in qubits]
            indices = self.argsort(indices)
            transpose_indices = list(indices)
            transpose_indices.extend(indices + rank)
            gmatrix = self.transpose(gmatrix, transpose_indices)
            gmatrix = self.reshape(gmatrix, original_shape)
            # fuse the individual gate matrix to the total ``FusedGate`` matrix
            # we are using sparse matrices to improve perfomances
            matrix = sparse.csr_matrix(gmatrix).dot(matrix)

        return self.cast(matrix.toarray())

    def execute_distributed_circuit(self, circuit, initial_state=None, nshots=None):
        raise_error(
            NotImplementedError, f"{self} does not support distributed execution."
        )

    def calculate_eigenvectors(self, matrix, k: int = 6, hermitian: bool = True):
        if self.is_sparse(matrix):
            if k < matrix.shape[0]:
                from scipy.sparse.linalg import eigsh

                return eigsh(matrix, k=k, which="SA")
            else:  # pragma: no cover
                matrix = self.to_numpy(matrix)
        if hermitian:
            return self.eigh(matrix)
        return self.eig(matrix)

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.is_sparse(matrix):
            if self.is_sparse(matrix):
                from scipy.sparse.linalg import expm
            else:
                from scipy.linalg import expm
            return expm(-1j * a * matrix)
        expd = self.diag(self.exp(-1j * a * eigenvalues))
        ud = self.transpose(self.conj(eigenvectors))
        return self.matmul(eigenvectors, self.matmul(expd, ud))

    def calculate_matrix_power(
        self,
        matrix,
        power: Union[float, int],
        precision_singularity: float = 1e-14,
    ):
        if not isinstance(power, (float, int)):
            raise_error(
                TypeError,
                f"``power`` must be either float or int, but it is type {type(power)}.",
            )

        if power < 0.0:
            # negative powers of singular matrices via SVD
            determinant = self.det(matrix)
            if abs(determinant) < precision_singularity:
                return _calculate_negative_power_singular_matrix(
                    matrix, power, precision_singularity, self
                )

        return fractional_matrix_power(matrix, power)

    def calculate_jacobian_matrix(
        self, circuit, parameters=None, initial_state=None, return_complex: bool = True
    ):
        raise_error(
            NotImplementedError,
            "This method is only implemented in backends that allow automatic differentiation, "
            + "e.g. ``PytorchBackend`` and ``TensorflowBackend``.",
        )

    # core methods
    # ^^^^^^^^^^^^

    # array creation and manipulation
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    @staticmethod
    def eye(N, **kwargs):
        """Numpy-like eye: https://numpy.org/devdocs/reference/generated/numpy.eye.html"""
        return np.eye(N, **kwargs)

    @staticmethod
    def zeros(shape, **kwargs):
        """Numpy-like zeros: https://numpy.org/devdocs/reference/generated/numpy.zeros.html"""
        return np.zeros(shape, **kwargs)

    @staticmethod
    def ones(shape, **kwargs):
        """Numpy-like ones: https://numpy.org/devdocs/reference/generated/numpy.ones.html"""
        return np.ones(shape, **kwargs)

    @staticmethod
    def arange(*args, **kwargs):
        """Numpy-like arange: https://numpy.org/devdocs/reference/generated/numpy.arange.html"""
        return np.arange(*args, **kwargs)

    @staticmethod
    def copy(a, **kwargs):
        """Numpy-like copy: https://numpy.org/devdocs/reference/generated/numpy.copy.html"""
        return np.copy(a, **kwargs)

    @staticmethod
    def reshape(a, shape, **kwargs):
        """Numpy-like reshape: https://numpy.org/devdocs/reference/generated/numpy.reshape.html"""
        return np.reshape(a, shape, **kwargs)

    @staticmethod
    def ravel(a, **kwargs):
        """Numpy-like reshape: https://numpy.org/devdocs/reference/generated/numpy.reshape.html"""
        return np.ravel(a, **kwargs)

    @staticmethod
    def transpose(a, axes=None):
        """Numpy-like transpose: https://numpy.org/devdocs/reference/generated/numpy.transpose.html"""
        return np.transpose(a, axes=axes)

    @staticmethod
    def concatenate(*a, **kwargs):
        """Numpy-like concatenate: https://numpy.org/devdocs/reference/generated/numpy.concatenate.html"""
        return np.concatenate(*a, **kwargs)

    @staticmethod
    def expand_dims(a, axis):
        """Numpy-like expand_dims: https://numpy.org/devdocs/reference/generated/numpy.expand_dims.html"""
        return np.expand_dims(a, axis)

    @staticmethod
    def squeeze(a, **kwargs):
        """Numpy-like squeeze: https://numpy.org/devdocs/reference/generated/numpy.squeeze.html"""
        return np.squeeze(a, **kwargs)

    @staticmethod
    def stack(arrays, axis=0, **kwargs):
        """Numpy-like stack: https://numpy.org/devdocs/reference/generated/numpy.stack.html"""
        return np.stack(arrays, axis=axis, **kwargs)

    @staticmethod
    def vstack(tup, **kwargs):
        """Numpy-like vstack: https://numpy.org/devdocs/reference/generated/numpy.vstack.html"""
        return np.vstack(tup, **kwargs)

    @staticmethod
    def unique(ar, **kwargs):
        """Numpy-like unique: https://numpy.org/devdocs/reference/generated/numpy.unique.html"""
        return np.unique(ar, **kwargs)

    @staticmethod
    def where(condition, *args):
        """Numpy-like where: https://numpy.org/doc/stable/reference/generated/numpy.where.html"""
        return np.where(condition, *args)

    @staticmethod
    def flip(m, axis=None):
        """Numpy-like flip: https://numpy.org/doc/stable/reference/generated/numpy.flip.html"""
        return np.flip(m, axis=axis)

    @staticmethod
    def swapaxes(a, axis1, axis2):
        """Numpy-like swapaxes: https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html"""
        return np.swapaxes(a, axis1, axis2)

    @staticmethod
    def diagonal(a, offset=0, axis1=0, axis2=1):
        """Numpy-like diagonal: https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html"""
        return np.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)

    @staticmethod
    def nonzero(a):
        """Numpy-like nonzero: https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html"""
        return np.nonzero(a)

    @staticmethod
    def sign(a, **kwargs):
        """Numpy-like element-wise sign function: https://numpy.org/doc/stable/reference/generated/numpy.sign.html"""
        return np.sign(a, **kwargs)

    @staticmethod
    def delete(a, obj, axis=None):
        """Numpy-like element-wise delete function: https://numpy.org/doc/stable/reference/generated/numpy.delete.html"""
        return np.delete(a, obj=obj, axis=axis)

    # linear algebra
    # ^^^^^^^^^^^^^^

    @staticmethod
    def einsum(subscripts, *operands, **kwargs):
        """Numpy-like einsum: https://numpy.org/devdocs/reference/generated/numpy.einsum.html"""
        return np.einsum(subscripts, *operands, **kwargs)

    @staticmethod
    def matmul(a, b, **kwargs):
        """Numpy-like matmul: https://numpy.org/devdocs/reference/generated/numpy.matmul.html"""
        return np.matmul(a, b, **kwargs)

    @staticmethod
    def multiply(a, b, **kwargs):
        """Numpy-like multiply: https://numpy.org/doc/stable/reference/generated/numpy.multiply.html"""
        return np.multiply(a, b, **kwargs)

    @staticmethod
    def prod(a, **kwargs):
        """Numpy-like prod: https://numpy.org/doc/stable/reference/generated/numpy.prod.html"""
        return np.prod(a, **kwargs)

    @staticmethod
    def tensordot(a, b, axes=2):
        """Numpy-like tensordot: https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html"""
        return np.tensordot(a, b, axes=axes)

    @staticmethod
    def kron(a, b):
        """Numpy-like kron: https://numpy.org/doc/stable/reference/generated/numpy.kron.html"""
        return np.kron(a, b)

    @staticmethod
    def outer(a, b, out=None):
        """Numpy-like outer: https://numpy.org/doc/stable/reference/generated/numpy.outer.html"""
        return np.outer(a, b, out=out)

    @staticmethod
    def diag(a, k=0):
        """Numpy-like diag: https://numpy.org/devdocs/reference/generated/numpy.diag.html"""
        return np.diag(a, k=k)

    @staticmethod
    def trace(a, **kwargs):
        """Numpy-like trace: https://numpy.org/devdocs/reference/generated/numpy.trace.html"""
        return np.trace(a, **kwargs)

    @staticmethod
    def linalg_svd(a, **kwargs):
        """Numpy-like linalg.svd: https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html"""
        return np.linalg.svd(a, **kwargs)

    @staticmethod
    def linalg_norm(a, ord=None, axis=None, keepdims=False):
        """Numpy-like linalg.norm: https://numpy.org/devdocs/reference/generated/numpy.linalg.norm.html"""
        return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)

    @staticmethod
    def det(a):
        """Numpy-like matrix determinant: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html"""
        return np.linalg.det(a)

    @staticmethod
    def qr(a, **kwargs):
        """Numpy linear algebra QR decomposition: https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html"""
        return np.linalg.qr(a, **kwargs)

    @staticmethod
    def inverse(a):
        """Numpy linear algebra inverse: https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html"""
        return np.linalg.inv(a)

    @staticmethod
    def eigvalsh(a, **kwargs):
        """Numpy-like eigvalsh: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html"""
        return np.linalg.eigvalsh(a, **kwargs)

    @staticmethod
    def eigvals(a):
        """Eigenvalues of a matrix: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html"""
        return np.linalg.eigvals(a)

    @staticmethod
    def eigh(a, **kwargs):
        """Numpy-like eigvals: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html"""
        return np.linalg.eigh(a, **kwargs)

    @staticmethod
    def eig(a):
        """Numpy-like eig: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html"""
        return np.linalg.eig(a)

    @staticmethod
    def expm(a):
        """Scipy-like expm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html"""
        return sp.linalg.expm(a)

    # randomization
    # ^^^^^^^^^^^^^

    @staticmethod
    def random_choice(a, size=None, replace=True, p=None):
        """
        Numpy-like random.choice: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        """
        return np.random.choice(a, size=size, replace=replace, p=p)

    @staticmethod
    def seed(seed=None):
        """
        Numpy-like random seed: https://numpy.org/devdocs/reference/random/generated/numpy.random.seed.html
        """
        return np.random.seed(seed)

    @staticmethod
    def permutation(x):
        """
        Numpy-like random permutation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        """
        return np.random.permutation(x)

    @staticmethod
    def multinomial(n, pvals, size=None):
        """
        Numpy-like multinomial: https://numpy.org/doc/2.0/reference/random/generated/numpy.random.multinomial.html
        """
        return np.random.multinomial(n, pvals, size=size)

    @staticmethod
    def default_rng(seed=None):
        """
        Numpy-like random default_rng: https://numpy.org/doc/stable/reference/random/generator.html
        """
        return np.random.default_rng(seed)

    @staticmethod
    def rand(*args):
        """
        Numpy-like random rand: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        """
        return np.random.rand(*args)

    # logical operations
    # ^^^^^^^^^^^^^^^^^^

    @staticmethod
    def less(x1, x2, **kwargs):
        """
        Numpy-like less: https://numpy.org/doc/stable/reference/generated/numpy.less.html
        """
        return np.less(x1, x2, **kwargs)

    @staticmethod
    def any(a, **kwargs):
        """
        Numpy-like any: https://numpy.org/doc/stable/reference/generated/numpy.any.html
        """
        return np.any(a, **kwargs)

    @staticmethod
    def allclose(a, b, **kwargs):
        """
        Numpy-like allclose: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        """
        return np.allclose(a, b, **kwargs)

    @staticmethod
    def right_shift(x1, x2, **kwargs):
        """
        Numpy-like element-wise right shift: https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html
        """
        return np.right_shift(x1, x2, **kwargs)

    # mathematical operations
    # ^^^^^^^^^^^^^^^^^^^^^^^

    @staticmethod
    def sum(a, axis=None, **kwargs):
        """Numpy-like sum: https://numpy.org/devdocs/reference/generated/numpy.sum.html"""
        return np.sum(a, axis=axis, **kwargs)

    @staticmethod
    def conj(a, **kwargs):
        """Numpy-like conj: https://numpy.org/devdocs/reference/generated/numpy.conj.html"""
        return np.conj(a, **kwargs)

    @staticmethod
    def exp(a, **kwargs):
        """Numpy-like exp: https://numpy.org/devdocs/reference/generated/numpy.exp.html"""
        return np.exp(a, **kwargs)

    @staticmethod
    def log(a, **kwargs):
        """Numpy-like log: https://numpy.org/doc/stable/reference/generated/numpy.log.html"""
        return np.log(a, **kwargs)

    @staticmethod
    def log2(a, **kwargs):
        """Numpy-like log2: https://numpy.org/doc/stable/reference/generated/numpy.log2.html"""
        return np.log2(a, **kwargs)

    @staticmethod
    def real(a):
        """Numpy-like real: https://numpy.org/devdocs/reference/generated/numpy.real.html"""
        return np.real(a)

    @staticmethod
    def imag(a):
        """Numpy-like imag: https://numpy.org/doc/stable/reference/generated/numpy.imag.html"""
        return np.imag(a)

    @staticmethod
    def abs(a, **kwargs):
        """Numpy-like abs: https://numpy.org/devdocs/reference/generated/numpy.abs.html"""
        return np.abs(a, **kwargs)

    @staticmethod
    def pow(a, b, **kwargs):
        """Numpy-like element-wise power: https://numpy.org/doc/stable/reference/generated/numpy.power.html"""
        return np.power(a, b, **kwargs)

    @staticmethod
    def square(a, **kwargs):
        """Numpy-like element-wise square: https://numpy.org/doc/stable/reference/generated/numpy.square.html"""
        return np.square(a, **kwargs)

    @staticmethod
    def sqrt(a, **kwargs):
        """Numpy-like sqrt: https://numpy.org/devdocs/reference/generated/numpy.sqrt.html"""
        return np.sqrt(a, **kwargs)

    @staticmethod
    def mean(a, axis=None, **kwargs):
        """Numpy-like mean: https://numpy.org/doc/stable/reference/generated/numpy.mean.html"""
        return np.mean(a, axis=axis, **kwargs)

    @staticmethod
    def std(a, **kwargs):
        """Numpy-like standard deviation: https://numpy.org/doc/stable/reference/generated/numpy.std.html"""
        return np.std(a, **kwargs)

    @staticmethod
    def cos(x, **kwargs):
        """Numpy-like cos: https://numpy.org/devdocs/reference/generated/numpy.cos.html"""
        return np.cos(x, **kwargs)

    @staticmethod
    def sin(x, **kwargs):
        """Numpy-like sin: https://numpy.org/devdocs/reference/generated/numpy.sin.html"""
        return np.sin(x, **kwargs)

    @staticmethod
    def arccos(x, **kwargs):
        """Numpy-like arccos: https://numpy.org/doc/stable/reference/generated/numpy.arccos.html"""
        return np.arccos(x, **kwargs)

    @staticmethod
    def arctan2(y, x, **kwargs):
        """Numpy-like arctan2: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html"""
        return np.arctan2(y, x, **kwargs)

    @staticmethod
    def angle(z, **kwargs):
        """Numpy-like angle: https://numpy.org/doc/stable/reference/generated/numpy.angle.html"""
        return np.angle(z, **kwargs)

    @staticmethod
    def mod(x, y, **kwargs):
        """Numpy-like element-wise modulus: https://numpy.org/doc/stable/reference/generated/numpy.mod.html"""
        return np.mod(x, y, **kwargs)

    @staticmethod
    def round(a, decimals=0, out=None):
        """Numpy-like element-wise round: https://numpy.org/doc/stable/reference/generated/numpy.round.html"""
        return np.round(a, decimals=decimals, out=out)

    # misc
    # ^^^^

    @staticmethod
    def sort(a, **kwargs):
        """Numpy-like sort: https://numpy.org/doc/stable/reference/generated/numpy.sort.html"""
        return np.sort(a, **kwargs)

    @staticmethod
    def argsort(a, **kwargs):
        """Numpy-like argsort: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html"""
        return np.argsort(a, **kwargs)

    @staticmethod
    def count_nonzero(a, **kwargs):
        """Numpy-like count_nonzero: https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html"""
        return np.count_nonzero(a, **kwargs)

    @staticmethod
    def finfo(dtype):
        """Numpy-like finfo: https://numpy.org/doc/stable/reference/generated/numpy.finfo.html"""
        return np.finfo(dtype)

    @staticmethod
    def device():
        """Computation device, e.g. CPU, GPU, ..."""
        return "CPU"

    @staticmethod
    def __version__():
        """Version of the backend engine."""
        return np.__version__

    @staticmethod
    @cache
    def get_dtype(type_name: str):
        """Backend engine dtype"""
        return getattr(np, type_name)

    @staticmethod
    def assert_allclose(a, b, **kwargs):
        """
        Numpy-like allclose: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        """
        return np.testing.assert_allclose(a, b, **kwargs)

    # Optimization
    # ^^^^^^^^^^^^^

    @staticmethod
    def jacobian(*args, **kwargs):
        """Compute the Jacobian matrix"""
        raise NotImplementedError


def _calculate_negative_power_singular_matrix(
    matrix, power: Union[float, int], precision_singularity: float, backend
):
    """Calculate negative power of singular matrix."""
    U, S, Vh = backend.calculate_singular_value_decomposition(matrix)
    # cast needed because of different dtypes in `torch`
    S = backend.cast(S)
    S_inv = backend.where(backend.abs(S) < precision_singularity, 0.0, S**power)

    return backend.inverse(Vh) @ backend.diag(S_inv) @ backend.inverse(U)
