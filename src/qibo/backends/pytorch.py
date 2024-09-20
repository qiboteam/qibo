"""PyTorch backend."""

import numpy as np

from qibo import __version__
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend


class TorchMatrices(NumpyMatrices):
    """Matrix representation of every gate as a torch Tensor.

    Args:
        dtype (torch.dtype): Data type of the matrices.
        requires_grad (bool): If ``True`` the matrices require gradient.
    """

    def __init__(self, dtype, requires_grad):
        import torch  # pylint: disable=import-outside-toplevel

        super().__init__(dtype)
        self.np = torch
        self.dtype = dtype
        self.requires_grad = requires_grad

    def _cast(self, x, dtype):
        flattened = [item for sublist in x for item in sublist]
        tensor_list = [self.np.as_tensor(i, dtype=dtype) for i in flattened]
        return self.np.stack(tensor_list).reshape(len(x), len(x))

    def _cast_parameter(self, x):
        return self.np.tensor(x, dtype=self.dtype, requires_grad=self.requires_grad)

    def Unitary(self, u):
        return self._cast(u, dtype=self.dtype)


class PyTorchBackend(NumpyBackend):
    def __init__(self):
        super().__init__()
        import torch  # pylint: disable=import-outside-toplevel

        # Global variable to enable or disable gradient calculation
        self.gradients = True

        self.np = torch

        self.name = "pytorch"
        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "torch": self.np.__version__,
        }

        self.dtype = self._torch_dtype(self.dtype)
        self.matrices = TorchMatrices(self.dtype, requires_grad=self.gradients)
        self.device = self.np.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nthreads = 0
        self.tensor_types = (self.np.Tensor, np.ndarray)

        # These functions in Torch works in a different way than numpy or have different names
        self.np.transpose = self.np.permute
        self.np.copy = self.np.clone
        self.np.expand_dims = self.np.unsqueeze
        self.np.mod = self.np.remainder
        self.np.right_shift = self.np.bitwise_right_shift
        self.np.sign = self.np.sgn
        self.np.flatnonzero = lambda x: self.np.nonzero(x).flatten()

    def requires_grad(self, requires_grad):
        """Enable or disable gradient calculation."""
        self.gradients = requires_grad
        self.matrices.requires_grad = requires_grad

    def _torch_dtype(self, dtype):
        if dtype == "float":
            dtype += "32"
        return getattr(self.np, dtype)

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def cast(
        self,
        x,
        dtype=None,
        copy: bool = False,
        requires_grad: bool = None,
    ):
        """Casts input as a Torch tensor of the specified dtype.

        This method supports casting of single tensors or lists of tensors
        as for the :class:`qibo.backends.PyTorchBackend`.

        Args:
            x (Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray], int, float, complex]):
                Input to be casted.
            dtype (Union[str, torch.dtype, np.dtype, type]): Target data type.
                If ``None``, the default dtype of the backend is used.
                Defaults to ``None``.
            copy (bool, optional): If ``True``, the input tensor is copied before casting.
                Defaults to ``False``.
            requires_grad (bool, optional): If ``True``, the input tensor requires gradient.
                If ``False``, the input tensor does not require gradient.
                If ``None``, the default gradient setting of the backend is used.
        """
        if requires_grad is None:
            requires_grad = self.gradients

        if dtype is None:
            dtype = self.dtype
        elif isinstance(dtype, type):
            dtype = self._torch_dtype(dtype.__name__)
        elif not isinstance(dtype, self.np.dtype):
            dtype = self._torch_dtype(str(dtype))

        # check if dtype is an integer to remove gradients
        if dtype in [self.np.int32, self.np.int64, self.np.int8, self.np.int16]:
            requires_grad = False

        if isinstance(x, self.np.Tensor):
            x = x.to(dtype)
        elif (
            isinstance(x, list)
            and len(x) > 0
            and all(isinstance(row, self.np.Tensor) for row in x)
        ):
            x = self.np.stack(x)
        else:
            x = self.np.tensor(x, dtype=dtype, requires_grad=requires_grad)

        if copy:
            return x.clone()

        return x

    def is_sparse(self, x):
        if isinstance(x, self.np.Tensor):
            return x.is_sparse

        return super().is_sparse(x)

    def to_numpy(self, x):
        if isinstance(x, list):
            return np.asarray([self.to_numpy(i) for i in x])

        if isinstance(x, self.np.Tensor):
            return x.numpy(force=True)

        return x

    def _order_probabilities(self, probs, qubits, nqubits):
        """Arrange probabilities according to the given ``qubits`` ordering."""
        if probs.dim() == 0:  # pragma: no cover
            return probs
        unmeasured, reduced = [], {}
        for i in range(nqubits):
            if i in qubits:
                reduced[i] = i - len(unmeasured)
            else:
                unmeasured.append(i)
        return self.np.transpose(probs, [reduced.get(i) for i in qubits])

    def calculate_probabilities(self, state, qubits, nqubits):
        rtype = self.np.real(state).dtype
        unmeasured_qubits = tuple(i for i in range(nqubits) if i not in qubits)
        state = self.np.reshape(self.np.abs(state) ** 2, nqubits * (2,))
        if len(unmeasured_qubits) == 0:
            probs = self.cast(state, dtype=rtype)
        else:
            probs = self.np.sum(self.cast(state, dtype=rtype), axis=unmeasured_qubits)
        return self._order_probabilities(probs, qubits, nqubits).ravel()

    def set_seed(self, seed):
        self.np.manual_seed(seed)
        np.random.seed(seed)

    def sample_shots(self, probabilities, nshots):
        return self.np.multinomial(
            self.cast(probabilities, dtype="float"), nshots, replacement=True
        )

    def calculate_eigenvalues(self, matrix, k: int = 6, hermitian: bool = True):
        if hermitian:
            return self.np.linalg.eigvalsh(matrix)  # pylint: disable=not-callable
        return self.np.linalg.eigvals(matrix)  # pylint: disable=not-callable

    def calculate_eigenvectors(self, matrix, k: int = 6, hermitian: int = True):
        if hermitian:
            return self.np.linalg.eigh(matrix)  # pylint: disable=not-callable
        return self.np.linalg.eig(matrix)  # pylint: disable=not-callable

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.is_sparse(matrix):
            return self.np.linalg.matrix_exp(  # pylint: disable=not-callable
                -1j * a * matrix
            )
        expd = self.np.diag(self.np.exp(-1j * a * eigenvalues))
        ud = self.np.conj(eigenvectors).T
        return self.np.matmul(eigenvectors, self.np.matmul(expd, ud))

    def calculate_matrix_power(self, matrix, power):
        copied = self.to_numpy(self.np.copy(matrix))
        copied = super().calculate_matrix_power(copied, power)
        return self.cast(copied, dtype=copied.dtype)

    def _test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            ]

        if name == "test_probabilistic_measurement":
            if self.device == "cuda":  # pragma: no cover
                return {0: 273, 1: 233, 2: 242, 3: 252}
            return {1: 270, 2: 248, 3: 244, 0: 238}

        if name == "test_unbalanced_probabilistic_measurement":
            if self.device == "cuda":  # pragma: no cover
                return {0: 196, 1: 153, 2: 156, 3: 495}
            return {3: 492, 2: 176, 0: 168, 1: 164}

        if name == "test_post_measurement_bitflips_on_circuit":
            return [
                {5: 30},
                {5: 17, 4: 5, 7: 4, 1: 2, 6: 2},
                {4: 9, 2: 5, 5: 5, 3: 4, 6: 4, 0: 1, 1: 1, 7: 1},
            ]
