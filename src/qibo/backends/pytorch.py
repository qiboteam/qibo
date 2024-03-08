"""PyTorch backend."""

from typing import Union

import numpy as np
import torch

from qibo import __version__
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend

torch_dtype_dict = {
    "int": torch.int32,
    "float": torch.float32,
    "complex": torch.complex64,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


class TorchMatrices(NumpyMatrices):
    """Matrix representation of every gate as a torch Tensor."""

    def __init__(self, dtype):
        super().__init__(dtype)
        self.torch = torch
        self.dtype = torch_dtype_dict[dtype]

    def _cast(self, x, dtype):
        return self.torch.tensor(x, dtype=dtype)

    def Unitary(self, u):
        return self._cast(u, dtype=self.dtype)


class PyTorchBackend(NumpyBackend):
    def __init__(self):
        super().__init__()

        self.name = "pytorch"
        self.versions = {
            "qibo": __version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        }

        self.matrices = TorchMatrices(self.dtype)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nthreads = 0
        self.np = torch
        self.dtype = torch_dtype_dict[self.dtype]
        self.tensor_types = (self.np.Tensor, np.ndarray)
        # Transpose function in Torch works in a different way than numpy
        self.np.transpose = torch.permute

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def set_seed(self, seed):
        self.np.manual_seed(seed)
        np.random.seed(seed)

    def cast(
        self,
        x: Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray]],
        dtype: Union[str, torch.dtype, np.dtype, type] = None,
        copy: bool = False,
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
        """
        if dtype is None:
            dtype = self.dtype
        elif isinstance(dtype, type):
            dtype = torch_dtype_dict[dtype.__name__]
        elif not isinstance(dtype, torch.dtype):
            dtype = torch_dtype_dict[str(dtype)]

        if isinstance(x, self.np.Tensor):
            x = x.to(dtype)
        elif isinstance(x, list) and all(isinstance(row, self.np.Tensor) for row in x):
            x = self.np.stack(x)
        else:
            x = self.np.tensor(x, dtype=dtype)

        if copy:
            return x.clone()

        return x

    def issparse(self, x):
        if isinstance(x, self.np.Tensor):
            return x.is_sparse

        return super().issparse(x)

    def to_numpy(self, x):
        if isinstance(x, list):
            return np.asarray([self.to_numpy(i) for i in x])

        if isinstance(x, self.np.Tensor):
            return x.numpy(force=True)

        return x

    def compile(self, func):
        return func

    def sample_shots(self, probabilities, nshots):
        return self.np.multinomial(
            self.cast(probabilities, dtype="float"), nshots, replacement=True
        )

    def samples_to_decimal(self, samples, nqubits):
        samples = self.cast(samples, dtype="int32")
        qrange = self.np.arange(nqubits - 1, -1, -1, dtype=torch.int32)
        qrange = (2**qrange).unsqueeze(1)
        return self.np.matmul(samples, qrange).squeeze(1)

    def samples_to_binary(self, samples, nqubits):
        samples = self.cast(samples, dtype="int32")
        qrange = self.np.arange(nqubits - 1, -1, -1, dtype=self.np.int32)
        samples = samples.int()
        samples = samples[:, None] >> qrange
        return samples % 2

    def calculate_norm(self, state, order=2):
        state = self.cast(state)
        return self.np.norm(state, p=order)

    def calculate_norm_density_matrix(self, state, order="nuc"):
        state = self.cast(state)
        if order == "nuc":
            return self.np.trace(state)
        return self.np.norm(state, p=order)

    def calculate_eigenvalues(self, matrix, k=6):
        return self.np.linalg.eigvalsh(matrix)  # pylint: disable=not-callable

    def calculate_eigenvectors(self, matrix, k=6):
        return self.np.linalg.eigh(matrix)  # pylint: disable=not-callable

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.issparse(matrix):
            return self.np.linalg.matrix_exp(  # pylint: disable=not-callable
                -1j * a * matrix
            )
        expd = self.np.diag(self.np.exp(-1j * a * eigenvalues))
        ud = self.np.conj(eigenvectors).T
        return self.np.matmul(eigenvectors, self.np.matmul(expd, ud))

    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        if self.issparse(matrix1) or self.issparse(matrix2):
            return self.np.sparse.mm(matrix1, matrix2)  # pylint: disable=E1102
        return self.np.matmul(matrix1, matrix2)

    def calculate_hamiltonian_state_product(self, matrix, state):
        return self.np.matmul(matrix, state)

    def calculate_overlap_density_matrix(self, state1, state2):
        return self.np.trace(
            self.np.matmul(self.np.conj(self.cast(state1)).T, self.cast(state2))
        )

    def _append_zeros(self, state, qubits, results):
        """Helper method for collapse."""
        for q, r in zip(qubits, results):
            state = self.np.unsqueeze(state, dim=q)
            if r:
                state = self.np.cat([self.np.zeros_like(state), state], dim=q)
            else:
                state = self.np.cat([state, self.np.zeros_like(state)], dim=q)
        return state

    def calculate_probabilities(self, state, qubits, nqubits):
        rtype = self.np.real(state).dtype
        unmeasured_qubits = tuple(i for i in range(nqubits) if i not in qubits)
        state = self.np.reshape(self.np.abs(state) ** 2, nqubits * (2,))
        if len(unmeasured_qubits) == 0:
            probs = self.cast(state, dtype=rtype)
        else:
            probs = self.np.sum(self.cast(state, dtype=rtype), axis=unmeasured_qubits)
        return self._order_probabilities(probs, qubits, nqubits).ravel()

    def _order_probabilities(self, probs, qubits, nqubits):
        """Arrange probabilities according to the given ``qubits`` ordering."""
        if probs.dim() == 0:
            return probs
        unmeasured, reduced = [], {}
        for i in range(nqubits):
            if i in qubits:
                reduced[i] = i - len(unmeasured)
            else:
                unmeasured.append(i)
        return self.np.transpose(probs, [reduced.get(i) for i in qubits])

    def test_regressions(self, name):
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
