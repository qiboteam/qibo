import collections

import numpy as np
import torch

from qibo import __version__
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class TorchMatrices(NumpyMatrices):
    # Redefine parametrized gate matrices for backpropagation to work

    def __init__(self, dtype):
        self.np = np
        super().__init__(dtype)

    def RX(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        isin = -1j * self.np.sin(theta / 2.0)
        return torch.tensor([[cos, isin], [isin, cos]], dtype=self.dtype)

    def RY(self, theta):
        cos = self.np.cos(theta / 2.0) + 0j
        sin = self.np.sin(theta / 2.0) + 0j
        return torch.tensor([[cos, -sin], [sin, cos]], dtype=self.dtype)

    def RZ(self, theta):
        phase = self.np.exp(0.5j * theta)
        return torch.tensor([[self.np.conj(phase), 0], [0, phase]], dtype=self.dtype)

    def U1(self, theta):
        phase = self.np.exp(1j * theta)
        return torch.tensor([[1, 0], [0, phase]], dtype=self.dtype)

    def U2(self, phi, lam):
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return torch.tensor(
            [[self.np.conj(eplus), -self.np.conj(eminus)], [eminus, eplus]],
            dtype=self.dtype,
        ) / self.np.sqrt(2)

    def U3(self, theta, phi, lam):
        cost = self.np.cos(theta / 2)
        sint = self.np.sin(theta / 2)
        eplus = self.np.exp(1j * (phi + lam) / 2.0)
        eminus = self.np.exp(1j * (phi - lam) / 2.0)
        return torch.tensor(
            [
                [self.np.conj(eplus) * cost, -self.np.conj(eminus) * sint],
                [eminus * sint, eplus * cost],
            ],
            dtype=self.dtype,
        )

    def CRX(self, theta):
        r = self.RX(theta)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CRY(self, theta):
        r = self.RY(theta)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CRZ(self, theta):
        r = self.RZ(theta)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CU1(self, theta):
        r = self.U1(theta)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CU2(self, phi, lam):
        r = self.U2(phi, lam)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def CU3(self, theta, phi, lam):
        r = self.U3(theta, phi, lam)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, r[0, 0], r[0, 1]],
                [0, 0, r[1, 0], r[1, 1]],
            ],
            dtype=self.dtype,
        )

    def fSim(self, theta, phi):
        cost = self.np.cos(theta) + 0j
        isint = -1j * self.np.sin(theta)
        phase = self.np.exp(-1j * phi)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cost, isint, 0],
                [0, isint, cost, 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def GeneralizedfSim(self, u, phi):
        phase = self.np.exp(-1j * phi)
        return torch.tensor(
            [
                [1, 0, 0, 0],
                [0, u[0, 0], u[0, 1], 0],
                [0, u[1, 0], u[1, 1], 0],
                [0, 0, 0, phase],
            ],
            dtype=self.dtype,
        )

    def Unitary(self, u):
        return torch.tensor(u, dtype=self.dtype)


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
        self.tensor_types = (np.ndarray, torch.Tensor)

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def cast(self, x, dtype=None, copy=False):
        if dtype is None:
            dtype = self.dtype
        x = torch.tensor(x, dtype=dtype)
        if copy:
            return x.clone()
        return x

    def issparse(self, x):
        return x.is_sparse

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def compile(self, func):
        return torch.jit.script(func)

    def zero_state(self, nqubits):
        state = torch.zeros(2**nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits):
        state = torch.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state[0, 0] = 1
        return state

    def matrix(self, gate):
        npmatrix = super().matrix(gate)
        return torch.tensor(npmatrix, dtype=self.dtype)

    def matrix_parametrized(self, gate):
        npmatrix = super().matrix_parametrized(gate)
        return torch.tensor(npmatrix, dtype=self.dtype)

    def matrix_fused(self, gate):
        npmatrix = super().matrix_fused(gate)
        return torch.tensor(npmatrix, dtype=self.dtype)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        if initial_state is not None:
            initial_state = initial_state.to(self.device)
        return super().execute_circuit(circuit, initial_state, nshots)

    def execute_circuit_repeated(self, circuit, nshots, initial_state=None):
        if initial_state is not None:
            initial_state = initial_state.to(self.device)
        return super().execute_circuit_repeated(circuit, nshots, initial_state)

    def sample_shots(self, probabilities, nshots):
        return torch.multinomial(probabilities, nshots)

    def samples_to_binary(self, samples, nqubits):
        qrange = torch.arange(nqubits - 1, -1, -1, dtype=torch.int32)
        samples = samples.int()
        samples = samples[:, None] >> qrange
        return samples % 2

    def calculate_frequencies(self, samples):
        res, counts = torch.unique(samples, return_counts=True)
        res, counts = res.tolist(), counts.tolist()
        return collections.Counter({k: v for k, v in zip(res, counts)})

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        unique_samples, counts = torch.unique(samples, return_counts=True)
        frequencies.index_add_(0, unique_samples, counts)
        return frequencies

    def calculate_norm(self, state, order=2):
        state = self.cast(state)
        return torch.norm(state, p=order)

    def calculate_norm_density_matrix(self, state, order="nuc"):
        state = self.cast(state)
        if order == "nuc":
            return np.trace(state)
        return torch.norm(state, p=order)

    def calculate_eigenvalues(self, matrix):
        return torch.linalg.eigvalsh(matrix)  # pylint: disable=not-callable

    def calculate_eigenvectors(self, matrix):
        return torch.linalg.eigh(matrix)  # pylint: disable=not-callable

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.issparse(matrix):
            return torch.linalg.matrix_exp(  # pylint: disable=not-callable
                -1j * a * matrix
            )
        else:
            return super().calculate_matrix_exp(a, matrix, eigenvectors, eigenvalues)

    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        if self.issparse(matrix1) or self.issparse(matrix2):
            return torch.sparse.mm(matrix1, matrix2)  # pylint: disable=not-callable
        return super().calculate_hamiltonian_matrix_product(matrix1, matrix2)

    def calculate_hamiltonian_state_product(self, matrix, state):
        rank = len(tuple(state.shape))
        if rank == 1:  # vector
            return np.matmul(matrix, state[:, np.newaxis])[:, 0]
        elif rank == 2:  # matrix
            return np.matmul(matrix, state)
        else:
            raise_error(
                ValueError,
                "Cannot multiply Hamiltonian with " "rank-{} tensor.".format(rank),
            )

    def test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
                [4, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            ]
        elif name == "test_probabilistic_measurement":
            if "cuda" in self.device:  # pragma: no cover
                return {0: 273, 1: 233, 2: 242, 3: 252}
            else:
                return {0: 271, 1: 239, 2: 242, 3: 248}
        elif name == "test_unbalanced_probabilistic_measurement":
            if "cuda" in self.device:  # pragma: no cover
                return {0: 196, 1: 153, 2: 156, 3: 495}
            else:
                return {0: 168, 1: 188, 2: 154, 3: 490}
        elif name == "test_post_measurement_bitflips_on_circuit":
            return [
                {5: 30},
                {5: 16, 7: 10, 6: 2, 3: 1, 4: 1},
                {3: 6, 5: 6, 7: 5, 2: 4, 4: 3, 0: 2, 1: 2, 6: 2},
            ]
