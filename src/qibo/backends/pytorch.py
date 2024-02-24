"""PyTorch backend."""

from collections import Counter
from typing import Union

import numpy as np
import torch

from qibo import __version__
from qibo.backends import einsum_utils
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

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def set_seed(self, seed):
        self.np.manual_seed(seed)

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
        elif isinstance(dtype, self.np.dtype):
            dtype = dtype
        elif isinstance(dtype, type):
            dtype = torch_dtype_dict[dtype.__name__]
        else:
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

    def apply_gate(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.np.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, targets = einsum_utils.control_order(gate, nqubits)
            state = state.permute(*order)
            state = self.np.reshape(state, (2**ncontrol,) + nactive * (2,))
            opstring = einsum_utils.apply_gate_string(targets, nactive)
            updates = self.np.einsum(opstring, state[-1], matrix)
            state = self.np.cat([state[:-1], updates[None]], axis=0)
            state = self.np.reshape(state, nqubits * (2,))
            state = state.permute(*einsum_utils.reverse_order(order))
        else:
            matrix = self.np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.np.einsum(opstring, state, matrix)
        return self.np.reshape(state, (2**nqubits,))

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
        # return self.np.jit.script(func)

    def matrix(self, gate):
        npmatrix = super().matrix(gate)
        return self.np.tensor(npmatrix, dtype=self.dtype)

    def matrix_parametrized(self, gate):
        npmatrix = super().matrix_parametrized(gate)
        return self.np.tensor(npmatrix, dtype=self.dtype)

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
        return probs.permute(*[reduced.get(i) for i in qubits])

    def calculate_probabilities(self, state, qubits, nqubits):
        rtype = state.real.dtype
        unmeasured_qubits = tuple(i for i in range(nqubits) if i not in qubits)
        state = self.np.reshape(self.np.abs(state) ** 2, nqubits * (2,))
        probs = self.np.sum(state.type(rtype), unmeasured_qubits)
        return self._order_probabilities(probs, qubits, nqubits).reshape(-1)

    def calculate_probabilities_density_matrix(self, state, qubits, nqubits):
        order = tuple(sorted(qubits))
        order += tuple(i for i in range(nqubits) if i not in qubits)
        order = order + tuple(i + nqubits for i in order)
        shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
        state = self.np.reshape(state, 2 * nqubits * (2,))
        state = self.np.reshape(state.permute(*order), shape)
        probs = self.np.abs(self.np.einsum("abab->a", state))
        probs = self.np.reshape(probs, len(qubits) * (2,))
        return self._order_probabilities(probs, qubits, nqubits).view(-1)

    def calculate_frequencies(self, samples):
        res, counts = self.np.unique(samples, return_counts=True)
        res, counts = res.tolist(), counts.tolist()
        return Counter(zip(res, counts))

    def update_frequencies(self, frequencies, probabilities, nsamples):
        frequencies = self.cast(frequencies, dtype="int")
        samples = self.sample_shots(probabilities, nsamples)
        unique_samples, counts = self.np.unique(samples, return_counts=True)
        frequencies.index_add_(
            0, self.cast(unique_samples, dtype="int"), self.cast(counts, dtype="int")
        )
        return frequencies

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
        ud = self.np.transpose(self.np.conj(eigenvectors), dim0=0, dim1=1)
        return self.np.matmul(eigenvectors, self.np.matmul(expd, ud))

    def calculate_expectation_state(self, hamiltonian, state, normalize):
        state = self.cast(state)
        statec = self.np.conj(state)
        hstate = self.cast(hamiltonian @ state)
        ev = self.np.real(self.np.sum(statec * hstate))
        if normalize:
            ev = ev / self.np.sum(self.np.square(self.np.abs(state)))
        return ev

    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        if self.issparse(matrix1) or self.issparse(matrix2):
            return self.np.sparse.mm(matrix1, matrix2)  # pylint: disable=E1102
        return self.np.matmul(matrix1, matrix2)

    def calculate_hamiltonian_state_product(self, matrix, state):
        return self.np.matmul(matrix, state)

    def calculate_overlap(self, state1, state2):
        return self.np.abs(
            self.np.sum(self.np.conj(self.cast(state1)) * self.cast(state2))
        )

    def calculate_overlap_density_matrix(self, state1, state2):
        return self.np.trace(
            self.np.matmul(self.np.conj(self.cast(state1)).T, self.cast(state2))
        )

    def apply_gate_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.np.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            matrixc = self.np.conj(matrix)
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            n = 2**ncontrol

            order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
            state = state.permute(*order)
            state = self.np.reshape(state, 2 * (n,) + 2 * nactive * (2,))

            leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(
                targets, nactive
            )
            state01 = state[: n - 1, n - 1]
            state01 = self.np.einsum(rightc, state01, matrixc)
            state10 = state[n - 1, : n - 1]
            state10 = self.np.einsum(leftc, state10, matrix)

            left, right = einsum_utils.apply_gate_density_matrix_string(
                targets, nactive
            )
            state11 = state[n - 1, n - 1]
            state11 = self.np.einsum(right, state11, matrixc)
            state11 = self.np.einsum(left, state11, matrix)

            state00 = state[range(n - 1)]
            state00 = state00[:, range(n - 1)]
            state01 = self.np.cat([state00, state01[:, None]], dim=1)
            state10 = self.np.cat([state10, state11[None]], dim=0)
            state = self.np.cat([state01, state10[None]], dim=0)
            state = self.np.reshape(state, 2 * nqubits * (2,))
            state = state.permute(*einsum_utils.reverse_order(order))
        else:
            matrix = self.np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            matrixc = self.np.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.np.einsum(right, state, matrixc)
            state = self.np.einsum(left, state, matrix)
        return self.np.reshape(state, 2 * (2**nqubits,))

    def partial_trace(self, state, qubits, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, nqubits * (2,))
        axes = 2 * [list(qubits)]
        rho = self.np.tensordot(state, self.np.conj(state), dims=axes)
        shape = 2 * (2 ** (nqubits - len(qubits)),)
        return self.np.reshape(rho, shape)

    def partial_trace_density_matrix(self, state, qubits, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, 2 * nqubits * (2,))
        order = list(sorted(qubits))
        order += [i for i in range(nqubits) if i not in qubits]
        order += [i + nqubits for i in order]
        state = state.permute(*order)
        shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
        state = self.np.reshape(state, shape)
        return self.np.einsum("abac->bc", state)

    def _append_zeros(self, state, qubits, results):
        """Helper method for collapse."""
        for q, r in zip(qubits, results):
            state = self.np.unsqueeze(state, dim=q)
            if r:
                state = self.np.cat([self.np.zeros_like(state), state], dim=q)
            else:
                state = self.np.cat([state, self.np.zeros_like(state)], dim=q)
        return state

    def collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = self.samples_to_binary(shot, len(qubits))[0]
        state = self.np.reshape(state, nqubits * (2,))
        order = list(qubits) + [q for q in range(nqubits) if q not in qubits]
        state = state.permute(*order)
        subshape = (2 ** len(qubits),) + (nqubits - len(qubits)) * (2,)
        state = self.np.reshape(state, subshape)[int(shot)]
        if normalize:
            norm = self.np.sqrt(self.np.sum(self.np.abs(state) ** 2))
            state = state / norm
        state = self._append_zeros(state, qubits, binshot)
        return self.np.reshape(state, shape)

    def collapse_density_matrix(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = list(self.samples_to_binary(shot, len(qubits))[0])
        order = list(qubits) + [q + nqubits for q in qubits]
        order.extend(q for q in range(nqubits) if q not in qubits)
        order.extend(q + nqubits for q in range(nqubits) if q not in qubits)
        state = self.np.reshape(state, 2 * nqubits * (2,))
        state = state.permute(*order)
        subshape = 2 * (2 ** len(qubits),) + 2 * (nqubits - len(qubits)) * (2,)
        state = self.np.reshape(state, subshape)[int(shot), int(shot)]
        n = 2 ** (len(state.shape) // 2)
        if normalize:
            norm = self.np.trace(self.np.reshape(state, (n, n)))
            state = state / norm
        qubits = qubits + [q + nqubits for q in qubits]
        state = self._append_zeros(state, qubits, 2 * binshot)
        return self.np.reshape(state, shape)

    def reset_error_density_matrix(self, gate, state, nqubits):
        from qibo.gates import X  # pylint: disable=C0415

        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits[0]
        p_0, p_1 = gate.init_kwargs["p_0"], gate.init_kwargs["p_1"]
        trace = self.partial_trace_density_matrix(state, (q,), nqubits)
        trace = self.np.reshape(trace, 2 * (nqubits - 1) * (2,))
        zero = self.zero_density_matrix(1)
        zero = self.np.tensordot(trace, zero, dims=0)
        order = list(range(2 * nqubits - 2))
        order.insert(q, 2 * nqubits - 2)
        order.insert(q + nqubits, 2 * nqubits - 1)
        zero = self.np.reshape(zero.permute(*order), shape)
        state = (1 - p_0 - p_1) * state + p_0 * zero
        return state + p_1 * self.apply_gate_density_matrix(X(q), zero, nqubits)

    def thermal_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        shape = state.shape
        state = self.apply_gate(gate, state.view(-1), 2 * nqubits)
        return self.np.reshape(state, shape)

    def identity_density_matrix(self, nqubits, normalize: bool = True):
        state = self.np.eye(2**nqubits, dtype=self.np.complex128)
        if normalize is True:
            state /= 2**nqubits
        return state

    def depolarizing_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits
        lam = gate.init_kwargs["lam"]
        trace = self.partial_trace_density_matrix(state, q, nqubits)
        trace = self.np.reshape(trace, 2 * (nqubits - len(q)) * (2,))
        identity = self.identity_density_matrix(len(q))
        identity = self.np.reshape(identity, 2 * len(q) * (2,))
        identity = self.np.tensordot(trace, identity, dims=0)
        qubits = list(range(nqubits))
        for j in q:
            qubits.pop(qubits.index(j))
        qubits.sort()
        qubits += list(q)
        qubit_1 = list(range(nqubits - len(q))) + list(
            range(2 * (nqubits - len(q)), 2 * nqubits - len(q))
        )
        qubit_2 = list(range(nqubits - len(q), 2 * (nqubits - len(q)))) + list(
            range(2 * nqubits - len(q), 2 * nqubits)
        )
        qs = [qubit_1, qubit_2]
        order = []
        for qj in qs:
            qj = [qj[qubits.index(i)] for i in range(len(qubits))]
            order += qj
        identity = identity.permute(*order)
        identity = self.np.reshape(identity, shape)
        state = (1 - lam) * state + lam * identity
        return state

    def test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 2, 2, 4, 4, 0],
                [4, 0, 0, 1, 0, 0, 0, 4, 4, 0],
                [4, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            ]

        if name == "test_probabilistic_measurement":
            if "cuda" in self.device:  # pragma: no cover
                return {0: 273, 1: 233, 2: 242, 3: 252}
            return {0: 271, 1: 239, 2: 242, 3: 248}

        if name == "test_unbalanced_probabilistic_measurement":
            if "cuda" in self.device:  # pragma: no cover
                return {0: 196, 1: 153, 2: 156, 3: 495}
            return {0: 168, 1: 188, 2: 154, 3: 490}

        if name == "test_post_measurement_bitflips_on_circuit":
            return [
                {5: 30},
                {5: 16, 7: 10, 6: 2, 3: 1, 4: 1},
                {3: 6, 5: 6, 7: 5, 2: 4, 4: 3, 0: 2, 1: 2, 6: 2},
            ]
