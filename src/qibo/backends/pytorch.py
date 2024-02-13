import collections
from typing import Union

import numpy as np
import torch

from qibo import __version__
from qibo.backends import einsum_utils
from qibo.backends.npmatrices import NumpyMatrices
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState

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
    # Redefine parametrized gate matrices for backpropagation to work

    def __init__(self, dtype):
        super().__init__(dtype)
        self.torch = torch
        self.torch_dtype = torch_dtype_dict[dtype]

    def _cast(self, x, dtype):
        return self.torch.tensor(x, dtype=dtype)

    def Unitary(self, u):
        return self._cast(u, dtype=self.torch_dtype)


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
        self.torch = torch
        self.torch_dtype = torch_dtype_dict[self.dtype]
        self.tensor_types = (self.torch.Tensor, np.ndarray)

    def set_device(self, device):  # pragma: no cover
        self.device = device

    def cast(
        self,
        x: Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray]],
        dtype: Union[str, torch.dtype, np.dtype, type] = None,
        copy=False,
    ):
        """Casts input as a Torch tensor of the specified dtype.
        This method supports casting of single tensors or lists of tensors as for the Tensoflow backend.

        Args:
            x (Union[torch.Tensor, list[torch.Tensor], np.ndarray, list[np.ndarray], int, float, complex]): Input to be casted.
            dtype (Union[str, torch.dtype, np.dtype, type]): Target data type. If None, the default dtype of the backend is used.
            copy (bool): If True, the input tensor is copied before casting.
        """
        if dtype is None:
            dtype = self.torch_dtype
        elif isinstance(dtype, self.torch.dtype):
            dtype = dtype
        elif isinstance(dtype, type):
            dtype = torch_dtype_dict[dtype.__name__]
        else:
            dtype = torch_dtype_dict[str(dtype)]
        if isinstance(x, self.torch.Tensor):
            x = x.to(dtype)
        elif isinstance(x, list):
            if all(isinstance(i, self.torch.Tensor) for i in x):
                x = [i.to(dtype) for i in x]
            else:
                x = [self.torch.tensor(i, dtype=dtype) for i in x]
        else:
            x = self.torch.tensor(x, dtype=dtype)
        if copy:
            return x.clone()
        return x

    def apply_gate(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.torch.reshape(state, nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.torch.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, targets = einsum_utils.control_order(gate, nqubits)
            state = state.permute(*order)
            state = self.torch.reshape(state, (2**ncontrol,) + nactive * (2,))
            opstring = einsum_utils.apply_gate_string(targets, nactive)
            updates = self.torch.einsum(opstring, state[-1], matrix)
            state = self.torch.cat([state[:-1], updates[None]], axis=0)
            state = self.torch.reshape(state, nqubits * (2,))
            state = state.permute(*einsum_utils.reverse_order(order))
        else:
            matrix = self.torch.reshape(matrix, 2 * len(gate.qubits) * (2,))
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.torch.einsum(opstring, state, matrix)
        return self.torch.reshape(state, (2**nqubits,))

    def issparse(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.is_sparse
        return super().issparse(x)

    def to_numpy(self, x):
        if type(x) is self.torch.Tensor:
            return x.detach().cpu().numpy()
        return x

    def compile(self, func):
        return self.torch.jit.script(func)

    def zero_state(self, nqubits):
        state = self.torch.zeros(2**nqubits, dtype=self.torch_dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits):
        state = self.torch.zeros(2 * (2**nqubits,), dtype=self.torch_dtype)
        state[0, 0] = 1
        return state

    def matrix(self, gate):
        npmatrix = super().matrix(gate)
        return self.torch.tensor(npmatrix, dtype=self.torch_dtype)

    def matrix_parametrized(self, gate):
        npmatrix = super().matrix_parametrized(gate)
        return self.torch.tensor(npmatrix, dtype=self.torch_dtype)

    def matrix_fused(self, gate):
        npmatrix = super().matrix_fused(gate)
        return self.torch.tensor(npmatrix, dtype=self.torch_dtype)

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        if initial_state is not None:
            initial_state = self.cast(initial_state)
        return super().execute_circuit(circuit, initial_state, nshots)

    def execute_circuit_repeated(self, circuit, nshots, initial_state=None):
        """
        Execute the circuit `nshots` times to retrieve probabilities, frequencies
        and samples. Note that this method is called only if a unitary channel
        is present in the circuit (i.e. noisy simulation) and `density_matrix=False`, or
        if some collapsing measuremnt is performed.
        """

        if (
            circuit.has_collapse
            and not circuit.measurements
            and not circuit.density_matrix
        ):
            raise RuntimeError(
                "The circuit contains only collapsing measurements (`collapse=True`) but `density_matrix=False`. Please set `density_matrix=True` to retrieve the final state after execution."
            )

        results, final_states = [], []
        nqubits = circuit.nqubits

        if not circuit.density_matrix:
            samples = []
            target_qubits = [
                measurement.target_qubits for measurement in circuit.measurements
            ]
            target_qubits = sum(target_qubits, tuple())

        for _ in range(nshots):
            if circuit.density_matrix:
                if initial_state is None:
                    state = self.zero_density_matrix(nqubits)
                else:
                    state = self.cast(initial_state, copy=True)

                for gate in circuit.queue:
                    if gate.symbolic_parameters:
                        gate.substitute_symbols()
                    state = gate.apply_density_matrix(self, state, nqubits)
            else:
                if circuit.accelerators:  # pragma: no cover
                    # pylint: disable=E1111
                    state = self.execute_distributed_circuit(circuit, initial_state)
                else:
                    if initial_state is None:
                        state = self.zero_state(nqubits)
                    else:
                        state = self.cast(initial_state, copy=True)

                    for gate in circuit.queue:
                        if gate.symbolic_parameters:
                            gate.substitute_symbols()
                        state = gate.apply(self, state, nqubits)

            if circuit.density_matrix:
                final_states.append(state)
            if circuit.measurements:
                result = CircuitResult(
                    state, circuit.measurements, backend=self, nshots=1
                )
                sample = result.samples()[0]
                results.append(sample)
                if not circuit.density_matrix:
                    samples.append("".join([str(s) for s in sample]))
                for gate in circuit.measurements:
                    gate.result.reset()

        if circuit.density_matrix:  # this implies also it has_collapse
            assert circuit.has_collapse
            final_state = self.torch.mean(self.torch.stack(final_states), 0)
            if circuit.measurements:
                qubits = [q for m in circuit.measurements for q in m.target_qubits]
                final_result = CircuitResult(
                    final_state,
                    circuit.measurements,
                    backend=self,
                    samples=self.aggregate_shots(results),
                    nshots=nshots,
                )
            else:
                final_result = QuantumState(final_state, backend=self)
            circuit._final_state = final_result
            return final_result
        else:
            final_result = MeasurementOutcomes(
                circuit.measurements,
                backend=self,
                samples=self.aggregate_shots(results),
                nshots=nshots,
            )
            final_result._repeated_execution_frequencies = self.calculate_frequencies(
                samples
            )
            circuit._final_state = final_result
            return final_result

    def sample_shots(self, probabilities, nshots):
        return self.torch.multinomial(
            self.cast(probabilities, dtype="float"), nshots, replacement=True
        )

    def samples_to_binary(self, samples, nqubits):
        qrange = self.torch.arange(nqubits - 1, -1, -1, dtype=self.torch.int32)
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
        state = self.torch.reshape(self.torch.abs(state) ** 2, nqubits * (2,))
        probs = self.torch.sum(state.type(rtype), dim=unmeasured_qubits)
        return self._order_probabilities(probs, qubits, nqubits).view(-1)

    def calculate_probabilities_density_matrix(self, state, qubits, nqubits):
        order = tuple(sorted(qubits))
        order += tuple(i for i in range(nqubits) if i not in qubits)
        order = order + tuple(i + nqubits for i in order)
        shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
        state = self.torch.reshape(state, 2 * nqubits * (2,))
        state = self.torch.reshape(state.permute(*order), shape)
        probs = self.torch.abs(self.torch.einsum("abab->a", state))
        probs = self.torch.reshape(probs, len(qubits) * (2,))
        return self._order_probabilities(probs, qubits, nqubits).view(-1)

    def sample_frequencies(self, probabilities, nshots):
        from qibo.config import SHOT_BATCH_SIZE

        nprobs = probabilities / self.torch.sum(probabilities)
        frequencies = self.torch.zeros(len(nprobs), dtype=self.torch.int64)
        for _ in range(nshots // SHOT_BATCH_SIZE):
            frequencies = self.update_frequencies(frequencies, nprobs, SHOT_BATCH_SIZE)
        frequencies = self.update_frequencies(
            frequencies, nprobs, nshots % SHOT_BATCH_SIZE
        )
        return collections.Counter(
            {i: f.item() for i, f in enumerate(frequencies) if f > 0}
        )

    def calculate_frequencies(self, samples):
        res, counts = self.torch.unique(samples, return_counts=True)
        res, counts = res.tolist(), counts.tolist()
        return collections.Counter({k: v for k, v in zip(res, counts)})

    def update_frequencies(self, frequencies, probabilities, nsamples):
        frequencies = self.cast(frequencies, dtype="int")
        samples = self.sample_shots(probabilities, nsamples)
        unique_samples, counts = self.torch.unique(samples, return_counts=True)
        frequencies.index_add_(
            0, self.cast(unique_samples, dtype="int"), self.cast(counts, dtype="int")
        )
        return frequencies

    def calculate_norm(self, state, order=2):
        state = self.cast(state)
        return self.torch.norm(state, p=order)

    def calculate_norm_density_matrix(self, state, order="nuc"):
        state = self.cast(state)
        if order == "nuc":
            return np.trace(state)
        return self.torch.norm(state, p=order)

    def calculate_eigenvalues(self, matrix, k=6):
        return self.torch.linalg.eigvalsh(matrix)  # pylint: disable=not-callable

    def calculate_eigenvectors(self, matrix, k=6):
        return self.torch.linalg.eigh(matrix)  # pylint: disable=not-callable

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.issparse(matrix):
            return self.torch.linalg.matrix_exp(  # pylint: disable=not-callable
                -1j * a * matrix
            )
        return super().calculate_matrix_exp(a, matrix, eigenvectors, eigenvalues)

    def calculate_expectation_state(self, hamiltonian, state, normalize):
        state = self.cast(state)
        statec = self.torch.conj(state)
        hstate = self.cast(hamiltonian @ state)
        ev = self.torch.real(self.torch.sum(statec * hstate))
        if normalize:
            ev = ev / self.torch.sum(self.torch.square(self.torch.abs(state)))
        return ev

    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        if self.issparse(matrix1) or self.issparse(matrix2):
            return self.torch.sparse.mm(
                matrix1, matrix2
            )  # pylint: disable=not-callable
        return self.torch.matmul(matrix1, matrix2)

    def calculate_hamiltonian_state_product(self, matrix, state):
        return self.torch.matmul(matrix, state)

    def calculate_overlap(self, state1, state2):
        return self.torch.abs(
            self.torch.sum(self.torch.conj(self.cast(state1)) * self.cast(state2))
        )

    def calculate_overlap_density_matrix(self, state1, state2):
        return self.torch.trace(
            self.torch.matmul(self.torch.conj(self.cast(state1)).T, self.cast(state2))
        )

    def apply_gate_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.torch.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.torch.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            matrixc = self.torch.conj(matrix)
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            n = 2**ncontrol

            order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
            state = state.permute(*order)
            state = self.torch.reshape(state, 2 * (n,) + 2 * nactive * (2,))

            leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(
                targets, nactive
            )
            state01 = state[: n - 1, n - 1]
            state01 = self.torch.einsum(rightc, state01, matrixc)
            state10 = state[n - 1, : n - 1]
            state10 = self.torch.einsum(leftc, state10, matrix)

            left, right = einsum_utils.apply_gate_density_matrix_string(
                targets, nactive
            )
            state11 = state[n - 1, n - 1]
            state11 = self.torch.einsum(right, state11, matrixc)
            state11 = self.torch.einsum(left, state11, matrix)

            state00 = state[range(n - 1)]
            state00 = state00[:, range(n - 1)]
            state01 = self.torch.cat([state00, state01[:, None]], dim=1)
            state10 = self.torch.cat([state10, state11[None]], dim=0)
            state = self.torch.cat([state01, state10[None]], dim=0)
            state = self.torch.reshape(state, 2 * nqubits * (2,))
            state = state.permute(*einsum_utils.reverse_order(order))
        else:
            matrix = self.torch.reshape(matrix, 2 * len(gate.qubits) * (2,))
            matrixc = self.torch.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.torch.einsum(right, state, matrixc)
            state = self.torch.einsum(left, state, matrix)
        return self.torch.reshape(state, 2 * (2**nqubits,))

    def partial_trace(self, state, qubits, nqubits):
        state = self.cast(state)
        state = self.torch.reshape(state, nqubits * (2,))
        axes = 2 * [list(qubits)]
        rho = self.torch.tensordot(state, self.torch.conj(state), dims=axes)
        shape = 2 * (2 ** (nqubits - len(qubits)),)
        return self.torch.reshape(rho, shape)

    def partial_trace_density_matrix(self, state, qubits, nqubits):
        state = self.cast(state)
        state = self.torch.reshape(state, 2 * nqubits * (2,))
        order = list(sorted(qubits))
        order += [i for i in range(nqubits) if i not in qubits]
        order += [i + nqubits for i in order]
        state = state.permute(*order)
        shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
        state = self.torch.reshape(state, shape)
        return self.torch.einsum("abac->bc", state)

    def _append_zeros(self, state, qubits, results):
        """Helper method for collapse."""
        for q, r in zip(qubits, results):
            state = self.torch.unsqueeze(state, dim=q)
            if r:
                state = self.torch.cat([self.torch.zeros_like(state), state], dim=q)
            else:
                state = self.torch.cat([state, self.torch.zeros_like(state)], dim=q)
        return state

    def collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = self.samples_to_binary(shot, len(qubits))[0]
        state = self.torch.reshape(state, nqubits * (2,))
        order = list(qubits) + [q for q in range(nqubits) if q not in qubits]
        state = state.permute(*order)
        subshape = (2 ** len(qubits),) + (nqubits - len(qubits)) * (2,)
        state = self.torch.reshape(state, subshape)[int(shot)]
        if normalize:
            norm = self.torch.sqrt(self.torch.sum(self.torch.abs(state) ** 2))
            state = state / norm
        state = self._append_zeros(state, qubits, binshot)
        return self.torch.reshape(state, shape)

    def collapse_density_matrix(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = list(self.samples_to_binary(shot, len(qubits))[0])
        order = list(qubits) + [q + nqubits for q in qubits]
        order.extend(q for q in range(nqubits) if q not in qubits)
        order.extend(q + nqubits for q in range(nqubits) if q not in qubits)
        state = self.torch.reshape(state, 2 * nqubits * (2,))
        state = state.permute(*order)
        subshape = 2 * (2 ** len(qubits),) + 2 * (nqubits - len(qubits)) * (2,)
        state = self.torch.reshape(state, subshape)[int(shot), int(shot)]
        n = 2 ** (len(state.shape) // 2)
        if normalize:
            norm = self.torch.trace(self.torch.reshape(state, (n, n)))
            state = state / norm
        qubits = qubits + [q + nqubits for q in qubits]
        state = self._append_zeros(state, qubits, 2 * binshot)
        return self.torch.reshape(state, shape)

    def reset_error_density_matrix(self, gate, state, nqubits):
        from qibo.gates import X

        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits[0]
        p_0, p_1 = gate.init_kwargs["p_0"], gate.init_kwargs["p_1"]
        trace = self.partial_trace_density_matrix(state, (q,), nqubits)
        trace = self.torch.reshape(trace, 2 * (nqubits - 1) * (2,))
        zero = self.zero_density_matrix(1)
        zero = self.torch.tensordot(trace, zero, dims=0)
        order = list(range(2 * nqubits - 2))
        order.insert(q, 2 * nqubits - 2)
        order.insert(q + nqubits, 2 * nqubits - 1)
        zero = self.torch.reshape(zero.permute(*order), shape)
        state = (1 - p_0 - p_1) * state + p_0 * zero
        return state + p_1 * self.apply_gate_density_matrix(X(q), zero, nqubits)

    def thermal_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        shape = state.shape
        state = self.apply_gate(gate, state.view(-1), 2 * nqubits)
        return self.torch.reshape(state, shape)

    def identity_density_matrix(self, nqubits, normalize: bool = True):
        state = self.torch.eye(2**nqubits, dtype=self.torch.complex128)
        if normalize is True:
            state /= 2**nqubits
        return state

    def depolarizing_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits
        lam = gate.init_kwargs["lam"]
        trace = self.partial_trace_density_matrix(state, q, nqubits)
        trace = self.torch.reshape(trace, 2 * (nqubits - len(q)) * (2,))
        identity = self.identity_density_matrix(len(q))
        identity = self.torch.reshape(identity, 2 * len(q) * (2,))
        identity = self.torch.tensordot(trace, identity, dims=0)
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
        identity = self.torch.reshape(identity, shape)
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
