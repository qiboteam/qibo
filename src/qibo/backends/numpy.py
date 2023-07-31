import collections

import numpy as np

from qibo import __version__
from qibo.backends import einsum_utils
from qibo.backends.abstract import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import log, raise_error
from qibo.states import CircuitResult


class NumpyBackend(Backend):
    def __init__(self):
        super().__init__()
        self.np = np
        self.name = "numpy"
        self.matrices = NumpyMatrices(self.dtype)
        self.tensor_types = np.ndarray
        self.versions = {"qibo": __version__, "numpy": self.np.__version__}
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
        elif self.issparse(x):
            return x.astype(dtype, copy=copy)
        return np.array(x, dtype=dtype, copy=copy)

    def issparse(self, x):
        from scipy import sparse

        return sparse.issparse(x)

    def to_numpy(self, x):
        if self.issparse(x):
            return x.toarray()
        return x

    def compile(self, func):
        return func

    def zero_state(self, nqubits):
        state = self.np.zeros(2**nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits):
        state = self.np.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state[0, 0] = 1
        return state

    def identity_density_matrix(self, nqubits, normalize: bool = True):
        state = self.np.eye(2**nqubits, dtype=self.dtype)
        if normalize is True:
            state /= 2**nqubits
        return state

    def plus_state(self, nqubits):
        state = self.np.ones(2**nqubits, dtype=self.dtype)
        state /= self.np.sqrt(2**nqubits)
        return state

    def plus_density_matrix(self, nqubits):
        state = self.np.ones(2 * (2**nqubits,), dtype=self.dtype)
        state /= 2**nqubits
        return state

    def asmatrix(self, gate):
        """Convert a gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        matrix = getattr(self.matrices, name)
        return matrix(2 ** len(gate.target_qubits)) if callable(matrix) else matrix

    def asmatrix_parametrized(self, gate):
        """Convert a parametrized gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        return getattr(self.matrices, name)(*gate.parameters)

    def asmatrix_fused(self, fgate):
        rank = len(fgate.target_qubits)
        matrix = np.eye(2**rank, dtype=self.dtype)
        for gate in fgate.gates:
            # transfer gate matrix to numpy as it is more efficient for
            # small tensor calculations
            # explicit to_numpy see https://github.com/qiboteam/qibo/issues/928
            gmatrix = self.to_numpy(gate.asmatrix(self))
            # Kronecker product with identity is needed to make the
            # original matrix have shape (2**rank x 2**rank)
            eye = np.eye(2 ** (rank - len(gate.qubits)), dtype=self.dtype)
            gmatrix = np.kron(gmatrix, eye)
            # Transpose the new matrix indices so that it targets the
            # target qubits of the original gate
            original_shape = gmatrix.shape
            gmatrix = np.reshape(gmatrix, 2 * rank * (2,))
            qubits = list(gate.qubits)
            indices = qubits + [q for q in fgate.target_qubits if q not in qubits]
            indices = np.argsort(indices)
            transpose_indices = list(indices)
            transpose_indices.extend(indices + rank)
            gmatrix = np.transpose(gmatrix, transpose_indices)
            gmatrix = np.reshape(gmatrix, original_shape)
            # fuse the individual gate matrix to the total ``FusedGate`` matrix
            matrix = gmatrix @ matrix
        return matrix

    def control_matrix(self, gate):
        if len(gate.control_qubits) > 1:
            raise_error(
                NotImplementedError,
                "Cannot calculate controlled "
                "unitary for more than two "
                "control qubits.",
            )
        matrix = gate.asmatrix(self)
        shape = matrix.shape
        if shape != (2, 2):
            raise_error(
                ValueError,
                "Cannot use ``control_unitary`` method on "
                "gate matrix of shape {}.".format(shape),
            )
        zeros = self.np.zeros((2, 2), dtype=self.dtype)
        part1 = self.np.concatenate([self.np.eye(2, dtype=self.dtype), zeros], axis=0)
        part2 = self.np.concatenate([zeros, matrix], axis=0)
        return self.np.concatenate([part1, part2], axis=1)

    def apply_gate(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, nqubits * (2,))
        matrix = gate.asmatrix(self)
        if gate.is_controlled_by:
            matrix = self.np.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, targets = einsum_utils.control_order(gate, nqubits)
            state = self.np.transpose(state, order)
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = self.np.reshape(state, (2**ncontrol,) + nactive * (2,))
            opstring = einsum_utils.apply_gate_string(targets, nactive)
            updates = self.np.einsum(opstring, state[-1], matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = self.np.concatenate([state[:-1], updates[self.np.newaxis]], axis=0)
            state = self.np.reshape(state, nqubits * (2,))
            # Put qubit indices back to their proper places
            state = self.np.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = self.np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.np.einsum(opstring, state, matrix)
        return self.np.reshape(state, (2**nqubits,))

    def apply_gate_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, 2 * nqubits * (2,))
        matrix = gate.asmatrix(self)
        if gate.is_controlled_by:
            matrix = self.np.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            matrixc = self.np.conj(matrix)
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            n = 2**ncontrol

            order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
            state = self.np.transpose(state, order)
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
            state01 = self.np.concatenate(
                [state00, state01[:, self.np.newaxis]], axis=1
            )
            state10 = self.np.concatenate([state10, state11[self.np.newaxis]], axis=0)
            state = self.np.concatenate([state01, state10[self.np.newaxis]], axis=0)
            state = self.np.reshape(state, 2 * nqubits * (2,))
            state = self.np.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = self.np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            matrixc = self.np.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.np.einsum(right, state, matrixc)
            state = self.np.einsum(left, state, matrix)
        return self.np.reshape(state, 2 * (2**nqubits,))

    def apply_gate_half_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = np.reshape(state, 2 * nqubits * (2,))
        matrix = gate.asmatrix(self)
        if gate.is_controlled_by:  # pragma: no cover
            raise_error(
                NotImplementedError,
                "Gate density matrix half call is "
                "not implemented for ``controlled_by``"
                "gates.",
            )
        else:
            matrix = np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            left, _ = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = np.einsum(left, state, matrix)
        return np.reshape(state, 2 * (2**nqubits,))

    def apply_channel(self, channel, state, nqubits):
        for coeff, gate in zip(channel.coefficients, channel.gates):
            if self.np.random.random() < coeff:
                state = self.apply_gate(gate, state, nqubits)
        return state

    def apply_channel_density_matrix(self, channel, state, nqubits):
        state = self.cast(state)
        new_state = (1 - channel.coefficient_sum) * state
        for coeff, gate in zip(channel.coefficients, channel.gates):
            new_state += coeff * self.apply_gate_density_matrix(gate, state, nqubits)
        return new_state

    def _append_zeros(self, state, qubits, results):
        """Helper method for collapse."""
        for q, r in zip(qubits, results):
            state = self.np.expand_dims(state, axis=q)
            if r:
                state = self.np.concatenate([self.np.zeros_like(state), state], axis=q)
            else:
                state = self.np.concatenate([state, self.np.zeros_like(state)], axis=q)
        return state

    def collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = self.samples_to_binary(shot, len(qubits))[0]
        state = self.np.reshape(state, nqubits * (2,))
        order = list(qubits) + [q for q in range(nqubits) if q not in qubits]
        state = self.np.transpose(state, order)
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
        state = self.np.transpose(state, order)
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
        from qibo.gates import X

        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits[0]
        p_0, p_1 = gate.init_kwargs["p_0"], gate.init_kwargs["p_1"]
        trace = self.partial_trace_density_matrix(state, (q,), nqubits)
        trace = self.np.reshape(trace, 2 * (nqubits - 1) * (2,))
        zero = self.zero_density_matrix(1)
        zero = self.np.tensordot(trace, zero, axes=0)
        order = list(range(2 * nqubits - 2))
        order.insert(q, 2 * nqubits - 2)
        order.insert(q + nqubits, 2 * nqubits - 1)
        zero = self.np.reshape(self.np.transpose(zero, order), shape)
        state = (1 - p_0 - p_1) * state + p_0 * zero
        return state + p_1 * self.apply_gate_density_matrix(X(q), zero, nqubits)

    def thermal_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        shape = state.shape
        state = self.apply_gate(gate, state.ravel(), 2 * nqubits)
        return self.np.reshape(state, shape)

    def depolarizing_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits
        lam = gate.init_kwargs["lam"]
        trace = self.partial_trace_density_matrix(state, q, nqubits)
        trace = self.np.reshape(trace, 2 * (nqubits - len(q)) * (2,))
        identity = self.identity_density_matrix(len(q))
        identity = self.np.reshape(identity, 2 * len(q) * (2,))
        identity = self.np.tensordot(trace, identity, axes=0)
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
        identity = self.np.reshape(self.np.transpose(identity, order), shape)
        state = (1 - lam) * state + lam * identity
        return state

    def execute_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):
        if isinstance(initial_state, type(circuit)):
            if not initial_state.density_matrix == circuit.density_matrix:
                raise_error(
                    ValueError,
                    f"""Cannot set circuit with density_matrix {initial_state.density_matrix} as
                      initial state for circuit with density_matrix {circuit.density_matrix}.""",
                )
            elif (
                not initial_state.accelerators == circuit.accelerators
            ):  # pragma: no cover
                raise_error(
                    ValueError,
                    f"""Cannot set circuit with accelerators {initial_state.density_matrix} as
                      initial state for circuit with accelerators {circuit.density_matrix}.""",
                )
            else:
                return self.execute_circuit(initial_state + circuit, None, nshots)

        if circuit.repeated_execution:
            return self.execute_circuit_repeated(circuit, initial_state, nshots)

        if circuit.accelerators:  # pragma: no cover
            return self.execute_distributed_circuit(circuit, initial_state, nshots)

        try:
            nqubits = circuit.nqubits
            if isinstance(initial_state, CircuitResult):
                initial_state = initial_state.state()

            if circuit.density_matrix:
                if initial_state is None:
                    state = self.zero_density_matrix(nqubits)
                else:
                    # cast to proper complex type
                    state = self.cast(initial_state)

                for gate in circuit.queue:
                    state = gate.apply_density_matrix(self, state, nqubits)

            else:
                if initial_state is None:
                    state = self.zero_state(nqubits)
                else:
                    # cast to proper complex type
                    state = self.cast(initial_state)

                for gate in circuit.queue:
                    state = gate.apply(self, state, nqubits)

            if return_array:
                return state
            else:
                circuit._final_state = CircuitResult(self, circuit, state, nshots)
                return circuit._final_state

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def execute_circuit_repeated(self, circuit, initial_state=None, nshots=None):
        if nshots is None:
            nshots = 1

        results = []
        nqubits = circuit.nqubits

        if not circuit.density_matrix:
            samples = []
            target_qubits = [
                measurement.target_qubits for measurement in circuit.measurements
            ]
            target_qubits = sum(target_qubits, tuple())
            probabilities = np.zeros(2 ** len(target_qubits), dtype=float)
            probabilities = self.cast(probabilities, dtype=probabilities.dtype)

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
                    state = self.execute_distributed_circuit(
                        circuit, initial_state, return_array=True
                    )
                else:
                    if initial_state is None:
                        state = self.zero_state(nqubits)
                    else:
                        state = self.cast(initial_state, copy=True)

                    for gate in circuit.queue:
                        if gate.symbolic_parameters:
                            gate.substitute_symbols()
                        state = gate.apply(self, state, nqubits)

            if circuit.measurements:
                result = CircuitResult(self, circuit, state, 1)
                sample = result.samples()[0]
                results.append(sample)
                if not circuit.density_matrix:
                    probabilities += result.probabilities()
                    samples.append("".join([str(s) for s in sample]))
            else:
                results.append(state)

        if circuit.measurements:
            final_result = CircuitResult(self, circuit, state, nshots)
            final_result._samples = self.aggregate_shots(results)
            if not circuit.density_matrix:
                final_result._repeated_execution_probabilities = probabilities / nshots
                final_result._repeated_execution_frequencies = (
                    self.calculate_frequencies(samples)
                )
            circuit._final_state = final_result
            return final_result

        circuit._final_state = CircuitResult(self, circuit, results[-1], nshots)

        return results

    def execute_distributed_circuit(
        self, circuit, initial_state=None, nshots=None, return_array=False
    ):
        raise_error(
            NotImplementedError, f"{self} does not support distributed execution."
        )

    def circuit_result_representation(self, result):
        return result.symbolic()

    def circuit_result_tensor(self, result):
        return result.execution_result

    def circuit_result_probabilities(self, result, qubits=None):
        if qubits is None:
            qubits = result.measurement_gate.qubits

        state = self.circuit_result_tensor(result)
        if result.density_matrix:
            return self.calculate_probabilities_density_matrix(
                state, qubits, result.nqubits
            )
        else:
            return self.calculate_probabilities(state, qubits, result.nqubits)

    def calculate_symbolic(
        self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20
    ):
        state = self.to_numpy(state)
        terms = []
        for i in np.nonzero(state)[0]:
            b = bin(i)[2:].zfill(nqubits)
            if np.abs(state[i]) >= cutoff:
                x = np.round(state[i], decimals)
                terms.append(f"{x}|{b}>")
            if len(terms) >= max_terms:
                terms.append("...")
                return terms
        return terms

    def calculate_symbolic_density_matrix(
        self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20
    ):
        state = self.to_numpy(state)
        terms = []
        indi, indj = np.nonzero(state)
        for i, j in zip(indi, indj):
            bi = bin(i)[2:].zfill(nqubits)
            bj = bin(j)[2:].zfill(nqubits)
            if np.abs(state[i, j]) >= cutoff:
                x = np.round(state[i, j], decimals)
                terms.append(f"{x}|{bi}><{bj}|")
            if len(terms) >= max_terms:
                terms.append("...")
                return terms
        return terms

    def _order_probabilities(self, probs, qubits, nqubits):
        """Arrange probabilities according to the given ``qubits`` ordering."""
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
        probs = self.np.sum(state.astype(rtype), axis=unmeasured_qubits)
        return self._order_probabilities(probs, qubits, nqubits).ravel()

    def calculate_probabilities_density_matrix(self, state, qubits, nqubits):
        order = tuple(sorted(qubits))
        order += tuple(i for i in range(nqubits) if i not in qubits)
        order = order + tuple(i + nqubits for i in order)
        shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
        state = self.np.reshape(state, 2 * nqubits * (2,))
        state = self.np.reshape(self.np.transpose(state, order), shape)
        probs = self.np.abs(self.np.einsum("abab->a", state))
        probs = self.np.reshape(probs, len(qubits) * (2,))
        return self._order_probabilities(probs, qubits, nqubits).ravel()

    def set_seed(self, seed):
        self.np.random.seed(seed)

    def sample_shots(self, probabilities, nshots):
        return self.np.random.choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    def aggregate_shots(self, shots):
        return self.np.array(shots, dtype=shots[0].dtype)

    def samples_to_binary(self, samples, nqubits):
        qrange = self.np.arange(nqubits - 1, -1, -1, dtype="int32")
        return self.np.mod(self.np.right_shift(samples[:, self.np.newaxis], qrange), 2)

    def samples_to_decimal(self, samples, nqubits):
        qrange = self.np.arange(nqubits - 1, -1, -1, dtype="int32")
        qrange = (2**qrange)[:, self.np.newaxis]
        return self.np.matmul(samples, qrange)[:, 0]

    def calculate_frequencies(self, samples):
        res, counts = self.np.unique(samples, return_counts=True)
        res, counts = self.np.array(res), self.np.array(counts)
        return collections.Counter({k: v for k, v in zip(res, counts)})

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.np.unique(samples, return_counts=True)
        frequencies[res] += counts
        return frequencies

    def sample_frequencies(self, probabilities, nshots):
        from qibo.config import SHOT_BATCH_SIZE

        nprobs = probabilities / self.np.sum(probabilities)
        frequencies = self.np.zeros(len(nprobs), dtype="int64")
        for _ in range(nshots // SHOT_BATCH_SIZE):
            frequencies = self.update_frequencies(frequencies, nprobs, SHOT_BATCH_SIZE)
        frequencies = self.update_frequencies(
            frequencies, nprobs, nshots % SHOT_BATCH_SIZE
        )
        return collections.Counter({i: f for i, f in enumerate(frequencies) if f > 0})

    def apply_bitflips(self, noiseless_samples, bitflip_probabilities):
        fprobs = self.np.array(bitflip_probabilities, dtype="float64")
        sprobs = self.np.random.random(noiseless_samples.shape)
        flip_0 = self.np.array(sprobs < fprobs[0], dtype=noiseless_samples.dtype)
        flip_1 = self.np.array(sprobs < fprobs[1], dtype=noiseless_samples.dtype)
        noisy_samples = noiseless_samples + (1 - noiseless_samples) * flip_0
        noisy_samples = noisy_samples - noiseless_samples * flip_1
        return noisy_samples

    def partial_trace(self, state, qubits, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, nqubits * (2,))
        axes = 2 * [list(qubits)]
        rho = self.np.tensordot(state, self.np.conj(state), axes=axes)
        shape = 2 * (2 ** (nqubits - len(qubits)),)
        return self.np.reshape(rho, shape)

    def partial_trace_density_matrix(self, state, qubits, nqubits):
        state = self.cast(state)
        state = self.np.reshape(state, 2 * nqubits * (2,))

        order = tuple(sorted(qubits))
        order += tuple(i for i in range(nqubits) if i not in qubits)
        order += tuple(i + nqubits for i in order)
        shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))

        state = self.np.transpose(state, order)
        state = self.np.reshape(state, shape)
        return self.np.einsum("abac->bc", state)

    def entanglement_entropy(self, rho):
        from qibo.config import EIGVAL_CUTOFF

        # Diagonalize
        eigvals = self.np.linalg.eigvalsh(rho).real
        # Treating zero and negative eigenvalues
        masked_eigvals = eigvals[eigvals > EIGVAL_CUTOFF]
        spectrum = -1 * self.np.log(masked_eigvals)
        entropy = self.np.sum(masked_eigvals * spectrum) / self.np.log(2.0)
        return entropy, spectrum

    def calculate_norm(self, state):
        state = self.cast(state)
        return self.np.sqrt(self.np.sum(self.np.abs(state) ** 2))

    def calculate_norm_density_matrix(self, state):
        state = self.cast(state)
        return self.np.trace(state)

    def calculate_overlap(self, state1, state2):
        state1 = self.cast(state1)
        state2 = self.cast(state2)
        return self.np.abs(self.np.sum(self.np.conj(state1) * state2))

    def calculate_overlap_density_matrix(self, state1, state2):
        raise_error(NotImplementedError)

    def calculate_eigenvalues(self, matrix, k=6):
        if self.issparse(matrix):
            log.warning(
                "Calculating sparse matrix eigenvectors because "
                "sparse modules do not provide ``eigvals`` method."
            )
            return self.calculate_eigenvectors(matrix, k=k)[0]
        return np.linalg.eigvalsh(matrix)

    def calculate_eigenvectors(self, matrix, k=6):
        if self.issparse(matrix):
            if k < matrix.shape[0]:
                from scipy.sparse.linalg import eigsh

                return eigsh(matrix, k=k, which="SA")
            else:  # pragma: no cover
                matrix = self.to_numpy(matrix)
        return np.linalg.eigh(matrix)

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.issparse(matrix):
            if self.issparse(matrix):
                from scipy.sparse.linalg import expm
            else:
                from scipy.linalg import expm
            return expm(-1j * a * matrix)
        else:
            expd = self.np.diag(self.np.exp(-1j * a * eigenvalues))
            ud = self.np.transpose(self.np.conj(eigenvectors))
            return self.np.matmul(eigenvectors, self.np.matmul(expd, ud))

    def calculate_expectation_state(self, hamiltonian, state, normalize):
        statec = self.np.conj(state)
        hstate = hamiltonian @ state
        ev = self.np.real(self.np.sum(statec * hstate))
        if normalize:
            norm = self.np.sum(self.np.square(self.np.abs(state)))
            ev = ev / norm
        return ev

    def calculate_expectation_density_matrix(self, hamiltonian, state, normalize):
        ev = self.np.real(self.np.trace(hamiltonian @ state))
        if normalize:
            norm = self.np.real(self.np.trace(state))
            ev = ev / norm
        return ev

    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        return self.np.dot(matrix1, matrix2)

    def calculate_hamiltonian_state_product(self, matrix, state):
        rank = len(tuple(state.shape))
        state = self.cast(state)
        if rank == 1:  # vector
            return matrix.dot(state[:, np.newaxis])[:, 0]
        elif rank == 2:  # matrix
            return matrix.dot(state)
        else:
            raise_error(
                ValueError,
                "Cannot multiply Hamiltonian with " "rank-{} tensor.".format(rank),
            )

    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):
        value = self.to_numpy(value)
        target = self.to_numpy(target)
        np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def test_regressions(self, name):
        if name == "test_measurementresult_apply_bitflips":
            return [
                [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            ]
        elif name == "test_probabilistic_measurement":
            return {0: 249, 1: 231, 2: 253, 3: 267}
        elif name == "test_unbalanced_probabilistic_measurement":
            return {0: 171, 1: 148, 2: 161, 3: 520}
        elif name == "test_post_measurement_bitflips_on_circuit":
            return [
                {5: 30},
                {5: 18, 4: 5, 7: 4, 1: 2, 6: 1},
                {4: 8, 2: 6, 5: 5, 1: 3, 3: 3, 6: 2, 7: 2, 0: 1},
            ]
