import collections
import math
from typing import Union

import numpy as np
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
        return np.asarray(x, dtype=dtype, copy=copy if copy else None)

    def is_sparse(self, x):
        from scipy import sparse

        return sparse.issparse(x)

    def to_numpy(self, x):
        if self.is_sparse(x):
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
        state /= math.sqrt(2**nqubits)
        return state

    def plus_density_matrix(self, nqubits):
        state = self.np.ones(2 * (2**nqubits,), dtype=self.dtype)
        state /= 2**nqubits
        return state

    def matrix(self, gate):
        """Convert a gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if callable(_matrix):
            _matrix = _matrix(2 ** len(gate.target_qubits))
        return self.cast(_matrix, dtype=_matrix.dtype)

    def matrix_parametrized(self, gate):
        """Convert a parametrized gate to its matrix representation in the computational basis."""
        name = gate.__class__.__name__
        _matrix = getattr(self.matrices, name)
        if name == "GeneralizedRBS":
            _matrix = _matrix(
                qubits_in=gate.init_args[0],
                qubits_out=gate.init_args[1],
                theta=gate.init_kwargs["theta"],
                phi=gate.init_kwargs["phi"],
            )
        else:
            _matrix = _matrix(*gate.parameters)
        return self.cast(_matrix, dtype=_matrix.dtype)

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
                    np.eye(2 ** len(gate.qubits) - len(gmatrix)), gmatrix
                )
            # Kronecker product with identity is needed to make the
            # original matrix have shape (2**rank x 2**rank)
            eye = np.eye(2 ** (rank - len(gate.qubits)))
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
            # we are using sparse matrices to improve perfomances
            matrix = sparse.csr_matrix(gmatrix).dot(matrix)

        return self.cast(matrix.toarray())

    def apply_gate(self, gate, state, nqubits):
        state = self.np.reshape(state, nqubits * (2,))
        matrix = gate.matrix(self)
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
            state = self.np.concatenate([state[:-1], updates[None]], axis=0)
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
        matrix = gate.matrix(self)
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
            state01 = self.np.concatenate([state00, state01[:, None]], axis=1)
            state10 = self.np.concatenate([state10, state11[None]], axis=0)
            state = self.np.concatenate([state01, state10[None]], axis=0)
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
        state = self.np.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:  # pragma: no cover
            raise_error(
                NotImplementedError,
                "Gate density matrix half call is "
                "not implemented for ``controlled_by``"
                "gates.",
            )
        else:
            matrix = self.np.reshape(matrix, 2 * len(gate.qubits) * (2,))
            left, _ = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.np.einsum(left, state, matrix)
        return self.np.reshape(state, 2 * (2**nqubits,))

    def apply_channel(self, channel, state, nqubits):
        probabilities = channel.coefficients + (1 - np.sum(channel.coefficients),)
        index = self.sample_shots(probabilities, 1)[0]
        if index != len(channel.gates):
            gate = channel.gates[index]
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
            state = self.np.expand_dims(state, q)
            state = (
                self.np.concatenate([self.np.zeros_like(state), state], q)
                if r == 1
                else self.np.concatenate([state, self.np.zeros_like(state)], q)
            )
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
        from qibo.gates import X  # pylint: disable=C0415
        from qibo.quantum_info.linalg_operations import (  # pylint: disable=C0415
            partial_trace,
        )

        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits[0]
        p_0, p_1 = gate.init_kwargs["p_0"], gate.init_kwargs["p_1"]
        trace = partial_trace(state, (q,), backend=self)
        trace = self.np.reshape(trace, 2 * (nqubits - 1) * (2,))
        zero = self.zero_density_matrix(1)
        zero = self.np.tensordot(trace, zero, 0)
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
        from qibo.quantum_info.linalg_operations import (  # pylint: disable=C0415
            partial_trace,
        )

        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits
        lam = gate.init_kwargs["lam"]
        trace = partial_trace(state, q, backend=self)
        trace = self.np.reshape(trace, 2 * (nqubits - len(q)) * (2,))
        identity = self.identity_density_matrix(len(q))
        identity = self.np.reshape(identity, 2 * len(q) * (2,))
        identity = self.np.tensordot(trace, identity, 0)
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

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
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
        elif initial_state is not None:
            initial_state = self.cast(initial_state)
            valid_shape = (
                2 * (2**circuit.nqubits,)
                if circuit.density_matrix
                else (2**circuit.nqubits,)
            )
            if tuple(initial_state.shape) != valid_shape:
                raise_error(
                    ValueError,
                    f"Given initial state has shape {initial_state.shape} instead of "
                    f"the expected {valid_shape}.",
                )

        if circuit.repeated_execution:
            if circuit.measurements or circuit.has_collapse:
                return self.execute_circuit_repeated(circuit, nshots, initial_state)
            else:
                raise_error(
                    RuntimeError,
                    "Attempting to perform noisy simulation with `density_matrix=False` "
                    + "and no Measurement gate in the Circuit. If you wish to retrieve the "
                    + "statistics of the outcomes please include measurements in the circuit, "
                    + "otherwise set `density_matrix=True` to recover the final state.",
                )

        if circuit.accelerators:  # pragma: no cover
            return self.execute_distributed_circuit(circuit, initial_state, nshots)

        try:
            nqubits = circuit.nqubits

            if circuit.density_matrix:
                if initial_state is None:
                    state = self.zero_density_matrix(nqubits)
                else:
                    state = self.cast(initial_state)

                for gate in circuit.queue:
                    state = gate.apply_density_matrix(self, state, nqubits)

            else:
                if initial_state is None:
                    state = self.zero_state(nqubits)
                else:
                    state = self.cast(initial_state)

                for gate in circuit.queue:
                    state = gate.apply(self, state, nqubits)

            if circuit.has_unitary_channel:
                # here we necessarily have `density_matrix=True`, otherwise
                # execute_circuit_repeated would have been called
                if circuit.measurements:
                    circuit._final_state = CircuitResult(
                        state, circuit.measurements, backend=self, nshots=nshots
                    )
                    return circuit._final_state
                else:
                    circuit._final_state = QuantumState(state, backend=self)
                    return circuit._final_state

            else:
                if circuit.measurements:
                    circuit._final_state = CircuitResult(
                        state, circuit.measurements, backend=self, nshots=nshots
                    )
                    return circuit._final_state
                else:
                    circuit._final_state = QuantumState(state, backend=self)
                    return circuit._final_state

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
            )

    def execute_circuits(
        self, circuits, initial_states=None, nshots=1000, processes=None
    ):
        from qibo.parallel import parallel_circuits_execution

        return parallel_circuits_execution(
            circuits, initial_states, nshots, processes, backend=self
        )

    def execute_circuit_repeated(self, circuit, nshots, initial_state=None):
        """
        Execute the circuit `nshots` times to retrieve probabilities, frequencies
        and samples. Note that this method is called only if a unitary channel
        is present in the circuit (i.e. noisy simulation) and `density_matrix=False`, or
        if some collapsing measurement is performed.
        """

        if (
            circuit.has_collapse
            and not circuit.measurements
            and not circuit.density_matrix
        ):
            raise_error(
                RuntimeError,
                "The circuit contains only collapsing measurements (`collapse=True`) but "
                + "`density_matrix=False`. Please set `density_matrix=True` to retrieve "
                + "the final state after execution.",
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
                    samples.append("".join([str(int(s)) for s in sample]))
                for gate in circuit.measurements:
                    gate.result.reset()

        if circuit.density_matrix:  # this implies also it has_collapse
            assert circuit.has_collapse
            final_state = self.cast(np.mean(self.to_numpy(final_states), 0))
            if circuit.measurements:
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

    def execute_distributed_circuit(self, circuit, initial_state=None, nshots=None):
        raise_error(
            NotImplementedError, f"{self} does not support distributed execution."
        )

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
        probs = self.np.sum(self.cast(state, dtype=rtype), axis=unmeasured_qubits)
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
        return self.cast(shots, dtype=shots[0].dtype)

    def samples_to_binary(self, samples, nqubits):
        qrange = np.arange(nqubits - 1, -1, -1, dtype=np.int32)
        return np.mod(np.right_shift(samples[:, None], qrange), 2)

    def samples_to_decimal(self, samples, nqubits):
        ### This is faster just staying @ NumPy.
        qrange = np.arange(nqubits - 1, -1, -1, dtype=np.int32)
        qrange = (2**qrange)[:, None]
        samples = np.asarray(samples.tolist())
        return np.matmul(samples, qrange)[:, 0]

    def calculate_frequencies(self, samples):
        # Samples are a list of strings so there is no advantage in using other backends
        res, counts = np.unique(samples, return_counts=True)
        res = self.to_numpy(res).tolist()
        counts = self.to_numpy(counts).tolist()
        return collections.Counter(dict(zip(res, counts)))

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.np.unique(samples, return_counts=True)
        frequencies[res] += counts
        return frequencies

    def sample_frequencies(self, probabilities, nshots):
        from qibo.config import SHOT_BATCH_SIZE

        nprobs = probabilities / self.np.sum(probabilities)
        frequencies = self.np.zeros(len(nprobs), dtype=self.np.int64)
        for _ in range(nshots // SHOT_BATCH_SIZE):
            frequencies = self.update_frequencies(frequencies, nprobs, SHOT_BATCH_SIZE)
        frequencies = self.update_frequencies(
            frequencies, nprobs, nshots % SHOT_BATCH_SIZE
        )
        return collections.Counter(
            {i: int(f) for i, f in enumerate(frequencies) if f > 0}
        )

    def apply_bitflips(self, noiseless_samples, bitflip_probabilities):
        noiseless_samples = self.cast(noiseless_samples, dtype=noiseless_samples.dtype)
        fprobs = self.cast(bitflip_probabilities, dtype="float64")
        sprobs = self.cast(np.random.random(noiseless_samples.shape), dtype="float64")
        flip_0 = self.cast(sprobs < fprobs[0], dtype=noiseless_samples.dtype)
        flip_1 = self.cast(sprobs < fprobs[1], dtype=noiseless_samples.dtype)
        noisy_samples = noiseless_samples + (1 - noiseless_samples) * flip_0
        noisy_samples = noisy_samples - noiseless_samples * flip_1
        return noisy_samples

    def calculate_vector_norm(self, state, order=2):
        state = self.cast(state)
        return self.np.linalg.norm(state, order)

    def calculate_matrix_norm(self, state, order="nuc"):
        state = self.cast(state)
        return self.np.linalg.norm(state, ord=order)

    def calculate_overlap(self, state1, state2):
        return self.np.abs(
            self.np.sum(self.np.conj(self.cast(state1)) * self.cast(state2))
        )

    def calculate_overlap_density_matrix(self, state1, state2):
        return self.np.trace(
            self.np.matmul(self.np.conj(self.cast(state1)).T, self.cast(state2))
        )

    def calculate_eigenvalues(self, matrix, k: int = 6, hermitian: bool = True):
        if self.is_sparse(matrix):
            log.warning(
                "Calculating sparse matrix eigenvectors because "
                "sparse modules do not provide ``eigvals`` method."
            )
            return self.calculate_eigenvectors(matrix, k=k)[0]
        if hermitian:
            return np.linalg.eigvalsh(matrix)
        return np.linalg.eigvals(matrix)

    def calculate_eigenvectors(self, matrix, k: int = 6, hermitian: bool = True):
        if self.is_sparse(matrix):
            if k < matrix.shape[0]:
                from scipy.sparse.linalg import eigsh

                return eigsh(matrix, k=k, which="SA")
            else:  # pragma: no cover
                matrix = self.to_numpy(matrix)
        if hermitian:
            return np.linalg.eigh(matrix)
        return np.linalg.eig(matrix)

    def calculate_expectation_state(self, hamiltonian, state, normalize):
        statec = self.np.conj(state)
        hstate = hamiltonian @ state
        ev = self.np.real(self.np.sum(statec * hstate))
        if normalize:
            ev /= self.np.sum(self.np.square(self.np.abs(state)))
        return ev

    def calculate_expectation_density_matrix(self, hamiltonian, state, normalize):
        ev = self.np.real(self.np.trace(self.cast(hamiltonian @ state)))
        if normalize:
            norm = self.np.real(self.np.trace(state))
            ev /= norm
        return ev

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.is_sparse(matrix):
            if self.is_sparse(matrix):
                from scipy.sparse.linalg import expm
            else:
                from scipy.linalg import expm
            return expm(-1j * a * matrix)
        expd = self.np.diag(self.np.exp(-1j * a * eigenvalues))
        ud = self.np.transpose(np.conj(eigenvectors))
        return self.np.matmul(eigenvectors, self.np.matmul(expd, ud))

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
            determinant = self.np.linalg.det(matrix)
            if abs(determinant) < precision_singularity:
                return _calculate_negative_power_singular_matrix(
                    matrix, power, precision_singularity, self.np, self
                )

        return fractional_matrix_power(matrix, power)

    def calculate_singular_value_decomposition(self, matrix):
        return self.np.linalg.svd(matrix)

    def calculate_jacobian_matrix(
        self, circuit, parameters=None, initial_state=None, return_complex: bool = True
    ):
        raise_error(
            NotImplementedError,
            "This method is only implemented in backends that allow automatic differentiation, "
            + "e.g. ``PytorchBackend`` and ``TensorflowBackend``.",
        )

    # TODO: remove this method
    def calculate_hamiltonian_matrix_product(self, matrix1, matrix2):
        return matrix1 @ matrix2

    # TODO: remove this method
    def calculate_hamiltonian_state_product(self, matrix, state):
        if len(tuple(state.shape)) > 2:
            raise_error(
                ValueError,
                f"Cannot multiply Hamiltonian with rank-{len(tuple(state.shape))} tensor.",
            )
        return matrix @ state

    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):
        if isinstance(value, CircuitResult) or isinstance(value, QuantumState):
            value = value.state()
        if isinstance(target, CircuitResult) or isinstance(target, QuantumState):
            target = target.state()
        value = self.to_numpy(value)
        target = self.to_numpy(target)
        np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)

    def _test_regressions(self, name):
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


def _calculate_negative_power_singular_matrix(
    matrix, power: Union[float, int], precision_singularity: float, engine, backend
):
    """Calculate negative power of singular matrix."""
    U, S, Vh = backend.calculate_singular_value_decomposition(matrix)
    # cast needed because of different dtypes in `torch`
    S = backend.cast(S)
    S_inv = engine.where(engine.abs(S) < precision_singularity, 0.0, S**power)

    return engine.linalg.inv(Vh) @ backend.np.diag(S_inv) @ engine.linalg.inv(U)
