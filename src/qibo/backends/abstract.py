import abc
import collections
from typing import Optional, Union

from qibo.backends import einsum_utils
from qibo.config import log, raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


class Backend(abc.ABC):
    def __init__(self):
        super().__init__()
        self.name = "backend"
        self.platform = None

        self.precision = "double"
        self.dtype = "complex128"
        self.matrices = None

        self.device = "/CPU:0"
        self.nthreads = 1
        self.supports_multigpu = False
        self.oom_error = MemoryError

    def __reduce__(self):
        """Allow pickling backend objects that have references to modules."""
        return self.__class__, tuple()

    def __repr__(self):
        if self.platform is None:
            return self.name
        else:
            return f"{self.name} ({self.platform})"

    @property
    @abc.abstractmethod
    def qubits(self) -> Optional[list[Union[int, str]]]:  # pragma: no cover
        """Return the qubit names of the backend. If :class:`SimulationBackend`, return None."""
        raise_error(NotImplementedError)

    @property
    @abc.abstractmethod
    def connectivity(
        self,
    ) -> Optional[list[tuple[Union[int, str], Union[int, str]]]]:  # pragma: no cover
        """Return the available qubit pairs of the backend. If :class:`SimulationBackend`, return None."""
        raise_error(NotImplementedError)

    @property
    @abc.abstractmethod
    def natives(self) -> Optional[list[str]]:  # pragma: no cover
        """Return the native gates of the backend. If :class:`SimulationBackend`, return None."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_precision(self, precision):  # pragma: no cover
        """Set complex number precision.

        Args:
            precision (str): 'single' or 'double'.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_device(self, device):  # pragma: no cover
        """Set simulation device.

        Args:
            device (str): Device such as '/CPU:0', '/GPU:0', etc.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def set_threads(self, nthreads):  # pragma: no cover
        """Set number of threads for CPU simulation.

        Args:
            nthreads (int): Number of threads.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def cast(self, x, copy=False):  # pragma: no cover
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            copy (bool): If ``True`` a copy of the object is created in memory.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def to_numpy(self, x):  # pragma: no cover
        """Cast a given array to numpy."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def compile(self, func):  # pragma: no cover
        """Compile the given method.

        Available only for the tensorflow backend.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_eigenvectors(
        self, matrix, k: int = 6, hermitian: bool = True
    ):  # pragma: no cover
        """Calculate eigenvectors of a matrix."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_exp(
        self, a, matrix, eigenvectors=None, eigenvalues=None
    ):  # pragma: no cover
        """Calculate matrix exponential of a matrix.
        If the eigenvectors and eigenvalues are given the matrix diagonalization is
        used for exponentiation.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_matrix_power(
        self, matrix, power: Union[float, int], precision_singularity: float = 1e-14
    ):  # pragma: no cover
        """Calculate the (fractional) ``power`` :math:`\\alpha` of ``matrix`` :math:`A`,
        i.e. :math:`A^{\\alpha}`.

        .. note::
            For the ``pytorch`` backend, this method relies on a copy of the original tensor.
            This may break the gradient flow. For the GPU backends (i.e. ``cupy`` and
            ``cuquantum``), this method falls back to CPU whenever ``power`` is not
            an integer.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def calculate_jacobian_matrix(
        self, circuit, parameters, initial_state=None, return_complex: bool = True
    ):  # pragma: no cover
        """Calculate the Jacobian matrix of ``circuit`` with respect to varables ``params``."""
        raise_error(NotImplementedError)

    def zero_state(self, nqubits):
        state = self.zeros(2**nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits):
        state = self.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state[0, 0] = 1
        return state

    def identity_density_matrix(self, nqubits, normalize: bool = True):
        state = self.eye(2**nqubits, dtype=self.dtype)
        if normalize is True:
            state /= 2**nqubits
        return state

    def plus_state(self, nqubits):
        state = self.ones(2**nqubits, dtype=self.dtype)
        state /= self.sqrt(2**nqubits)
        return state

    def plus_density_matrix(self, nqubits):
        state = self.ones(2 * (2**nqubits,), dtype=self.dtype)
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
        matrix = self.sparse_eye(2**rank)

        for gate in fgate.gates:
            # transfer gate matrix to numpy as it is more efficient for
            # small tensor calculations
            # explicit to_numpy see https://github.com/qiboteam/qibo/issues/928
            # --> not sure whether going to numpy is still optimal in every case
            gmatrix = self.to_numpy(gate.matrix(self))
            # add controls if controls were instantiated using
            # the ``Gate.controlled_by`` method
            num_controls = len(gate.control_qubits)
            if num_controls > 0:
                gmatrix = self.block_diag(
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
            matrix = self.sparse_csr_matrix(gmatrix).dot(matrix)

        return self.cast(self.to_dense(matrix))

    def apply_gate(self, gate, state, nqubits):
        state = self.reshape(state, nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, targets = einsum_utils.control_order(gate, nqubits)
            state = self.transpose(state, order)
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = self.reshape(state, (2**ncontrol,) + nactive * (2,))
            opstring = einsum_utils.apply_gate_string(targets, nactive)
            updates = self.einsum(opstring, state[-1], matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = self.concatenate([state[:-1], updates[None]], axis=0)
            state = self.reshape(state, nqubits * (2,))
            # Put qubit indices back to their proper places
            state = self.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = self.einsum(opstring, state, matrix)
        return self.reshape(state, (2**nqubits,))

    def apply_gate_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:
            matrix = self.reshape(matrix, 2 * len(gate.target_qubits) * (2,))
            matrixc = self.conj(matrix)
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            n = 2**ncontrol

            order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
            state = self.transpose(state, order)
            state = self.reshape(state, 2 * (n,) + 2 * nactive * (2,))

            leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(
                targets, nactive
            )
            state01 = state[: n - 1, n - 1]
            state01 = self.einsum(rightc, state01, matrixc)
            state10 = state[n - 1, : n - 1]
            state10 = self.einsum(leftc, state10, matrix)

            left, right = einsum_utils.apply_gate_density_matrix_string(
                targets, nactive
            )
            state11 = state[n - 1, n - 1]
            state11 = self.einsum(right, state11, matrixc)
            state11 = self.einsum(left, state11, matrix)

            state00 = state[range(n - 1)]
            state00 = state00[:, range(n - 1)]
            state01 = self.concatenate([state00, state01[:, None]], axis=1)
            state10 = self.concatenate([state10, state11[None]], axis=0)
            state = self.concatenate([state01, state10[None]], axis=0)
            state = self.reshape(state, 2 * nqubits * (2,))
            state = self.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))
            matrixc = self.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.einsum(right, state, matrixc)
            state = self.einsum(left, state, matrix)
        return self.reshape(state, 2 * (2**nqubits,))

    def apply_gate_half_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = self.reshape(state, 2 * nqubits * (2,))
        matrix = gate.matrix(self)
        if gate.is_controlled_by:  # pragma: no cover
            raise_error(
                NotImplementedError,
                "Gate density matrix half call is "
                "not implemented for ``controlled_by``"
                "gates.",
            )
        else:
            matrix = self.reshape(matrix, 2 * len(gate.qubits) * (2,))
            left, _ = einsum_utils.apply_gate_density_matrix_string(
                gate.qubits, nqubits
            )
            state = self.einsum(left, state, matrix)
        return self.reshape(state, 2 * (2**nqubits,))

    def apply_channel(self, channel, state, nqubits):
        probabilities = channel.coefficients + (1 - self.sum(channel.coefficients),)
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
            state = self.expand_dims(state, q)
            state = (
                self.concatenate([self.zeros(state.shape, dtype=state.dtype), state], q)
                if r == 1
                else self.concatenate(
                    [state, self.zeros(state.shape, dtype=state.dtype)], q
                )
            )
        return state

    def collapse_state(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = self.samples_to_binary(shot, len(qubits))[0]
        state = self.reshape(state, nqubits * (2,))
        order = list(qubits) + [q for q in range(nqubits) if q not in qubits]
        state = self.transpose(state, order)
        subshape = (2 ** len(qubits),) + (nqubits - len(qubits)) * (2,)
        state = self.reshape(state, subshape)[int(shot)]
        if normalize:
            norm = self.sqrt(self.sum(self.abs(state) ** 2))
            state = state / norm
        state = self._append_zeros(state, qubits, binshot)
        return self.reshape(state, shape)

    def collapse_density_matrix(self, state, qubits, shot, nqubits, normalize=True):
        state = self.cast(state)
        shape = state.shape
        binshot = list(self.samples_to_binary(shot, len(qubits))[0])
        order = list(qubits) + [q + nqubits for q in qubits]
        order.extend(q for q in range(nqubits) if q not in qubits)
        order.extend(q + nqubits for q in range(nqubits) if q not in qubits)
        state = self.reshape(state, 2 * nqubits * (2,))
        state = self.transpose(state, order)
        subshape = 2 * (2 ** len(qubits),) + 2 * (nqubits - len(qubits)) * (2,)
        state = self.reshape(state, subshape)[int(shot), int(shot)]
        n = 2 ** (len(state.shape) // 2)
        if normalize:
            norm = self.trace(self.reshape(state, (n, n)))
            state = state / norm
        qubits = qubits + [q + nqubits for q in qubits]
        state = self._append_zeros(state, qubits, 2 * binshot)
        return self.reshape(state, shape)

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
        trace = self.reshape(trace, 2 * (nqubits - 1) * (2,))
        zero = self.zero_density_matrix(1)
        zero = self.tensordot(trace, zero, 0)
        order = list(range(2 * nqubits - 2))
        order.insert(q, 2 * nqubits - 2)
        order.insert(q + nqubits, 2 * nqubits - 1)
        zero = self.reshape(self.transpose(zero, order), shape)
        state = (1 - p_0 - p_1) * state + p_0 * zero
        return state + p_1 * self.apply_gate_density_matrix(X(q), zero, nqubits)

    def thermal_error_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        shape = state.shape
        state = self.apply_gate(gate, self.ravel(state), 2 * nqubits)
        return self.reshape(state, shape)

    def depolarizing_error_density_matrix(self, gate, state, nqubits):
        from qibo.quantum_info.linalg_operations import (  # pylint: disable=C0415
            partial_trace,
        )

        state = self.cast(state)
        shape = state.shape
        q = gate.target_qubits
        lam = gate.init_kwargs["lam"]
        trace = partial_trace(state, q, backend=self)
        trace = self.reshape(trace, 2 * (nqubits - len(q)) * (2,))
        identity = self.identity_density_matrix(len(q))
        identity = self.reshape(identity, 2 * len(q) * (2,))
        identity = self.tensordot(trace, identity, 0)
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
        identity = self.reshape(self.transpose(identity, order), shape)
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
            final_state = self.cast(self.mean(self.to_numpy(final_states), 0))
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

    def calculate_symbolic(
        self, state, nqubits, decimals=5, cutoff=1e-10, max_terms=20
    ):
        state = self.to_numpy(state)
        terms = []
        for i in self.nonzero(state)[0]:
            b = bin(i)[2:].zfill(nqubits)
            if self.abs(state[i]) >= cutoff:
                x = self.round(state[i], decimals)
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
        indi, indj = self.nonzero(state)
        for i, j in zip(indi, indj):
            bi = bin(i)[2:].zfill(nqubits)
            bj = bin(j)[2:].zfill(nqubits)
            if self.abs(state[i, j]) >= cutoff:
                x = self.round(state[i, j], decimals)
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
        return self.transpose(probs, [reduced.get(i) for i in qubits])

    def calculate_probabilities(self, state, qubits, nqubits):
        rtype = self.real(state).dtype
        unmeasured_qubits = tuple(i for i in range(nqubits) if i not in qubits)
        state = self.reshape(self.abs(state) ** 2, nqubits * (2,))
        probs = self.sum(self.cast(state, dtype=rtype), axis=unmeasured_qubits)
        return self.ravel(self._order_probabilities(probs, qubits, nqubits))

    def calculate_probabilities_density_matrix(self, state, qubits, nqubits):
        order = tuple(sorted(qubits))
        order += tuple(i for i in range(nqubits) if i not in qubits)
        order = order + tuple(i + nqubits for i in order)
        shape = 2 * (2 ** len(qubits), 2 ** (nqubits - len(qubits)))
        state = self.reshape(state, 2 * nqubits * (2,))
        state = self.reshape(self.transpose(state, order), shape)
        probs = self.abs(self.einsum("abab->a", state))
        probs = self.reshape(probs, len(qubits) * (2,))
        return self.ravel(self._order_probabilities(probs, qubits, nqubits))

    def set_seed(self, seed):
        self.seed(seed)

    def sample_shots(self, probabilities, nshots):
        return self.random_choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    def aggregate_shots(self, shots):
        return self.cast(shots, dtype=shots[0].dtype)

    def samples_to_binary(self, samples, nqubits):
        qrange = self.arange(nqubits - 1, -1, -1, dtype=self.get_dtype("int32"))
        return self.mod(self.right_shift(samples[:, None], qrange), 2)

    def samples_to_decimal(self, samples, nqubits):
        ### This is faster just staying @ NumPy.
        ## --> should we keep this method abstract then?
        qrange = self.arange(nqubits - 1, -1, -1, dtype=self.get_dtype("int32"))
        qrange = (2**qrange)[:, None]
        samples = self.cast(samples, dtype=self.get_dtype("int32"))
        return self.matmul(samples, qrange)[:, 0]

    def calculate_frequencies(self, samples):
        # Samples are a list of strings so there is no advantage in using other backends
        res, counts = self.unique(samples, return_counts=True)
        res = self.to_numpy(res).tolist()
        counts = self.to_numpy(counts).tolist()
        return collections.Counter(dict(zip(res, counts)))

    def update_frequencies(self, frequencies, probabilities, nsamples):
        samples = self.sample_shots(probabilities, nsamples)
        res, counts = self.unique(samples, return_counts=True)
        frequencies[res] += counts
        return frequencies

    def sample_frequencies(self, probabilities, nshots):
        from qibo.config import SHOT_BATCH_SIZE

        nprobs = probabilities / self.sum(probabilities)
        frequencies = self.zeros(len(nprobs), dtype=self.get_dtype("int64"))
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
        sprobs = self.cast(self.rand(*noiseless_samples.shape), dtype="float64")
        flip_0 = self.cast(sprobs < fprobs[0], dtype=noiseless_samples.dtype)
        flip_1 = self.cast(sprobs < fprobs[1], dtype=noiseless_samples.dtype)
        noisy_samples = noiseless_samples + (1 - noiseless_samples) * flip_0
        noisy_samples = noisy_samples - noiseless_samples * flip_1
        return noisy_samples

    def calculate_norm(self, state, order=2):
        state = self.cast(state)
        return self.linalg_norm(state, order)

    def calculate_norm_density_matrix(self, state, order="nuc"):
        state = self.cast(state)
        return self.linalg_norm(state, ord=order)

    def calculate_overlap(self, state1, state2):
        return self.abs(self.sum(self.conj(self.cast(state1)) * self.cast(state2)))

    def calculate_overlap_density_matrix(self, state1, state2):
        return self.trace(
            self.matmul(self.conj(self.cast(state1)).T, self.cast(state2))
        )

    def calculate_eigenvalues(self, matrix, k: int = 6, hermitian: bool = True):
        if self.is_sparse(matrix):
            log.warning(
                "Calculating sparse matrix eigenvectors because "
                "sparse modules do not provide ``eigvals`` method."
            )
            return self.calculate_eigenvectors(matrix, k=k)[0]
        if hermitian:
            return self.eigvalsh(matrix)
        return self.eigvals(matrix)

    def calculate_eigenvectors(self, matrix, k: int = 6, hermitian: bool = True):
        if self.is_sparse(matrix):
            if k < matrix.shape[0]:
                return self.sparse_eigsh(matrix, k=k, which="SA")
            else:  # pragma: no cover
                matrix = self.to_dense(matrix)
        if hermitian:
            return self.eigh(matrix)
        return self.eig(matrix)

    def calculate_matrix_exp(self, a, matrix, eigenvectors=None, eigenvalues=None):
        if eigenvectors is None or self.is_sparse(matrix):
            if self.is_sparse(matrix):
                return self.sparse_expm(-1j * a * matrix)
            return self.expm(-1j * a * matrix)
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
                return self._calculate_negative_power_singular_matrix(
                    matrix, power, precision_singularity
                )

        return self.fractional_matrix_power(matrix, power)

    def _calculate_negative_power_singular_matrix(
        self, matrix, power: Union[float, int], precision_singularity: float
    ):
        """Calculate negative power of singular matrix."""
        U, S, Vh = self.calculate_singular_value_decomposition(matrix)
        # cast needed because of different dtypes in `torch`
        S = self.cast(S)
        S_inv = self.where(self.abs(S) < precision_singularity, 0.0, S**power)

        return self.inverse(Vh) @ self.diag(S_inv) @ self.inverse(U)

    def calculate_expectation_state(self, hamiltonian, state, normalize):
        statec = self.conj(state)
        hstate = hamiltonian @ state
        ev = self.real(self.sum(statec * hstate))
        if normalize:
            ev /= self.sum(self.square(self.abs(state)))
        return ev

    def calculate_expectation_density_matrix(self, hamiltonian, state, normalize):
        ev = self.real(self.trace(self.cast(hamiltonian @ state)))
        if normalize:
            norm = self.real(self.trace(state))
            ev /= norm
        return ev

    def calculate_singular_value_decomposition(self, matrix):
        return self.linalg_svd(matrix)

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

    def assert_circuitclose(self, circuit, target_circuit, rtol=1e-7, atol=0.0):
        value = self.execute_circuit(circuit)._state
        target = self.execute_circuit(target_circuit)._state
        self.assert_allclose(value, target, rtol=rtol, atol=atol)

    # --------------------------------------------------------------------------------------------
    # New methods introduced by the refactor:
    # in my view this might be considered as some sort of the core of the backend,
    # i.e. the computation engine that defines how the single small operations
    # are performed and it is going to be completely abstract. All the methods defined
    # above are possibly going to be combination of the core methods below and, therefore,
    # directly implemented in the abstract backend.

    # array creation and manipulation
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def array(self, x: Union[list, tuple], **kwargs):
        """Construct a native array of the backend starting from a `list` or `tuple`.

        Args:
            x (list | tuple): input list or tuple.
            kwargs: keyword argument passed to the `Backend.cast` method.
        """
        return self.cast(x, **kwargs)

    @abc.abstractmethod
    def eye(self, *args, **kwargs):
        """Numpy-like eye: https://numpy.org/devdocs/reference/generated/numpy.eye.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def zeros(self, *args, **kwargs):
        """Numpy-like zeros: https://numpy.org/devdocs/reference/generated/numpy.zeros.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def ones(self, *args, **kwargs):
        """Numpy-like ones: https://numpy.org/devdocs/reference/generated/numpy.ones.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def arange(self, *args, **kwargs):
        """Numpy-like arange: https://numpy.org/devdocs/reference/generated/numpy.arange.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def copy(self, *args, **kwargs):
        """Numpy-like copy: https://numpy.org/devdocs/reference/generated/numpy.copy.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def reshape(self, *args, **kwargs):
        """Numpy-like reshape: https://numpy.org/devdocs/reference/generated/numpy.reshape.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def ravel(self, *args, **kwargs):
        """Numpy-like ravel: https://numpy.org/devdocs/reference/generated/numpy.ravel.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def transpose(self, *args, **kwargs):
        """Numpy-like transpose: https://numpy.org/devdocs/reference/generated/numpy.transpose.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def concatenate(self, *args, **kwargs):
        """Numpy-like concatenate: https://numpy.org/devdocs/reference/generated/numpy.concatenate.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def expand_dims(self, *args, **kwargs):
        """Numpy-like expand_dims: https://numpy.org/devdocs/reference/generated/numpy.expand_dims.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def squeeze(self, *args, **kwargs):
        """Numpy-like squeeze: https://numpy.org/devdocs/reference/generated/numpy.squeeze.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def stack(self, *args, **kwargs):
        """Numpy-like stack: https://numpy.org/devdocs/reference/generated/numpy.stack.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def vstack(self, *args, **kwargs):
        """Numpy-like vstack: https://numpy.org/devdocs/reference/generated/numpy.vstack.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def unique(self, *args, **kwargs):
        """Numpy-like unique: https://numpy.org/devdocs/reference/generated/numpy.unique.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def where(self, *args, **kwargs):
        """Numpy-like where: https://numpy.org/doc/stable/reference/generated/numpy.where.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def flip(self, *args, **kwargs):
        """Numpy-like flip: https://numpy.org/doc/stable/reference/generated/numpy.flip.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def swapaxes(self, *args, **kwargs):
        """Numpy-like swapaxes: https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def diagonal(self, *args, **kwargs):
        """Numpy-like diagonal: https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def nonzero(self, *args, **kwargs):
        """Numpy-like nonzero: https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def sign(self, *args, **kwargs):
        """Numpy-like element-wise sign function: https://numpy.org/doc/stable/reference/generated/numpy.sign.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def delete(a, obj, axis=None):
        """Numpy-like delete function: https://numpy.org/doc/stable/reference/generated/numpy.delete.html"""
        raise NotImplementedError

    # linear algebra
    # ^^^^^^^^^^^^^^

    @abc.abstractmethod
    def einsum(self, *args, **kwargs):
        """Numpy-like einsum: https://numpy.org/devdocs/reference/generated/numpy.einsum.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def matmul(self, *args, **kwargs):
        """Numpy-like matmul: https://numpy.org/devdocs/reference/generated/numpy.matmul.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def multiply(self, *args, **kwargs):
        """Numpy-like multiply: https://numpy.org/doc/stable/reference/generated/numpy.multiply.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def prod(self, *args, **kwargs):
        """Numpy-like prod: https://numpy.org/doc/stable/reference/generated/numpy.prod.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def tensordot(self, *args, **kwargs):
        """Numpy-like tensordot: https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def kron(self, *args, **kwargs):
        """Numpy-like kron: https://numpy.org/doc/stable/reference/generated/numpy.kron.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def outer(self, *args, **kwargs):
        """Numpy-like outer: https://numpy.org/doc/stable/reference/generated/numpy.outer.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def diag(self, *args, **kwargs):
        """Numpy-like diag: https://numpy.org/devdocs/reference/generated/numpy.diag.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def trace(self, *args, **kwargs):
        """Numpy-like trace: https://numpy.org/devdocs/reference/generated/numpy.trace.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def linalg_svd(self, *args, **kwargs):
        """Numpy-like linalg.svd: https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def linalg_norm(self, *args, **kwargs):
        """Numpy-like linalg.norm: https://numpy.org/devdocs/reference/generated/numpy.linalg.norm.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def det(self, *args, **kwargs):
        """Numpy-like matrix determinant: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def qr(self, *args, **kwargs):
        """Numpy-like linear algebra QR decomposition: https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self, *args, **kwargs):
        """Numpy-like linear algebra inverse: https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eigvalsh(self, *args, **kwargs):
        """Numpy-like eigvalsh: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eigvals(self, *args, **kwargs):
        """Eigenvalues of a matrix: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eigh(self, *args, **kwargs):
        """Numpy-like eigvals: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def eig(self, *args, **kwargs):
        """Numpy-like eig: https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def expm(self, *args, **kwargs):
        """Scipy-like expm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def fractional_matrix_power(self, A, t):
        """Scipy-like fractional_matrix_power: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.fractional_matrix_power.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def block_diag(self, *args, **kwargs):
        """Scipy-like block_diag: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html"""
        raise NotImplementedError

    # sparse
    # ^^^^^^

    @abc.abstractmethod
    def is_sparse(self, x):  # pragma: no cover
        """Scipy-like issparse: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.issparse.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sparse_csr_matrix(self, *args, **kwargs):
        """Scipy-like sparse csr matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def sparse_eye(self, *args, **kwargs):
        """Scipy-like sparse eye: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def to_dense(self, *args, **kwargs):
        """Scipy-like sparse to dense: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def sparse_expm(self, A):
        """Scipy-like sparse expm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def sparse_eigsh(self, *args, **kwargs):
        """Scipy-like sparse eigsh: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html"""
        raise NotImplementedError

    # randomization
    # ^^^^^^^^^^^^^

    @abc.abstractmethod
    def random_choice(self, *args, **kwargs):
        """Numpy-like random.choice: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def seed(self, *args, **kwargs):
        """Numpy-like random seed: https://numpy.org/devdocs/reference/random/generated/numpy.random.seed.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def permutation(self, *args, **kwargs):
        """Numpy-like random permutation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def multinomial(self, *args, **kwargs):
        """Numpy-like multinomial: https://numpy.org/doc/2.0/reference/random/generated/numpy.random.multinomial.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def default_rng(self, *args, **kwargs):
        """Numpy-like random default_rng: https://numpy.org/doc/stable/reference/random/generator.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def rand(self, *args, **kwargs):
        """Numpy-like random rand: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html"""
        raise NotImplementedError

    # logical operations
    # ^^^^^^^^^^^^^^^^^^

    @abc.abstractmethod
    def less(self, *args, **kwargs):
        """Numpy-like less: https://numpy.org/doc/stable/reference/generated/numpy.less.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def any(self, *args, **kwargs):
        """Numpy-like any: https://numpy.org/doc/stable/reference/generated/numpy.any.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def allclose(self, *args, **kwargs):
        """Numpy-like allclose: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def right_shift(self, *args, **kwargs):
        """Numpy-like element-wise right shift: https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html"""
        raise NotImplementedError

    # mathematical operations
    # ^^^^^^^^^^^^^^^^^^^^^^^

    @abc.abstractmethod
    def sum(self, *args, **kwargs):
        """Numpy-like sum: https://numpy.org/devdocs/reference/generated/numpy.sum.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def conj(self, *args, **kwargs):
        """Numpy-like conj: https://numpy.org/devdocs/reference/generated/numpy.conj.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def exp(self, *args, **kwargs):
        """Numpy-like exp: https://numpy.org/devdocs/reference/generated/numpy.exp.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def log(self, *args, **kwargs):
        """Numpy-like log: https://numpy.org/doc/stable/reference/generated/numpy.log.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def log2(self, *args, **kwargs):
        """Numpy-like log2: https://numpy.org/doc/stable/reference/generated/numpy.log2.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def real(self, *args, **kwargs):
        """Numpy-like real: https://numpy.org/devdocs/reference/generated/numpy.real.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def imag(self, *args, **kwargs):
        """Numpy-like imag: https://numpy.org/doc/stable/reference/generated/numpy.imag.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def abs(self, *args, **kwargs):
        """Numpy-like abs: https://numpy.org/devdocs/reference/generated/numpy.abs.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def pow(self, *args, **kwargs):
        """Numpy-like element-wise power: https://numpy.org/doc/stable/reference/generated/numpy.power.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def square(self, *args, **kwargs):
        """Numpy-like element-wise square: https://numpy.org/doc/stable/reference/generated/numpy.square.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def sqrt(self, *args, **kwargs):
        """Numpy-like sqrt: https://numpy.org/devdocs/reference/generated/numpy.sqrt.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, *args, **kwargs):
        """Numpy-like mean: https://numpy.org/doc/stable/reference/generated/numpy.mean.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def std(self, *args, **kwargs):
        """Numpy-like standard deviation: https://numpy.org/doc/stable/reference/generated/numpy.std.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def cos(self, *args, **kwargs):
        """Numpy-like cos: https://numpy.org/devdocs/reference/generated/numpy.cos.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sin(self, *args, **kwargs):
        """Numpy-like sin: https://numpy.org/devdocs/reference/generated/numpy.sin.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def arccos(self, *args, **kwargs):
        """Numpy-like arccos: https://numpy.org/doc/stable/reference/generated/numpy.arccos.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def arctan2(self, *args, **kwargs):
        """Numpy-like arctan2: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def angle(self, *args, **kwargs):
        """Numpy-like angle: https://numpy.org/doc/stable/reference/generated/numpy.angle.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def mod(self, *args, **kwargs):
        """Numpy-like element-wise mod: https://numpy.org/doc/stable/reference/generated/numpy.mod.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def round(self, *args, **kwargs):
        """Numpy-like element-wise round: https://numpy.org/doc/stable/reference/generated/numpy.round.html"""
        raise NotImplementedError

    # misc
    # ^^^^

    @abc.abstractmethod
    def sort(self, *args, **kwargs):
        """Numpy-like sort: https://numpy.org/doc/stable/reference/generated/numpy.sort.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def argsort(self, *args, **kwargs):
        """Numpy-like argsort: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def count_nonzero(self, *args, **kwargs):
        """Numpy-like count_nonzero: https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def finfo(self, *args, **kwargs):
        """Numpy-like finfo: https://numpy.org/doc/stable/reference/generated/numpy.finfo.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def device(
        self,
    ):
        """Computation device, e.g. CPU, GPU, ..."""
        raise NotImplementedError

    @abc.abstractmethod
    def __version__(
        self,
    ):
        """Version of the backend engine."""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def get_dtype(self, type_name: str):
        """Backend engine dtype"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def assert_allclose(self, *args, **kwargs):
        """Numpy-like testing.assert_allclose: https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html"""
        raise NotImplementedError

    # Optimization
    # ^^^^^^^^^^^^^

    @abc.abstractmethod
    def jacobian(self, *args, **kwargs):
        """Compute the Jacobian matrix"""
        raise NotImplementedError
