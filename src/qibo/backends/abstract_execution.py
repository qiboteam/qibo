import abc
import collections
from typing import Generic, Optional, TypeVar, Union

from qibo.backends import einsum_utils
from qibo.config import raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState

array = Generic[TypeVar("array")]


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
    def cast(self, x, dtype=None, copy=False):  # pragma: no cover
        """Cast an object as the array type of the current backend.

        Args:
            x: Object to cast to array.
            dtype: native backend array dtype.
            copy (bool): If ``True`` a copy of the object is created in memory.
        """
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def to_numpy(self, x: array):  # pragma: no cover
        """Cast a given array to numpy."""
        raise_error(NotImplementedError)

    # initial state
    # ^^^^^^^^^^^^^

    def zero_state(self, nqubits: int) -> array:
        state = self.zeros(2**nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits):
        state = self.zeros(2 * (2**nqubits,), dtype=self.dtype)
        state[0, 0] = 1
        return state

    # matrix representation
    # ^^^^^^^^^^^^^^^^^^^^^

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

    # gate application
    # ^^^^^^^^^^^^^^^^

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

    # circuit execution
    # ^^^^^^^^^^^^^^^^^

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
            final_state = self.mean(final_states, axis=0)
            if circuit.measurements:
                final_result = CircuitResult(
                    final_state,
                    circuit.measurements,
                    backend=self,
                    samples=self.concatenate(results, axis=0),
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
                samples=self.concatenate(results, axis=0),
                nshots=nshots,
            )
            final_result._repeated_execution_frequencies = self.calculate_frequencies(
                samples
            )
            circuit._final_state = final_result
            return final_result

    @abc.abstractmethod
    def execute_distributed_circuit(self, circuit, initial_state=None, nshots=None):
        raise_error(NotImplementedError)

    # shots and freq sampling
    # ^^^^^^^^^^^^^^^^^^^^^^^

    def calculate_frequencies(self, samples):
        res, counts = self.unique(samples, return_counts=True)
        res = self.to_numpy(res).tolist()
        # I would not cast the counts to a list, but rather keep the backend array dtype
        counts = self.to_numpy(counts).tolist()
        return collections.Counter(dict(zip(res, counts)))

    def sample_shots(self, probabilities, nshots):
        return self.random_choice(
            range(len(probabilities)), size=nshots, p=probabilities
        )

    # core methods
    # ^^^^^^^^^^^^

    # array creation and manipulation
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    @abc.abstractmethod
    def zeros(self, *args, **kwargs):
        """Numpy-like zeros: https://numpy.org/devdocs/reference/generated/numpy.zeros.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def eye(self, *args, **kwargs):
        """Numpy-like eye: https://numpy.org/devdocs/reference/generated/numpy.eye.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def reshape(self, *args, **kwargs):
        """Numpy-like reshape: https://numpy.org/devdocs/reference/generated/numpy.reshape.html"""
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
    def argsort(self, *args, **kwargs):
        """Numpy-like argsort: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def unique(self, *args, **kwargs):
        """Numpy-like unique: https://numpy.org/devdocs/reference/generated/numpy.unique.html"""
        raise NotImplementedError

    # basic math
    # ^^^^^^^^^^

    @abc.abstractmethod
    def conj(self, *args, **kwargs):
        """Numpy-like conj: https://numpy.org/devdocs/reference/generated/numpy.conj.html"""
        raise_error(NotImplementedError)

    @abc.abstractmethod
    def sum(self, *args, **kwargs):
        """Numpy-like sum: https://numpy.org/devdocs/reference/generated/numpy.sum.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, *args, **kwargs):
        """Numpy-like mean: https://numpy.org/doc/stable/reference/generated/numpy.mean.html"""
        raise NotImplementedError

    # linear algebra
    # ^^^^^^^^^^^^^^

    # shall we keep only einsum
    @abc.abstractmethod
    def einsum(self, *args, **kwargs):
        """Numpy-like einsum: https://numpy.org/devdocs/reference/generated/numpy.einsum.html"""
        raise NotImplementedError

    # this can probably be done through einsum
    @abc.abstractmethod
    def kron(self, *args, **kwargs):
        """Numpy-like kron: https://numpy.org/doc/stable/reference/generated/numpy.kron.html"""
        raise NotImplementedError

    @abc.abstractmethod
    def block_diag(self, *args, **kwargs):
        """Scipy-like block_diag: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html"""
        raise NotImplementedError

    # sparse utils
    # ^^^^^^^^^^^^

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

    # random utils
    # ^^^^^^^^^^^^

    @abc.abstractmethod
    def random_choice(self, *args, **kwargs):
        """Numpy-like random.choice: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html"""
        raise_error(NotImplementedError)
