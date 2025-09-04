"""Module defining the Numpy backend."""

import numpy as np
from scipy import sparse
from scipy.linalg import block_diag

from qibo import __version__
from qibo.backends.abstract import Backend
from qibo.backends.npmatrices import NumpyMatrices
from qibo.config import raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState


class NumpyBackend(Backend):
    def __init__(self):
        super().__init__()
        self.engine = np
        self.name = "numpy"
        self.matrices = NumpyMatrices(self.dtype)
        self.tensor_types = np.ndarray
        self.numeric_types += (
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
        self.versions[self.name] = self.engine.__version__

    def set_device(self, device):
        if device != "/CPU:0":
            raise_error(
                ValueError, f"Device {device} is not available for {self} backend."
            )

    def set_threads(self, nthreads):
        if nthreads > 1:
            raise_error(ValueError, "``numpy`` does not support more than one thread.")

    def cast(self, x, dtype=None, copy: bool = False):
        if dtype is None:
            dtype = self.dtype

        if isinstance(x, self.tensor_types):
            return x.astype(dtype, copy=copy)

        if self.is_sparse(x):
            return x.astype(dtype, copy=copy)

        return np.asarray(x, dtype=dtype, copy=copy if copy else None)

    def to_numpy(self, array):
        if self.is_sparse(array):
            return array.toarray()
        return array

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

    def execute_circuit(self, circuit, initial_state=None, nshots=1000):
        if isinstance(initial_state, type(circuit)):
            if not initial_state.density_matrix == circuit.density_matrix:
                raise_error(
                    ValueError,
                    f"""Cannot set circuit with density_matrix {initial_state.density_matrix} as
                      initial state for circuit with density_matrix {circuit.density_matrix}.""",
                )

            if (
                not initial_state.accelerators == circuit.accelerators
            ):  # pragma: no cover
                raise_error(
                    ValueError,
                    f"""Cannot set circuit with accelerators {initial_state.density_matrix} as
                      initial state for circuit with accelerators {circuit.density_matrix}.""",
                )
            return self.execute_circuit(initial_state + circuit, None, nshots)
        elif initial_state is not None:
            initial_state = self.cast(initial_state, dtype=initial_state.dtype)
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
            density_matrix = circuit.density_matrix

            state = (
                self.zero_state(nqubits, density_matrix=density_matrix)
                if initial_state is None
                else self.cast(initial_state)
            )

            print(state.shape)

            for gate in circuit.queue:
                state = gate.apply(self, state, nqubits, density_matrix=density_matrix)
                print(state.shape)

            if circuit.has_unitary_channel:
                # here we necessarily have `density_matrix=True`, otherwise
                # execute_circuit_repeated would have been called
                if circuit.measurements:
                    circuit._final_state = CircuitResult(
                        state, circuit.measurements, backend=self, nshots=nshots
                    )
                    return circuit._final_state

                circuit._final_state = QuantumState(state, backend=self)
                return circuit._final_state

            if circuit.measurements:
                circuit._final_state = CircuitResult(
                    state, circuit.measurements, backend=self, nshots=nshots
                )
                return circuit._final_state

            circuit._final_state = QuantumState(state, backend=self)

            return circuit._final_state

        except self.oom_error:
            raise_error(
                RuntimeError,
                f"State does not fit in {self.device} memory."
                "Please switch the execution device to a "
                "different one using ``qibo.set_device``.",
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
                    state = self.zero_state(nqubits, density_matrix=True)
                else:
                    state = self.cast(initial_state, copy=True)

                for gate in circuit.queue:
                    if gate.symbolic_parameters:
                        gate.substitute_symbols()
                    state = gate.apply(
                        self, state, nqubits, density_matrix=circuit.density_matrix
                    )
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

    def samples_to_decimal(self, samples, nqubits):
        ### This is faster just staying @ NumPy.
        qrange = np.arange(nqubits - 1, -1, -1, dtype=np.int32)
        qrange = (2**qrange)[:, None]
        samples = np.asarray(samples.tolist())
        return np.matmul(samples, qrange)[:, 0]
