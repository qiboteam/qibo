"""Module defining the Numpy backend."""

import numpy as np

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
            self.int8,
            self.int32,
            self.int64,
            self.float32,
            self.float64,
            self.complex64,
            self.complex128,
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

    # def matrix_fused(self, fgate):
    #     rank = len(fgate.target_qubits)
    #     matrix = sparse.eye(2**rank)

    #     for gate in fgate.gates:
    #         # transfer gate matrix to numpy as it is more efficient for
    #         # small tensor calculations
    #         # explicit to_numpy see https://github.com/qiboteam/qibo/issues/928
    #         gmatrix = self.to_numpy(gate.matrix(self))
    #         # add controls if controls were instantiated using
    #         # the ``Gate.controlled_by`` method
    #         num_controls = len(gate.control_qubits)
    #         if num_controls > 0:
    #             gmatrix = block_diag(
    #                 np.eye(2 ** len(gate.qubits) - len(gmatrix)), gmatrix
    #             )
    #         # Kronecker product with identity is needed to make the
    #         # original matrix have shape (2**rank x 2**rank)
    #         eye = np.eye(2 ** (rank - len(gate.qubits)))
    #         gmatrix = np.kron(gmatrix, eye)
    #         # Transpose the new matrix indices so that it targets the
    #         # target qubits of the original gate
    #         original_shape = gmatrix.shape
    #         gmatrix = np.reshape(gmatrix, 2 * rank * (2,))
    #         qubits = list(gate.qubits)
    #         indices = qubits + [q for q in fgate.target_qubits if q not in qubits]
    #         indices = np.argsort(indices)
    #         transpose_indices = list(indices)
    #         transpose_indices.extend(indices + rank)
    #         gmatrix = np.transpose(gmatrix, transpose_indices)
    #         gmatrix = np.reshape(gmatrix, original_shape)
    #         # fuse the individual gate matrix to the total ``FusedGate`` matrix
    #         # we are using sparse matrices to improve perfomances
    #         matrix = sparse.csr_matrix(gmatrix).dot(matrix)

    #     return self.cast(matrix.toarray())

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
                    state = gate.apply(self, state, nqubits)
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
