import numpy as np
from qibo.config import raise_error
from qibo.gates import FusedGate
from qibo.engines import einsum_utils
from qibo.engines.abstract import Simulator
from qibo.engines.matrices import Matrices


class NumpyEngine(Simulator):

    def __init__(self):
        super().__init__()
        self.name = "numpy"
        self.matrices = Matrices(self.dtype)

    def set_threads(self, nthreads):
        if nthreads > 1:
            raise_error(ValueError, "numpy does not support more than one thread.")

    def to_numpy(self, x):
        return x

    def zero_state(self, nqubits):
        """Generate |000...0> state as an array."""
        state = np.zeros(2 ** nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def asmatrix_fused(self, fgate):
        rank = len(fgate.target_qubits)
        matrix = np.eye(2 ** rank, dtype=self.dtype)
        for gate in fgate.gates:
            # transfer gate matrix to numpy as it is more efficient for
            # small tensor calculations
            gmatrix = self.asmatrix(gate)
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

    def asmatrix_special(self, gate):
        if isinstance(gate, FusedGate):
            return self.asmatrix_fused(gate)
        else:
            raise_error(NotImplementedError)

    def control_matrix(self, gate):
        if len(gate.control_qubits) > 1:
            raise_error(NotImplementedError, "Cannot calculate controlled "
                                             "unitary for more than two "
                                             "control qubits.")
        matrix = self.asmatrix(gate)
        shape = matrix.shape
        if shape != (2, 2):
            raise_error(ValueError, "Cannot use ``control_unitary`` method on "
                                    "gate matrix of shape {}.".format(shape))
        zeros = np.zeros((2, 2), dtype=self.dtype)
        part1 = np.concatenate([np.eye(2, dtype=self.dtype), zeros], axis=0)
        part2 = np.concatenate([zeros, matrix], axis=0)
        return np.concatenate([part1, part2], axis=1)

    def apply_gate(self, gate, state, nqubits):
        state = np.reshape(state, nqubits * (2,))
        if gate.is_controlled_by:
            matrix = np.reshape(self.asmatrix(gate), 2  * len(gate.target_qubits) * (2,))
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, targets = einsum_utils.control_order(gate, nqubits)
            state = np.transpose(state, order)
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = np.reshape(state, (2 ** ncontrol,) + nactive * (2,))
            opstring = einsum_utils.apply_gate_string(targets, nactive)
            updates = np.einsum(opstring, state[-1], matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = np.concatenate([state[:-1], updates[np.newaxis]], axis=0)
            state = np.reshape(state, nqubits * (2,))
            # Put qubit indices back to their proper places
            reverse_order = len(order) * [0]
            for i, r in enumerate(order):
                reverse_order[r] = i
            state = np.transpose(state, reverse_order)
        else:
            matrix = np.reshape(self.asmatrix(gate), 2  * len(gate.qubits) * (2,))
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = np.einsum(opstring, state, matrix)
        return np.reshape(state, (2 ** nqubits,))

    def apply_gate_density_matrix(self, gate, state, nqubits):
        state = np.reshape(state, 2 * nqubits * (2,))
        #matrix = np.reshape(self.asmatrix(gate), 2 * len(gate.target_qubits) * (2,))
        #matrixc = np.conj(matrix)
        if gate.is_controlled_by:
            ncontrol = len(gate.control_qubits)
            nactive = gate.nqubits - ncontrol
            n = 2 ** ncontrol
            state = self.transpose(state, gate.cache.control_cache.order(True))
            state = self.reshape(state, 2 * (n,) + 2 * nactive * (2,))
            state01 = self.gather(state, indices=range(n - 1), axis=0)
            state01 = self.squeeze(self.gather(state01, indices=[n - 1], axis=1), axis=1)
            state01 = self.einsum_call(gate.cache.calculation_cache.right0, state01, matrixc)
            state10 = self.gather(state, indices=range(n - 1), axis=1)
            state10 = self.squeeze(self.gather(state10, indices=[n - 1], axis=0), axis=0)
            state10 = self.einsum_call(gate.cache.calculation_cache.left0,
                                       state10, matrix)

            state11 = self.squeeze(self.gather(state, indices=[n - 1], axis=0), axis=0)
            state11 = self.squeeze(self.gather(state11, indices=[n - 1], axis=0), axis=0)
            state11 = self.einsum_call(gate.cache.calculation_cache.right, state11, matrixc)
            state11 = self.einsum_call(gate.cache.calculation_cache.left, state11, matrix)

            state00 = self.gather(state, indices=range(n - 1), axis=0)
            state00 = self.gather(state00, indices=range(n - 1), axis=1)
            state01 = self.concatenate([state00, state01[:, self.newaxis]], axis=1)
            state10 = self.concatenate([state10, state11[self.newaxis]], axis=0)
            state = self.concatenate([state01, state10[self.newaxis]], axis=0)
            state = self.reshape(state, 2 * gate.nqubits * (2,))
            state = self.transpose(state, gate.cache.control_cache.reverse(True))
        else:
            matrix = np.reshape(self.asmatrix(gate), 2 * len(gate.qubits) * (2,))
            matrixc = np.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(gate.qubits, nqubits)
            state = np.einsum(right, state, matrixc)
            state = np.einsum(left, state, matrix)
        return np.reshape(state, 2 * (2 ** nqubits,))

    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):
        np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)
