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

    def set_device(self, device):
        if device != "/CPU:0":
            raise_error(ValueError, f"Device {device} is not available for {self} backend.")

    def set_threads(self, nthreads):
        if nthreads > 1:
            raise_error(ValueError, "numpy does not support more than one thread.")

    def cast(self, x):
        return np.array(x, dtype=self.dtype, copy=False)

    def to_numpy(self, x):
        return x

    def zero_state(self, nqubits):
        state = np.zeros(2 ** nqubits, dtype=self.dtype)
        state[0] = 1
        return state

    def zero_density_matrix(self, nqubits):
        state = np.zeros(2 * (2 ** nqubits,), dtype=self.dtype)
        state[0, 0] = 1
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
        state = self.cast(state)
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
            state = np.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = np.reshape(self.asmatrix(gate), 2  * len(gate.qubits) * (2,))
            opstring = einsum_utils.apply_gate_string(gate.qubits, nqubits)
            state = np.einsum(opstring, state, matrix)
        return np.reshape(state, (2 ** nqubits,))

    def apply_gate_density_matrix(self, gate, state, nqubits):
        state = self.cast(state)
        state = np.reshape(state, 2 * nqubits * (2,))
        if gate.is_controlled_by:
            matrix = np.reshape(self.asmatrix(gate), 2  * len(gate.target_qubits) * (2,))
            matrixc = np.conj(matrix)
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            n = 2 ** ncontrol

            order, targets = einsum_utils.control_order_density_matrix(gate, nqubits)
            state = np.transpose(state, order)
            state = np.reshape(state, 2 * (n,) + 2 * nactive * (2,))

            leftc, rightc = einsum_utils.apply_gate_density_matrix_controlled_string(targets, nactive)
            state01 = state[:n - 1, n - 1]
            state01 = np.einsum(rightc, state01, matrixc)
            state10 = state[n - 1, :n - 1]
            state10 = np.einsum(leftc, state10, matrix)

            left, right = einsum_utils.apply_gate_density_matrix_string(targets, nactive)
            state11 = state[n - 1, n - 1]
            state11 = np.einsum(right, state11, matrixc)
            state11 = np.einsum(left, state11, matrix)

            state00 = state[range(n - 1)]
            state00 = state00[:, range(n - 1)]
            state01 = np.concatenate([state00, state01[:, np.newaxis]], axis=1)
            state10 = np.concatenate([state10, state11[np.newaxis]], axis=0)
            state = np.concatenate([state01, state10[np.newaxis]], axis=0)
            state = np.reshape(state, 2 * nqubits * (2,))
            state = np.transpose(state, einsum_utils.reverse_order(order))
        else:
            matrix = np.reshape(self.asmatrix(gate), 2 * len(gate.qubits) * (2,))
            matrixc = np.conj(matrix)
            left, right = einsum_utils.apply_gate_density_matrix_string(gate.qubits, nqubits)
            state = np.einsum(right, state, matrixc)
            state = np.einsum(left, state, matrix)
        return np.reshape(state, 2 * (2 ** nqubits,))

    def apply_channel(self, channel, state, nqubits):
        # TODO: Think how to implement seed
        for coeff, gate in zip(channel.coefficients, channel.gates):
            if np.random.random() < coeff:
                state = self.apply_gate(gate, state, nqubits)
        return state

    def apply_channel_density_matrix(self, channel, state, nqubits):
        # TODO: Think how to implement seed
        # TODO: Inverse gates may be needed for qibojit (in-place updates)
        state = self.cast(state)
        new_state = (1 - channel.coefficient_sum) * state
        for coeff, gate in zip(channel.coefficients, channel.gates):
            new_state += coeff * self.apply_gate_density_matrix(gate, state, nqubits)
        return new_state

    def assert_allclose(self, value, target, rtol=1e-7, atol=0.0):
        value = self.to_numpy(value)
        target = self.to_numpy(target)
        np.testing.assert_allclose(value, target, rtol=rtol, atol=atol)
