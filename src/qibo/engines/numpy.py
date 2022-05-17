import numpy as np
from qibo.config import EINSUM_CHARS
from qibo.engines.abstract import Simulator
from qibo.engines.matrices import Matrices


class NumpyEngine(Simulator):

    def __init__(self, dtype="complex128"):
        self.name = "numpy"
        self.dtype = dtype
        self.matrices = Matrices(dtype)

    def asmatrix(self, gate):
        return getattr(self.matrices, gate.__class__.__name__)(*gate.parameters)

    def _einsum_string(self, gate, nqubits):
        inp = list(EINSUM_CHARS[:nqubits])
        out = inp[:]
        trans = list(EINSUM_CHARS[nqubits : nqubits + len(gate.qubits)])
        for i, q in enumerate(gate.qubits):
            trans.append(inp[q])
            out[q] = trans[i]
        return "{},{}->{}".format("".join(inp), "".join(trans), "".join(out))

    def _control_order(self, gate, nqubits):
        loop_start = 0
        order = list(gate.control_qubits)
        targets = list(gate.target_qubits)
        for control in gate.control_qubits:
            for i in range(loop_start, control):
                order.append(i)
            loop_start = control + 1
            for i, t in enumerate(gate.target_qubits):
                if t > control:
                    targets[i] -= 1
        for i in range(loop_start, nqubits):
            order.append(i)
        return order, targets

    def apply_gate(self, gate, state, nqubits):
        # TODO: Implement density matrices
        # (most likely in another method or a different engine?)
        state = np.reshape(state, nqubits * (2,))
        matrix = np.reshape(self.asmatrix(gate), 2  * len(gate.qubits) * (2,))
        opstring = self._einsum_string(gate, nqubits)
        if gate.is_controlled_by:
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, _ = self._control_order(gate, nqubits)
            state = np.transpose(state, order)
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = np.reshape(state, (2 ** ncontrol,) + nactive * (2,))
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
            state = np.einsum(opstring, state, matrix)
        return np.reshape(state, (2 ** nqubits,))

    def zero_state(self, nqubits):
        """Generate |000...0> state as an array."""
        state = np.zeros(2 ** nqubits, dtype=self.dtype)
        state[0] = 1
        return state
