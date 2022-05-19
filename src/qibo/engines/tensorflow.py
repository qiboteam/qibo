import os
import numpy as np
from qibo.engines.numpy import NumpyEngine
from qibo.config import TF_LOG_LEVEL


class TensorflowEngine(NumpyEngine):

    def __init__(self):
        super().__init__()
        self.name = "tensorflow"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(TF_LOG_LEVEL)
        import tensorflow as tf
        self.tf = tf

    def asmatrix(self, gate):
        npmatrix = super().asmatrix(gate)
        return self.tf.cast(npmatrix, dtype=self.dtype)

    def apply_gate(self, gate, state, nqubits):
        # TODO: Implement density matrices (most likely in another method)
        state = self.tf.reshape(state, nqubits * (2,))
        matrix = self.tf.reshape(self.asmatrix(gate), 2  * len(gate.qubits) * (2,))
        opstring = self._einsum_string(gate, nqubits)
        if gate.is_controlled_by:
            ncontrol = len(gate.control_qubits)
            nactive = nqubits - ncontrol
            order, _ = self._control_order(gate, nqubits)
            state = self.tf.transpose(state, order)
            # Apply `einsum` only to the part of the state where all controls
            # are active. This should be `state[-1]`
            state = self.tf.reshape(state, (2 ** ncontrol,) + nactive * (2,))
            updates = self.tf.einsum(opstring, state[-1], matrix)
            # Concatenate the updated part of the state `updates` with the
            # part of of the state that remained unaffected `state[:-1]`.
            state = self.tf.concatenate([state[:-1], updates[self.tf.newaxis]], axis=0)
            state = self.tf.reshape(state, nqubits * (2,))
            # Put qubit indices back to their proper places
            reverse_order = len(order) * [0]
            for i, r in enumerate(order):
                reverse_order[r] = i
            state = self.tf.transpose(state, reverse_order)
        else:
            state = self.tf.einsum(opstring, state, matrix)
        return self.tf.reshape(state, (2 ** nqubits,))

    def zero_state(self, nqubits):
        """Generate |000...0> state as an array."""
        idx = self.tf.constant([[0]], dtype="int32")
        state = self.tf.zeros((2 ** nqubits,), dtype=self.dtype)
        update = self.tf.constant([1], dtype=self.dtype)
        state = self.tf.tensor_scatter_nd_update(state, idx, update)
        return state
