# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.config import matrices
from typing import List


class TensorflowGate:
    """The base Tensorflow gate.

    **Properties:**
        matrix: The matrix that represents the gate to be applied.
            This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
        qubits: List with the qubits that the gate is applied to.
    """

    dtype = matrices.dtype
    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        if self.nqubits is None:
            self.nqubits = len(tuple(state.shape))

        einsum_str = self._create_einsum_str(self.qubits)
        return tf.einsum(einsum_str, state, self.matrix)

    def _create_einsum_str(self, qubits: List[int]) -> str:
        """Creates index string for `tf.einsum`.

        Args:
            qubits: List with the qubit indices that the gate is applied to.

        Returns:
            String formated as {input state}{gate matrix}->{output state}.
        """
        if len(qubits) + self.nqubits > len(self._chars):
            raise NotImplementedError("Not enough einsum characters.")

        input_state = list(self._chars[: self.nqubits])
        output_state = input_state[:]
        gate_chars = list(self._chars[self.nqubits : self.nqubits + len(qubits)])

        for i, q in enumerate(qubits):
            gate_chars.append(input_state[q])
            output_state[q] = gate_chars[i]

        input_str = "".join(input_state)
        gate_str = "".join(gate_chars)
        output_str = "".join(output_state)
        return "{},{}->{}".format(input_str, gate_str, output_str)


class H(TensorflowGate, base_gates.H):

    def __init__(self, *args):
        base_gates.H.__init__(self, *args)
        self.matrix = matrices.H


class X(TensorflowGate, base_gates.X):

    def __init__(self, *args):
        base_gates.X.__init__(self, *args)
        self.matrix = matrices.X


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, *args):
        base_gates.Y.__init__(self, *args)
        self.matrix = matrices.Y


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, *args):
        base_gates.Z.__init__(self, *args)
        self.matrix = matrices.Z


class Barrier(TensorflowGate, base_gates.Barrier):

    def __init__(self):
        raise NotImplementedError


class Iden(TensorflowGate, base_gates.Iden):

    def __init__(self, *args):
        base_gates.Iden.__init__(self, *args)
        self.matrix = matrices.Iden


class MX(TensorflowGate, base_gates.MX):

    def __init__(self):
        raise NotImplementedError


class MY(TensorflowGate, base_gates.MY):

    def __init__(self):
        raise NotImplementedError


class MZ(TensorflowGate, base_gates.MZ):

    def __init__(self):
        raise NotImplementedError


class RX(TensorflowGate, base_gates.RX):

    def __init__(self, *args):
        base_gates.RX.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        self.matrix = phase * (cos * matrices.I - 1j * sin * matrices.X)


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, *args):
        base_gates.RY.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        self.matrix = phase * (cos * matrices.I - 1j * sin * matrices.Y)


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, *args):
        base_gates.RZ.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta)
        rz = tf.eye(2, dtype=self.dtype)
        self.matrix = tf.tensor_scatter_nd_update(rz, [[1, 1]], [phase])


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, *args):
        base_gates.CNOT.__init__(self, *args)
        self.matrix = matrices.CNOT


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, *args):
        base_gates.SWAP.__init__(self, *args)
        self.matrix = matrices.SWAP


class CRZ(TensorflowGate, base_gates.CRZ):

    def __init__(self, *args):
        base_gates.CRZ.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta)
        crz = tf.eye(4, dtype=self.dtype)
        crz = tf.tensor_scatter_nd_update(crz, [[3, 3]], [phase])
        self.matrix = tf.reshape(crz, 4 * (2,))


class Toffoli(TensorflowGate, base_gates.Toffoli):

    def __init__(self, *args):
        base_gates.Toffoli.__init__(self, *args)
        self.matrix = matrices.Toffoli


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, *args):
        base_gates.Flatten.__init__(self, *args)

    def __call__(self, state):
        if self.nqubits is None:
            self.nqubits = len(tuple(state.shape))

        if len(self.coefficients) != 2 ** self.nqubits:
                raise ValueError(
                    "Circuit was created with {} qubits but the "
                    "flatten layer state has {} coefficients."
                    "".format(self.nqubits, self.coefficients)
                )

        _state = np.array(self.coefficients).reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(_state, dtype=state.dtype)
