# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.config import matrices
from typing import List
# TODO: Inherit `base_gates` instead of redifining all __init__


class TensorflowGate:
    """The base Tensorflow gate.

    **Properties:**
        matrix: The matrix that represents the gate to be applied.
            This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
        qubits: List with the qubits that the gate is applied to.
    """

    matrix = None
    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self):
        self.nqubits = None

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


class CNOT(TensorflowGate):

    matrix = matrices.CNOT

    def __init__(self, q0, q1):
        super(CNOT, self).__init__()
        self.name = "CNOT"
        self.qubits = [q0, q1]


class H(TensorflowGate):

    matrix = matrices.H

    def __init__(self, q):
        super(H, self).__init__()
        self.name = "H"
        self.qubits = [q]


class X(TensorflowGate):

    matrix = matrices.X

    def __init__(self, q):
        super(X, self).__init__()
        self.name = "X"
        self.qubits = [q]


class Y(TensorflowGate):

    matrix = matrices.Y

    def __init__(self, q):
        super(Y, self).__init__()
        self.name = "Y"
        self.qubits = [q]


class Z(TensorflowGate):

    matrix = matrices.Z

    def __init__(self, q):
        super(Z, self).__init__()
        self.name = "Z"
        self.qubits = [q]


class Barrier:

    def __init__(self):
        raise NotImplementedError


class S:

    def __init__(self):
        raise NotImplementedError


class T:

    def __init__(self):
        raise NotImplementedError


class Iden(TensorflowGate):

    matrix = matrices.I

    def __init__(self, q):
        super(Iden, self).__init__()
        self.name = "Iden"
        self.qubits = [q]


class MX:

    def __init__(self):
        raise NotImplementedError


class MY:

    def __init__(self):
        raise NotImplementedError


class MZ:

    def __init__(self):
        raise NotImplementedError


class Flatten(TensorflowGate):

    def __init__(self, coefficients):
        super(Flatten, self).__init__()
        self.name = "Flatten"
        self.coefficients = coefficients

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