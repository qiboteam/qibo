# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.backends import common, config
from typing import List


class GateMatrices:
    # TODO: Add docstrings

    _AVAILABLE_GATES = ["I", "H", "X", "Y", "Z", "CNOT"]

    def __init__(self, dtype):
        self.dtype = dtype
        for gate in self._AVAILABLE_GATES:
            matrix = getattr(self, "np{}".format(gate))()
            tf_matrix = tf.convert_to_tensor(matrix, dtype=self.dtype)
            setattr(self, gate, tf_matrix)

    @property
    def nptype(self):
        if self.dtype == tf.complex128:
            return np.complex128
        elif self.dtype == tf.complex64:
            return np.complex64
        else:
            raise TypeError("Unknown complex type {}.".format(self.dtype))

    def npI(self):
        return np.eye(2, dtype=self.nptype)

    def npH(self):
        m = np.ones((2, 2), dtype=self.nptype)
        m[1, 1] = -1
        return m / np.sqrt(2)

    def npX(self):
        m = np.zeros((2, 2), dtype=self.nptype)
        m[0, 1], m[1, 0] = 1, 1
        return m

    def npY(self):
        m = np.zeros((2, 2), dtype=self.nptype)
        m[0, 1], m[1, 0] = -1j, 1j
        return m

    def npZ(self):
        m = np.eye(2, dtype=self.nptype)
        m[1, 1] = -1
        return m

    def npCNOT(self):
        m = np.eye(4, dtype=self.nptype)
        m[2, 2], m[2, 3] = 0, 1
        m[3, 2], m[3, 3] = 1, 0
        return m.reshape(4 * (2,))


class TensorflowBackend(common.Backend):
    """Implementation of all backend methods in Tensorflow."""

    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, dtype=tf.complex128):
        """Initializes the class attributes"""
        self._output = {"virtual_machine": None, "wave_func": None, "measure": []}
        self.dtype = dtype

        self.nqubits = None
        self.state = None
        self.matrices = GateMatrices(self.dtype)

    def CNOT(self, id0: int, id1: int):
        """The Controlled-NOT gate."""
        self._apply_gate(self.matrices.CNOT, [id0, id1])

    def H(self, id: int):
        """The Hadamard gate."""
        self._apply_gate(self.matrices.H, [id])

    def X(self, id):
        """The Pauli X gate."""
        self._apply_gate(self.matrices.X, [id])

    def Y(self, id):
        """The Pauli Y gate."""
        self._apply_gate(self.matrices.Y, [id])

    def Z(self, id):
        """The Pauli Z gate."""
        self._apply_gate(self.matrices.Z, [id])

    def Barrier(self, **args):
        """The barrier gate."""
        raise NotImplementedError

    def S(self, **args):
        """The swap gate."""
        raise NotImplementedError

    def T(self, **args):
        """The Toffoli gate."""
        raise NotImplementedError

    def Iden(self, **args):
        """The identity gate."""
        raise NotImplementedError

    def RX(self, id: int, theta: float):
        """The rotation around X-axis gate."""
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        mat = (phase * cos * self.matrices.I -
               1j * phase * sin * self.matrices.X)
        self._apply_gate(mat, [id])

    def RY(self, id: int, theta: float):
        """The rotation around Y-axis gate."""
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        mat = (phase * cos * self.matrices.I -
               1j * phase * sin * self.matrices.Y)
        self._apply_gate(mat, [id])

    def RZ(self, id: int, theta: float):
        """The rotation around Z-axis gate."""
        phase = tf.exp(1j * np.pi * theta)
        rz = tf.eye(2, dtype=self.dtype)
        rz = tf.tensor_scatter_nd_update(rz, [[1, 1]], [phase])
        self._apply_gate(rz, [id])

    def MX(self, **args):
        """The measure gate X."""
        raise NotImplementedError

    def MY(self, **args):
        """The measure gate Y."""
        raise NotImplementedError

    def MZ(self, **args):
        """The measure gate Z."""
        raise NotImplementedError

    def Flatten(self, coefficients):
        """Set wave function coefficients"""
        if len(coefficients) != 2 ** self.nqubits:
            raise ValueError(
                "Circuit was created with {} qubits but the "
                "flatten layer state has {} coefficients."
                "".format(self.nqubits, coefficients)
            )
        _state = np.array(coefficients).reshape(self.nqubits * (2,))
        self.state = tf.convert_to_tensor(_state, dtype=self.dtype)

    def execute(self, model):
        """Executes the circuit on tensorflow."""
        if self.state is not None or self.nqubits is not None:
            raise ValueError("Backend was already used.")
        self.nqubits = model.nqubits

        def _execute(initial_state):
            self.state = tf.cast(initial_state, dtype=self.dtype)
            for gate in model.queue:
                getattr(self, gate.name)(**gate.args)
            return self.state

        # Compile model
        compiled_execute = tf.function(_execute)

        # Initialize in |000...0> state
        initial_state = np.zeros(2 ** self.nqubits)
        initial_state[0] = 1
        initial_state = initial_state.reshape(self.nqubits * (2,))
        initial_state = tf.convert_to_tensor(initial_state, dtype=self.dtype)

        final_state = compiled_execute(initial_state)
        self._output["wave_func"] = final_state.numpy().ravel()
        return self._output["wave_func"]

    def _apply_gate(self, matrix: tf.Tensor, qubits: List[int]):
        """Applies gate represented by matrix to `self.state`.

        Args:
            matrix: The matrix that represents the gate to be applied.
                This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
            qubits: List with the qubits that the gate is applied to.
        """
        einsum_str = self._create_einsum_str(qubits)
        self.state = tf.einsum(einsum_str, self.state, matrix)

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
