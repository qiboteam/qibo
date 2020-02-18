# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import functools
import numpy as np
import tensorflow as tf
from src.qibo.backends import common
from typing import List


class TensorflowBackend(common.Backend):
    """Implementation of all backend methods in Tensorflow."""

    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, dtype=tf.complex64):
        """Initializes the class attributes"""
        self._output = {'virtual_machine': None, 'wave_func': None, 'measure': []}
        self.dtype = dtype

        self.nqubits = None
        self.state = None
        self._create_gates()

    def _create_gates(self):
        _H = np.ones((2, 2))
        _H[1, 1] = 1
        _H = _H / np.sqrt(2)
        self._H = tf.convert_to_tensor(_H, dtype=self.dtype)

        _X = np.zeros((2, 2))
        _X[1, 0], _X[0, 1] = 1, 1
        self._X = tf.convert_to_tensor(_X, dtype=self.dtype)

        _Z = np.eye(2)
        _Z[1, 1] = -1
        self._Z = tf.convert_to_tensor(_Z, dtype=self.dtype)

        _CNOT = np.eye(4)
        _CNOT[1, 1], _CNOT[1, 2] = 0, 1
        _CNOT[2, 1], _CNOT[2, 2] = 1, 0
        self._CNOT = tf.convert_to_tensor(_CNOT, dtype=self.dtype)

    def CNOT(self, id0: int, id1: int):
        """The Controlled-NOT gate."""
        self._apply_gate(self._CNOT, [id0, id1])

    def H(self, id: int):
        """The Hadamard gate."""
        self._apply_gate(self._H, [id])

    def X(self, id):
        """The Pauli X gate."""
        self._apply_gate(self._X, [id])

    def Y(self, id):
        """The Pauli Y gate."""
        raise NotImplementedError

    def Z(self, id):
        """The Pauli Z gate."""
        self._apply_gate(self._Z, [id])

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
        if len(coefficients) != 2**self.nqubits:
            raise ValueError("Circuit was created with {} qubits but the "
                             "flatten layer state has {} coefficients."
                             "".format(self.nqubits, coefficients))
        _state = np.array(coefficients).reshape(self.nqubits * (2,))
        self.state = tf.convert_to_tensor(_state, dtype=self.dtype)

    def _execute(self, initial_state, model):
        self.state = tf.cast(initial_state, dtype=self.dtype)
        for gate in model.queue:
            getattr(self, gate.name)(**gate.args)
        return self.state

    def execute(self, model, shots=None):
        """Executes the circuit on tensorflow."""
        if self.state is not None or self.nqubits is not None:
            raise ValueError("Backend was already used.")
        self.nqubits = model.nqubits

        # Compile model
        compiled_execute = tf.function(functools.partial(self._execute, model=model))

        # Initialize in |000...0> state
        initial_state = np.zeros(2**self.nqubits)
        initial_state[0] = 1
        initial_state = initial_state.reshape(self.nqubits * (2,))
        initial_state = tf.convert_to_tensor(initial_state, dtype=self.dtype)

        return compiled_execute(initial_state)

    def _apply_gate(self, matrix: tf.Tensor, qubits: List[int]):
        einsum_str = self._create_einsum_str(qubits)
        self.state = tf.einsum(einsum_str, self.state, matrix)

    def _create_einsum_str(self, qubits: List[int]) -> str:
        if len(qubits) + self.nqubits > len(self._chars):
          raise NotImplementedError("Not enough einsum characters.")

        input_state = list(self._chars[:self.nqubits])
        output_state = input_state[:]
        gate_chars = list(self._chars[self.nqubits: self.nqubits + len(qubits)])

        for i, q in enumerate(qubits):
          gate_chars.append(input_state[q])
          output_state[q] = gate_chars[i]

        input_str = "".join(input_state)
        gate_str = "".join(gate_chars)
        output_str = "".join(output_state)
        return "{},{}->{}".format(input_str, gate_str, output_str)