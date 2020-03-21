# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPECPX
from qibo.tensorflow import gates
from typing import Optional


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of circuit methods in Tensorflow."""

    def __init__(self, nqubits, dtype=DTYPECPX):
        """Initialize a Tensorflow circuit."""
        super(TensorflowCircuit, self).__init__(nqubits)
        self.dtype = dtype
        self.compiled_execute = None

    def __add__(self, circuit: "TensorflowCircuit") -> "TensorflowCircuit":
        return TensorflowCircuit._circuit_addition(self, circuit)

    def _execute_func(self, initial_state: tf.Tensor) -> tf.Tensor:
        """Simulates the circuit gates.

        Can be compiled using `tf.function` or used as it is in Eager mode.
        """
        state = tf.cast(initial_state, dtype=self.dtype)
        for gate in self.queue:
            state = gate(state)
        return tf.reshape(state, (2**self.nqubits,))

    def compile(self):
        """Compiles `_execute_func` using `tf.function`."""
        if self.compiled_execute is not None:
            raise RuntimeError("Circuit is already compiled.")
        self.compiled_execute = tf.function(self._execute_func)

    def execute(self, initial_state: Optional[tf.Tensor] = None,
                nshots: Optional[int] = None) -> tf.Tensor:
        """Executes the Tensorflow circuit."""
        self.is_executed = True
        if initial_state is None:
            state = self._default_initial_state()
        else:
            shape = tuple(initial_state.shape)
            if len(shape) == self.nqubits:
                state = tf.cast(initial_state, dtype=self.dtype)
            elif len(shape) == 1:
                state = tf.reshape(initial_state, self.nqubits * (2,))
            else:
                raise ValueError("Given initial state has unsupported shape "
                                 "{}.".format(shape))

        if self.compiled_execute is None:
            return self._execute_func(state)
        return self.compiled_execute(state)

    def __call__(self, initial_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        return self.execute(initial_state)

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        initial_state = np.zeros(2 ** self.nqubits)
        initial_state[0] = 1
        initial_state = initial_state.reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(initial_state, dtype=self.dtype)
