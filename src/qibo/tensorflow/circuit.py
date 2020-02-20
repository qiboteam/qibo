# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPECPX
from typing import Optional


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of circuit methods in Tensorflow."""

    def __init__(self, nqubits, dtype=DTYPECPX):
        """Initialize a Tensorflow circuit."""
        super(TensorflowCircuit, self).__init__(nqubits)
        self.dtype = dtype
        self.compiled_execute = None

    def compile(self):
        def _execute(initial_state):
            state = tf.cast(initial_state, dtype=self.dtype)
            for gate in self.queue:
                state = gate(state)
            return state

        if self.compiled_execute is not None:
            raise RuntimeError("Tensorflow circuit is already compiled.")

        self.compiled_execute = tf.function(_execute)

    def execute(self, initial_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Executes the Tensorflow circuit."""
        if initial_state is None:
            initial_state = self._default_initial_state()

        if self.compiled_execute is None:
            self.compile()

        final_state = self.compiled_execute(initial_state)
        return tf.reshape(final_state, (2**self.nqubits,)).numpy()

    def __call__(self, initial_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        return self.execute(initial_state)

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        initial_state = np.zeros(2 ** self.nqubits)
        initial_state[0] = 1
        initial_state = initial_state.reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(initial_state, dtype=self.dtype)