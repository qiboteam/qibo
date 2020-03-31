# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPECPX
from qibo.tensorflow import gates, measurements
from typing import Optional, Union


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        dtype: Tensorflow type for complex numbers.
            Read automatically from `config`.
    """

    def __init__(self, nqubits, dtype=DTYPECPX):
        super(TensorflowCircuit, self).__init__(nqubits)
        self.dtype = dtype
        self.compiled_execute = None
        self._final_state = None

    def __add__(self, circuit: "TensorflowCircuit") -> "TensorflowCircuit":
        return TensorflowCircuit._circuit_addition(self, circuit)

    def _execute_func(self, state: tf.Tensor, nshots: Optional[int] = None
                      ) -> tf.Tensor:
        """Simulates the circuit gates.

        Can be compiled using `tf.function` or used as it is in Eager mode.
        """
        for gate in self.queue:
            state = gate(state)
        return state

    def compile(self):
        """Compiles the circuit as a Tensorflow graph."""
        if self.compiled_execute is not None:
            raise RuntimeError("Circuit is already compiled.")
        self.compiled_execute = tf.function(self._execute_func)

    def execute(self,
                initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None,
                nshots: Optional[int] = None
                ) -> Union[tf.Tensor, measurements.CircuitResult]:
        """Propagates the state through the circuit applying the corresponding gates.

        In default usage the full final state vector is returned.
        If the circuit contains measurement gates and `nshots` is given, then
        the final state vector is sampled and the samples are returned.

        Args:
            initial_state (np.ndarray): Initial state vector as a numpy array.
                A Tensorflow tensor with shape nqubits * (2,) is also allowed as an initial state if it has the `dtype` of the circuit.
                If `initial_state` is `None` the |000...0> state will be used.
            nshots (int): Number of shots to sample if the circuit contains
                measurement gates.
                If `nshots` None the measurement gates will be ignored.

        Returns:
            If `nshots` is given and the circuit contains measurements
                A :class:`qibo.base.measurements.CircuitResult` object that contains the measured bitstrings.
            If `nshots` is `None` or the circuit does not contain measurements.
                The final state vector as a Tensorflow tensor of shape (2 ** nqubits,).
        """
        if initial_state is None:
            state = self._default_initial_state()
        elif isinstance(initial_state, np.ndarray):
            state = tf.cast(initial_state.reshape(self.nqubits * (2,)),
                            dtype=self.dtype)
        elif isinstance(initial_state, tf.Tensor):
            if tuple(initial_state.shape) != self.nqubits * (2,):
                raise ValueError("Initial state should be a rank-n tensor if "
                                 "it is passed as a Tensorflow tensor but it "
                                 "has shape {}.".format(initial_state.shape))
            if initial_state.dtype != self.dtype:
                raise TypeError("Circuit is of type {} but initial state is "
                                "{}.".format(self.dtype, initial_state.dtype))
            state = initial_state
        else:
            raise TypeError("Initial state type {} is not recognized."
                            "".format(type(initial_state)))

        if self.compiled_execute is None:
            state = self._execute_func(state, nshots)
        else:
            state = self.compiled_execute(state, nshots)

        if self.measurement_gate is None or nshots is None:
            self._final_state = tf.reshape(state, (2 ** self.nqubits,))
            return self._final_state

        samples = self.measurement_gate(state, nshots, samples_only=True)
        self._final_state = state

        self.measurement_gate_result = measurements.GateResult(
            self.measurement_gate.qubits, state, decimal_samples=samples)
        return measurements.CircuitResult(
            self.measurement_sets, self.measurement_gate_result)

    @property
    def final_state(self) -> tf.Tensor:
        """Final state as a Tensorflow tensor of shape (2 ** nqubits,).

        The circuit has to be executed at least once before accessing this
        property, otherwise a `ValueError` is raised. If the circuit is
        executed more than once, only the last final state is returned.
        """
        if self._final_state is None:
            raise ValueError("Cannot access final state before the circuit is "
                             "executed.")
        if self.measurement_gate_result is None:
            return self._final_state
        return tf.reshape(self._final_state, (2 ** self.nqubits,))

    def __call__(self, initial_state: Optional[tf.Tensor] = None,
                 nshots: Optional[int] = None) -> tf.Tensor:
        """Equivalent to `circuit.execute()`."""
        return self.execute(initial_state=initial_state, nshots=nshots)

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        initial_state = np.zeros(2 ** self.nqubits)
        initial_state[0] = 1
        initial_state = initial_state.reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(initial_state, dtype=self.dtype)
