# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPECPX
from qibo.tensorflow import gates, measurements, callbacks
from typing import List, Optional, Tuple, Union


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
        self.callbacks = []
        self._final_state = None

    def __add__(self, circuit: "TensorflowCircuit") -> "TensorflowCircuit":
        return TensorflowCircuit._circuit_addition(self, circuit)

    def _execute_func(self, state: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Simulates the circuit gates.

        Can be compiled using `tf.function` or used as it is in Eager mode.
        """
        # Calculate callbacks for initial state
        callback_results = [[callback(state)] for callback in self.callbacks]

        for ig, gate in enumerate(self.queue):
            state = gate(state)
            for ic, callback in enumerate(self.callbacks):
                if (ig + 1) % callback.steps == 0:
                    callback_results[ic].append(callback(state))

        # Stack all results for each callback
        callback_results = [tf.stack(r) for r in callback_results]

        return state, callback_results

    def compile(self, callback: Optional[callbacks.Callback] = None):
        """Compiles the circuit as a Tensorflow graph.

        Args:
            callback: A Callback to calculate during circuit execution.
                See :class:`qibo.tensorflow.callbacks.Callback` for more details.
                User can give a single callback or list of callbacks here.
        """
        if self.compiled_execute is not None:
            raise RuntimeError("Circuit is already compiled.")
        self._add_callbacks(callback)
        self.compiled_execute = tf.function(self._execute_func)

    def execute(self,
                initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None,
                nshots: Optional[int] = None,
                callback: Optional[callbacks.Callback] = None
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
            callback: A Callback to calculate during circuit execution.
                See :class:`qibo.tensorflow.callbacks.Callback` for more details.
                User can give a single callback or list of callbacks here.
                Note that if the Circuit is compiled then all callbacks should
                be passed when `compile` is called, not during execution.
                Otherwise an `RuntimeError` will be raised.

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
            self._add_callbacks(callback)
            state, callback_results = self._execute_func(state)
        else:
            if callback is not None:
                raise RuntimeError("Cannot add callbacks to compiled circuit. "
                                   "Please pass the callbacks when compiling.")
            state, callback_results = self.compiled_execute(state)

        # Append callback results to callbacks
        for callback, result in zip(self.callbacks, callback_results):
            callback.append(result)

        if self.measurement_gate is None or nshots is None:
            self._final_state = tf.reshape(state, (2 ** self.nqubits,))
            return self._final_state

        samples = self.measurement_gate(state, nshots, samples_only=True)
        self._final_state = state

        self.measurement_gate_result = measurements.GateResult(
            self.measurement_gate.qubits, state, decimal_samples=samples)
        return measurements.CircuitResult(
            self.measurement_tuples, self.measurement_gate_result)

    def __call__(self, initial_state: Optional[tf.Tensor] = None,
                 nshots: Optional[int] = None,
                 callback: Optional[callbacks.Callback] = None) -> tf.Tensor:
        """Equivalent to `circuit.execute()`."""
        return self.execute(initial_state=initial_state, nshots=nshots,
                            callback=callback)

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

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        initial_state = np.zeros(2 ** self.nqubits)
        initial_state[0] = 1
        initial_state = initial_state.reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(initial_state, dtype=self.dtype)

    def _add_callbacks(self, callback: callbacks.Callback):
        """Adds callbacks in the circuit."""
        n = len(self.callbacks)
        if isinstance(callback, list):
            self.callbacks += callback
        elif isinstance(callback, callbacks.Callback):
            self.callbacks.append(callback)
        # Set number of qubits in new callbacks
        for cb in self.callbacks[n:]:
            cb.nqubits = self.nqubits
