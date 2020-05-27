# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPECPX, DTYPEINT
from qibo import gates
from qibo.tensorflow import measurements, callbacks
from qibo.tensorflow import custom_operators as op
from typing import List, Optional, Tuple, Union


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        dtype: Tensorflow type for complex numbers.
            Read automatically from `config`.
    """
    _GATE_MODULE = gates

    def __init__(self, nqubits, dtype=DTYPECPX):
        super(TensorflowCircuit, self).__init__(nqubits)
        self.dtype = dtype
        self.compiled_execute = None
        self.callbacks = []

    def __add__(self, circuit: "TensorflowCircuit") -> "TensorflowCircuit":
        return TensorflowCircuit._circuit_addition(self, circuit)

    def _execute_func(self, state: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Simulates the circuit gates.

        Can be compiled using `tf.function` or used as it is in Eager mode.
        """
        # Calculate callbacks for initial state
        callback_results = [[callback(state)] for callback in self.callbacks]

        for ig, gate in enumerate(self.queue):
            if gate.is_channel and not self.using_density_matrix:
                # Switch from vector to density matrix
                self.using_density_matrix = True
                state = tf.tensordot(state, tf.math.conj(state), axes=0)

            state = gate(state, is_density_matrix=self.using_density_matrix)
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

        In default usage the full final state vector or density matrix is returned.
        If the circuit contains measurement gates and `nshots` is given, then
        the final state is sampled and the samples are returned.
        Circuit execution uses by default state vectors but switches automatically
        to density matrices if

        Args:
            initial_state (np.ndarray): Initial state vector as a numpy array of shape ``(2 ** nqubits,)``
                or a density matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
                A Tensorflow tensor with shape ``nqubits * (2,)`` (or ``2 * nqubits * (2,)`` for density matrices)
                is also allowed as an initial state but must have the `dtype` of the circuit.
                If ``initial_state`` is ``None`` the |000...0> state will be used.
            nshots (int): Number of shots to sample if the circuit contains
                measurement gates.
                If ``nshots`` None the measurement gates will be ignored.
            callback: A Callback to calculate during circuit execution.
                See :class:`qibo.tensorflow.callbacks.Callback` for more details.
                User can give a single callback or list of callbacks here.
                Note that if the Circuit is compiled then all callbacks should
                be passed when ``compile`` is called, not during execution.
                Otherwise an ``RuntimeError`` will be raised.

        Returns:
            If ``nshots`` is given and the circuit contains measurements
                A :class:`qibo.base.measurements.CircuitResult` object that contains the measured bitstrings.
            If ``nshots`` is ``None`` or the circuit does not contain measurements.
                The final state vector as a Tensorflow tensor of shape ``(2 ** nqubits,)`` or a density matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
        """
        state = self._cast_initial_state(initial_state)

        state = tf.reshape(state, (1 + self.using_density_matrix) * self.nqubits * (2,))

        if self.compiled_execute is None:
            self._add_callbacks(callback)
            state, callback_results = self._execute_func(state)
        else:
            if callback is not None:
                raise RuntimeError("Cannot add callbacks to compiled circuit. "
                                   "Please pass the callbacks when compiling.")
            state, callback_results = self.compiled_execute(state)

        shape = tf.cast((1+self.using_density_matrix) * (2 ** self.nqubits,),
                        dtype=DTYPEINT)
        state = tf.reshape(state, shape)

        self._final_state = state

        # Append callback results to callbacks
        for callback, result in zip(self.callbacks, callback_results):
            callback.append(result)

        if self.measurement_gate is None or nshots is None:
            return self._final_state

        samples = self.measurement_gate(state, nshots, samples_only=True,
                                        is_density_matrix=self.using_density_matrix)

        self.measurement_gate_result = measurements.GateResult(
            self.measurement_gate.qubits, state, decimal_samples=samples)
        return measurements.CircuitResult(
            self.measurement_tuples, self.measurement_gate_result)

    def __call__(self, initial_state: Optional[tf.Tensor] = None,
                 nshots: Optional[int] = None,
                 callback: Optional[callbacks.Callback] = None) -> tf.Tensor:
        """Equivalent to ``circuit.execute``."""
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
            raise RuntimeError("Cannot access final state before the circuit "
                               "is executed.")
        return self._final_state

    def _cast_initial_state(self, initial_state=None) -> tf.Tensor:
        if initial_state is None:
            return self._default_initial_state()

        if not (isinstance(initial_state, np.ndarray) or
                isinstance(initial_state, tf.Tensor)):
            raise TypeError("Initial state type {} is not recognized."
                            "".format(type(initial_state)))

        shape = tuple(initial_state.shape)
        def shape_error():
            raise ValueError("Invalid initial state shape {} for circuit "
                             "with {} qubits.".format(shape, self.nqubits))

        if len(shape) not in {1, 2}:
            shape_error()
        if len(shape) == 1 and 2 ** self.nqubits != shape[0]:
            shape_error()
        if len(shape) == 2:
            if 2 * (2 ** self.nqubits,) != shape:
                shape_error()
            self.using_density_matrix = True

        return tf.cast(initial_state, dtype=self.dtype)

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        zeros = tf.zeros(2 ** self.nqubits, dtype=self.dtype)
        initial_state = op.initial_state(zeros)
        return initial_state

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
