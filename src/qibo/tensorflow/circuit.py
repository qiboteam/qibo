# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPES
from qibo.tensorflow import measurements
from qibo.tensorflow import custom_operators as op
from typing import List, Optional, Tuple, Union


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """

    def __init__(self, nqubits):
        super(TensorflowCircuit, self).__init__(nqubits)
        self._compiled_execute = None

    def _eager_execute(self, state: tf.Tensor) -> tf.Tensor:
        """Simulates the circuit gates in eager mode."""
        for gate in self.queue:
            if gate.is_channel and not self.using_density_matrix:
                # Switch from vector to density matrix
                self.using_density_matrix = True
                state = tf.tensordot(state, tf.math.conj(state), axes=0)
            state = gate(state, is_density_matrix=self.using_density_matrix)
        return state

    def _execute_for_compile(self, state):
        from qibo import gates
        callback_results = {gate.callback: [] for gate in self.queue
                            if hasattr(gate, "callback")}
        for gate in self.queue:
            if gate.is_channel and not self.using_density_matrix:
                # Switch from vector to density matrix
                self.using_density_matrix = True
                state = tf.tensordot(state, tf.math.conj(state), axes=0)
            if isinstance(gate, gates.CallbackGate):
                callback = gate.callback
                value = callback(state,
                                 is_density_matrix=self.using_density_matrix)
                callback_results[callback].append(value)
            else:
                state = gate(state,
                             is_density_matrix=self.using_density_matrix)
        return state, callback_results

    def compile(self):
        """Compiles the circuit as a Tensorflow graph."""
        if self._compiled_execute is not None:
            raise RuntimeError("Circuit is already compiled.")
        if not self.queue:
            raise RuntimeError("Cannot compile circuit without gates.")
        if not self.using_tfgates:
            raise RuntimeError("Cannot compile circuit that uses custom "
                               "operators.")
        self._compiled_execute = tf.function(self._execute_for_compile)

    @property
    def using_tfgates(self) -> bool:
        """Determines if we are using Tensorflow native or custom gates."""
        from qibo.tensorflow import gates
        return gates.TensorflowGate == self.gate_module.TensorflowGate

    def execute(self,
                initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None,
                nshots: Optional[int] = None,
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

        Returns:
            If ``nshots`` is given and the circuit contains measurements
                A :class:`qibo.base.measurements.CircuitResult` object that contains the measured bitstrings.
            If ``nshots`` is ``None`` or the circuit does not contain measurements.
                The final state vector as a Tensorflow tensor of shape ``(2 ** nqubits,)`` or a density matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
        """
        state = self._cast_initial_state(initial_state)

        if self.using_tfgates:
            shape = (1 + self.using_density_matrix) * self.nqubits * (2,)
            state = tf.reshape(state, shape)

        if self._compiled_execute is None:
            state = self._eager_execute(state)
        else:
            state, callback_results = self._compiled_execute(state)
            for callback, results in callback_results.items():
                callback.extend(results)

        if self.using_tfgates:
            shape = tf.cast((1+self.using_density_matrix) * (2 ** self.nqubits,),
                            dtype=DTYPES.get('DTYPEINT'))
            state = tf.reshape(state, shape)

        self._final_state = state
        if self.measurement_gate is None or nshots is None:
            return self._final_state

        samples = self.measurement_gate(state, nshots, samples_only=True,
                                        is_density_matrix=self.using_density_matrix)

        self.measurement_gate_result = measurements.GateResult(
            self.measurement_gate.qubits, state, decimal_samples=samples)
        return measurements.CircuitResult(
            self.measurement_tuples, self.measurement_gate_result)

    def __call__(self, initial_state: Optional[tf.Tensor] = None,
                 nshots: Optional[int] = None) -> tf.Tensor:
        """Equivalent to ``circuit.execute``."""
        return self.execute(initial_state=initial_state, nshots=nshots)

    @property
    def final_state(self) -> tf.Tensor:
        """Final state as a Tensorflow tensor of shape ``(2 ** nqubits,)``.

        The circuit has to be executed at least once before accessing this
        property, otherwise a ``ValueError`` is raised. If the circuit is
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

        return tf.cast(initial_state, dtype=DTYPES.get('DTYPECPX'))

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        zeros = tf.zeros(2 ** self.nqubits, dtype=DTYPES.get('DTYPECPX'))
        initial_state = op.initial_state(zeros)
        return initial_state
