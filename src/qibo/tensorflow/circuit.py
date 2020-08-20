# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPES, DEVICES, BACKEND, raise_error
from qibo.tensorflow import measurements
from qibo.tensorflow import custom_operators as op
from typing import List, Optional, Tuple, Union
InitStateType = Union[np.ndarray, tf.Tensor]
OutputType = Union[tf.Tensor, measurements.CircuitResult]


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """
    from qibo.tensorflow import fusion

    def __init__(self, nqubits):
        super(TensorflowCircuit, self).__init__(nqubits)
        self._compiled_execute = None
        self.check_initial_state_shape = True

    def _set_nqubits(self, gate):
        if gate._nqubits is None:
            with tf.device(DEVICES['DEFAULT']):
                gate.nqubits = self.nqubits
        elif gate.nqubits != self.nqubits:
            super(TensorflowCircuit, self)._set_nqubits(gate)

    def set_parameters(self, parameters):
        if isinstance(parameters, (np.ndarray, tf.Tensor, tf.Variable)):
            super(TensorflowCircuit, self)._set_parameters_list(
                parameters, int(parameters.shape[0]))
        else:
            super(TensorflowCircuit, self).set_parameters(parameters)

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
            if gate.is_channel and not self.using_density_matrix: # pragma: no cover
                # compilation may be deprecated and is not sufficiently tested
                # Switch from vector to density matrix
                self.using_density_matrix = True
                state = tf.tensordot(state, tf.math.conj(state), axes=0)
            if isinstance(gate, gates.CallbackGate): # pragma: no cover
                # compilation may be deprecated and is not sufficiently tested
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
            raise_error(RuntimeError, "Circuit is already compiled.")
        if not self.queue:
            raise_error(RuntimeError, "Cannot compile circuit without gates.")
        if not self.using_tfgates:
            raise_error(RuntimeError, "Cannot compile circuit that uses custom "
                                      "operators.")
        self._compiled_execute = tf.function(self._execute_for_compile)

    @property
    def using_tfgates(self) -> bool:
        """Determines if we are using Tensorflow native or custom gates."""
        return BACKEND['GATES'] != 'custom'

    def _execute(self, initial_state: Optional[InitStateType] = None,
                 nshots: Optional[int] = None) -> OutputType:
        """Performs ``circuit.execute`` on specified device."""
        self.using_density_matrix = False
        state = self.get_initial_state(initial_state)

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

    def execute(self, initial_state: Optional[InitStateType] = None,
                nshots: Optional[int] = None) -> OutputType:
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
        oom_error = tf.python.framework.errors_impl.ResourceExhaustedError
        device = DEVICES['DEFAULT']
        try:
            with tf.device(device):
                return self._execute(initial_state=initial_state, nshots=nshots)
        except oom_error:
            raise_error(RuntimeError, f"State does not fit in {device} memory."
                                       "Please switch the execution device to a "
                                       "different one using ``qibo.set_device``.")

    def __call__(self, initial_state: Optional[InitStateType] = None,
                 nshots: Optional[int] = None) -> OutputType:
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
            raise_error(RuntimeError, "Cannot access final state before the circuit "
                                      "is executed.")
        return self._final_state

    def _check_initial_shape(self, state: InitStateType):
        """Checks shape of given initial state."""
        if not isinstance(state, (np.ndarray, tf.Tensor)):
            raise_error(TypeError, "Initial state type {} is not recognized."
                                    "".format(type(state)))

        shape = tuple(state.shape)
        def shape_error():
            raise_error(ValueError, "Invalid initial state shape {} for "
                                    "circuit with {} qubits."
                                    "".format(shape, self.nqubits))
        if len(shape) not in {1, 2}:
            shape_error()
        if len(shape) == 1 and 2 ** self.nqubits != shape[0]:
            shape_error()
        if len(shape) == 2:
            if 2 * (2 ** self.nqubits,) != shape:
                shape_error()
            self.using_density_matrix = True

    def _cast_initial_state(self, state: InitStateType) -> tf.Tensor:
        if isinstance(state, tf.Tensor):
            return state
        elif isinstance(state, np.ndarray):
            return tf.cast(state, dtype=DTYPES.get('DTYPECPX'))
        raise_error(TypeError, "Initial state type {} is not recognized."
                                "".format(type(state)))

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        zeros = tf.zeros(2 ** self.nqubits, dtype=DTYPES.get('DTYPECPX'))
        state = op.initial_state(zeros)
        return state

    def get_initial_state(self, state: Optional[InitStateType] = None
                           ) -> tf.Tensor:
        """"""
        if state is None:
            return self._default_initial_state()
        if self.check_initial_state_shape:
            self._check_initial_shape(state)
        return self._cast_initial_state(state)
