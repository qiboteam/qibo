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

    Performs simulation using state vectors.

    Example:
        ::

            from qibo import models, gates
            c = models.Circuit(3) # initialized circuit with 3 qubits
            c.add(gates.H(0)) # added Hadamard gate on qubit 0

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """
    from qibo.tensorflow import fusion

    def __init__(self, nqubits):
        super(TensorflowCircuit, self).__init__(nqubits)
        self._compiled_execute = None
        self.check_initial_state_shape = True
        self.shapes = {
            'TENSOR': self.nqubits * (2,),
            'FLAT': (2 ** self.nqubits,)
        }
        self.shapes['TF_FLAT'] = tf.cast(self.shapes.get('FLAT'),
                                         dtype=DTYPES.get('DTYPEINT'))

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
            state = gate(state)
        return state

    def _execute_for_compile(self, state):
        from qibo import gates
        callback_results = {gate.callback: [] for gate in self.queue
                            if hasattr(gate, "callback")}
        for gate in self.queue:
            if isinstance(gate, gates.CallbackGate): # pragma: no cover
                # compilation may be deprecated and is not sufficiently tested
                value = gate.callback(state)
                callback_results[gate.callback].append(value)
            else:
                state = gate(state)
        return state, callback_results

    def compile(self):
        """Compiles the circuit as a Tensorflow graph."""
        if self._compiled_execute is not None:
            raise_error(RuntimeError, "Circuit is already compiled.")
        if not self.queue:
            raise_error(RuntimeError, "Cannot compile circuit without gates.")
        if BACKEND['GATES'] == 'custom':
            raise_error(RuntimeError, "Cannot compile circuit that uses custom "
                                      "operators.")
        self._compiled_execute = tf.function(self._execute_for_compile)

    def _execute(self, initial_state: Optional[InitStateType] = None
                 ) -> tf.Tensor:
        """Performs all circuit gates on the state vector."""
        self._final_state = None
        state = self.get_initial_state(initial_state)
        if BACKEND['GATES'] != 'custom':
            state = tf.reshape(state, self.shapes.get('TENSOR'))

        if self._compiled_execute is None:
            state = self._eager_execute(state)
        else:
            state, callback_results = self._compiled_execute(state)
            for callback, results in callback_results.items():
                callback.extend(results)

        if BACKEND['GATES'] != 'custom':
            state = tf.reshape(state, self.shapes.get('TF_FLAT'))

        self._final_state = state
        return state

    def _device_execute(self, initial_state: Optional[InitStateType] = None
                        ) -> tf.Tensor:
        """Executes circuit on the specified device and checks for OOM errors."""
        oom_error = tf.python.framework.errors_impl.ResourceExhaustedError
        device = DEVICES['DEFAULT']
        try:
            with tf.device(device):
                state = self._execute(initial_state=initial_state)
        except oom_error:
            raise_error(RuntimeError, f"State does not fit in {device} memory."
                                       "Please switch the execution device to a "
                                       "different one using ``qibo.set_device``.")
        return state

    def _repeated_execute(self, nreps: int,
                          initial_state: Optional[InitStateType] = None
                          ) -> tf.Tensor:
        results = []
        for _ in range(nreps):
            state = self._device_execute(initial_state)
            if self.measurement_gate is not None:
                results.append(self.measurement_gate(state, nshots=1)[0])
                del(state)
            else:
                results.append(tf.identity(state))
        results = tf.stack(results, axis=0)

        if self.measurement_gate is None:
            return results

        mgate_result = measurements.GateResult(
                self.measurement_gate.qubits, decimal_samples=results)
        return measurements.CircuitResult(self.measurement_tuples, mgate_result)

    def execute(self, initial_state: Optional[InitStateType] = None,
                nshots: Optional[int] = None) -> OutputType:
        """Propagates the state through the circuit applying the corresponding gates.

        In default usage the full final state vector is returned.
        If the circuit contains measurement gates and ``nshots`` is given, then
        the final state is sampled and the samples are returned. We refer to
        the :ref:`How to perform measurements? <measurement-examples>` example
        for more details on how to perform measurements in Qibo.

        If channels are found within the circuits gates then Qibo will perform
        the simulation by repeating the circuit execution ``nshots`` times.
        If the circuit contains measurements the corresponding noisy measurement
        result will be returned, otherwise the final state vectors will be
        collected to a ``(nshots, 2 ** nqubits)`` tensor and returned.
        The latter usage is memory intensive and not recommended.
        If the circuit is created with the ``density_matrix = True`` flag and
        contains channels, then density matrices will be used instead of
        repeated execution.
        Note that some channels (:class:`qibo.base.gates.KrausChannel`) can
        only be simulated using density matrices and not repeated execution.
        For more details on noise simulation with and without density matrices
        we refer to :ref:`How to perform noisy simulation? <noisy-example>`

        Args:
            initial_state (np.ndarray): Initial state vector as a numpy array of shape ``(2 ** nqubits,)``.
                A Tensorflow tensor with shape ``nqubits * (2,)`` is also allowed
                allowed as an initial state but must have the `dtype` of the circuit.
                If ``initial_state`` is ``None`` the |000...0> state will be used.
            nshots (int): Number of shots to sample if the circuit contains
                measurement gates.
                If ``nshots`` is ``None`` the measurement gates will be ignored.

        Returns:
            If ``nshots`` is given and the circuit contains measurements
                A :class:`qibo.base.measurements.CircuitResult` object that contains the measured bitstrings.
            If ``nshots`` is ``None`` or the circuit does not contain measurements.
                The final state vector as a Tensorflow tensor of shape ``(2 ** nqubits,)``.
        """
        if nshots is not None and self.repeated_execution:
            self._final_state = None
            return self._repeated_execute(nshots, initial_state)

        state = self._device_execute(initial_state)
        if self.measurement_gate is None or nshots is None:
            return state

        mgate_result = self.measurement_gate(state, nshots)
        return measurements.CircuitResult(self.measurement_tuples, mgate_result)

    @property
    def final_state(self) -> tf.Tensor:
        """Final state as a Tensorflow tensor of shape ``(2 ** nqubits,)``.

        The circuit has to be executed at least once before accessing this
        property, otherwise a ``ValueError`` is raised. If the circuit is
        executed more than once, only the last final state is returned.
        """
        if self._final_state is None:
            raise_error(RuntimeError, "Cannot access final state before the "
                                      "circuit is executed.")
        return self._final_state

    def _cast_initial_state(self, state: InitStateType) -> tf.Tensor:
        if isinstance(state, tf.Tensor):
            return state
        elif isinstance(state, np.ndarray):
            return tf.cast(state.astype(DTYPES.get('NPTYPECPX')),
                           dtype=DTYPES.get('DTYPECPX'))
        raise_error(TypeError, "Initial state type {} is not recognized."
                                "".format(type(state)))

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        zeros = tf.zeros(self.shapes.get('TF_FLAT'), dtype=DTYPES.get('DTYPECPX'))
        state = op.initial_state(zeros)
        return state

    def get_initial_state(self, state: Optional[InitStateType] = None
                           ) -> tf.Tensor:
        """"""
        if state is None:
            return self._default_initial_state()
        state = self._cast_initial_state(state)
        if self.check_initial_state_shape:
            shape = tuple(state.shape)
            if shape != self.shapes.get('FLAT'):
                raise_error(ValueError, "Invalid initial state shape {} for "
                                        "circuit with {} qubits."
                                        "".format(shape, self.nqubits))
        return state


class TensorflowDensityMatrixCircuit(TensorflowCircuit):
    """Implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

    Performs simulation using density matrices. Can be initialized using the
    ``density_matrix=True`` flag and supports the use of channels.
    For more information on the use of density matrices we refer to the
    :ref:`Using density matrices? <densitymatrix-example>` example.

    Example:
        ::

            from qibo import models, gates
            c = models.Circuit(2, density_matrix=True)
            c.add(gates.H(0))
            c.add(gates.PauliNoiseChannel(1, px=0.2))

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """

    def __init__(self, nqubits):
        super(TensorflowDensityMatrixCircuit, self).__init__(nqubits)
        self.density_matrix = True
        self.shapes = {
            'TENSOR': 2 * self.nqubits * (2,),
            'VECTOR': (2 ** nqubits,),
            'FLAT': 2 * (2 ** self.nqubits,)
        }
        self.shapes['TF_FLAT'] = tf.cast(self.shapes.get('FLAT'),
                                         dtype=DTYPES.get('DTYPEINT'))

    def _cast_initial_state(self, state: InitStateType) -> tf.Tensor:
        # Allow using state vectors as initial states but transform them
        # to the equivalent density matrix
        if tuple(state.shape) == self.shapes['VECTOR']:
            if isinstance(state, tf.Tensor):
                state = tf.tensordot(state, tf.math.conj(state), axes=0)
            elif isinstance(state, np.ndarray):
                state = np.outer(state, state.conj())
        return TensorflowCircuit._cast_initial_state(self, state)
