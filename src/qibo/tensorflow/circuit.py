# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import collections
from qibo import K
from qibo.base import circuit
from qibo.config import DEVICES, BACKEND, raise_error
from qibo.tensorflow import measurements
from qibo.tensorflow import custom_operators as op
from typing import List, Optional, Tuple, Union
OutputType = Union[K.tensortype, measurements.CircuitResult]


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
        self.shapes['TENSOR_FLAT'] = K.cast(self.shapes.get('FLAT'),
                                            dtype='DTYPEINT')

    def set_nqubits(self, gate):
        super().set_nqubits(gate)
        gate.nqubits = self.nqubits
        gate.prepare()

    def set_parameters(self, parameters):
        if isinstance(parameters, K.tensor_types):
            super()._set_parameters_list(parameters, int(parameters.shape[0]))
        else:
            super().set_parameters(parameters)

    def _get_parameters_flatlist(self, parametrized_gates):
        params = []
        for gate in parametrized_gates:
            if isinstance(gate.parameters, K.tensor_types):
                params.extend(gate.parameters.ravel())
            elif isinstance(gate.parameters, collections.abc.Iterable):
                params.extend(gate.parameters)
            else:
                params.append(gate.parameters)
        return params

    def _eager_execute(self, state: K.tensortype) -> K.tensortype:
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
        self._compiled_execute = K.compile(self._execute_for_compile)

    def _execute(self, initial_state: Optional[K.tensortype] = None
                 ) -> K.tensortype:
        """Performs all circuit gates on the state vector."""
        self._final_state = None
        state = self.get_initial_state(initial_state)
        if BACKEND['GATES'] != 'custom':
            state = K.reshape(state, self.shapes.get('TENSOR'))

        if self._compiled_execute is None:
            state = self._eager_execute(state)
        else:
            state, callback_results = self._compiled_execute(state)
            for callback, results in callback_results.items():
                callback.extend(results)

        if BACKEND['GATES'] != 'custom':
            state = K.reshape(state, self.shapes.get('TENSOR_FLAT'))

        self._final_state = state
        return state

    def _device_execute(self, initial_state: Optional[K.tensortype] = None
                        ) -> K.tensortype:
        """Executes circuit on the specified device and checks for OOM errors."""
        device = DEVICES['DEFAULT']
        try:
            with K.device(device):
                state = self._execute(initial_state=initial_state)
        except K.oom_error:
            raise_error(RuntimeError, f"State does not fit in {device} memory."
                                       "Please switch the execution device to a "
                                       "different one using ``qibo.set_device``.")
        return state

    def _repeated_execute(self, nreps: int,
                          initial_state: Optional[K.tensortype] = None
                          ) -> K.tensortype:
        results = []
        for _ in range(nreps):
            state = self._device_execute(initial_state)
            if self.measurement_gate is not None:
                results.append(self.measurement_gate(state, nshots=1)[0])
                del(state)
            else:
                results.append(K.copy(state))
        results = K.stack(results, axis=0)

        if self.measurement_gate is None:
            return results

        mgate_result = measurements.GateResult(
                self.measurement_gate.qubits, decimal_samples=results)
        return measurements.CircuitResult(self.measurement_tuples, mgate_result)

    def execute(self, initial_state: Optional[K.tensortype] = None,
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
            initial_state (array): Initial state vector as a numpy array of shape ``(2 ** nqubits,)``.
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
    def final_state(self) -> K.tensortype:
        """Final state as a Tensorflow tensor of shape ``(2 ** nqubits,)``.

        The circuit has to be executed at least once before accessing this
        property, otherwise a ``ValueError`` is raised. If the circuit is
        executed more than once, only the last final state is returned.
        """
        if self._final_state is None:
            raise_error(RuntimeError, "Cannot access final state before the "
                                      "circuit is executed.")
        return self._final_state

    def _cast_initial_state(self, state: K.tensortype) -> K.tensortype:
        if isinstance(state, K.tensor_types):
            return K.cast(state)
        raise_error(TypeError, "Initial state type {} is not recognized."
                                "".format(type(state)))

    def _default_initial_state(self) -> K.tensortype:
        """Creates the |000...0> state for default initialization."""
        zeros = K.zeros(self.shapes.get('TENSOR_FLAT'))
        state = op.initial_state(zeros)
        return state

    def get_initial_state(self, state: Optional[K.tensortype] = None
                           ) -> K.tensortype:
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
        self.shapes['TENSOR_FLAT'] = K.cast(self.shapes.get('FLAT'),
                                            dtype='DTYPEINT')

    def _cast_initial_state(self, state: K.tensortype) -> K.tensortype:
        # Allow using state vectors as initial states but transform them
        # to the equivalent density matrix
        if tuple(state.shape) == self.shapes['VECTOR']:
            state = K.outer(state, K.conj(state))
        return TensorflowCircuit._cast_initial_state(self, state)
