# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import collections
from qibo import K
from qibo.abstractions import circuit
from qibo.config import raise_error
from qibo.core import states, measurements
from typing import List, Tuple


class Circuit(circuit.AbstractCircuit):
    """Backend implementation of :class:`qibo.abstractions.circuit.AbstractCircuit`.

    Performs simulation using state vectors.

    Example:
        .. testcode::

            from qibo import models, gates
            c = models.Circuit(3) # initialized circuit with 3 qubits
            c.add(gates.H(0)) # added Hadamard gate on qubit 0

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """

    def __init__(self, nqubits):
        super(Circuit, self).__init__(nqubits)
        self.param_tensor_types = K.tensor_types
        self._compiled_execute = None
        self.state_cls = states.VectorState

    def _set_nqubits(self, gate):
        if gate._nqubits is not None and gate.nqubits != self.nqubits:
            raise_error(RuntimeError, "Cannot add gate {} that acts on {} "
                                      "qubits to circuit that contains {}"
                                      "qubits.".format(
                                            gate, gate.nqubits, self.nqubits))
        gate.nqubits = self.nqubits

    def _add_layer(self, gate):
        for unitary in gate.unitaries:
            self._set_nqubits(unitary)
            self.queue.append(unitary)
        if gate.additional_unitary is not None:
            self._set_nqubits(gate.additional_unitary)
            self.queue.append(gate.additional_unitary)

    def unitary(self):
        """Creates the unitary matrix corresponding to all circuit gates.
        
        This is a ``(2 ** nqubits, 2 ** nqubits)`` matrix obtained by 
        multiplying all circuit gates.
        """
        from qibo import gates
        fgate = gates.FusedGate(*range(self.nqubits))
        for gate in self.queue:
            fgate.append(gate)
        return fgate.matrix

    def fuse(self, max_qubits=2):
        """Creates an equivalent circuit by fusing gates for increased simulation performance.

        Args:
            max_qubits (int): Maximum number of qubits in the fused gates.

        Returns:
            A :class:`qibo.core.circuit.Circuit` object containing
            :class:`qibo.abstractions.gates.FusedGate` gates, each of which
            corresponds to a group of some original gates.
            For more details on the fusion algorithm we refer to the
            :ref:`Circuit fusion <circuit-fusion>` section.

        Example:
            .. testcode::

                from qibo import models, gates
                c = models.Circuit(2)
                c.add([gates.H(0), gates.H(1)])
                c.add(gates.CNOT(0, 1))
                c.add([gates.Y(0), gates.Y(1)])
                # create circuit with fused gates
                fused_c = c.fuse()
                # now ``fused_c`` contains a single ``FusedGate`` that is
                # equivalent to applying the five original gates
        """
        queue = self.queue.to_fused()
        for gate in queue:
            if not gate.marked:
                for q in gate.qubits:
                    # fuse nearest neighbors forth in time
                    neighbor = gate.right_neighbors.get(q)
                    if gate.can_fuse(neighbor, max_qubits):
                        gate.fuse(neighbor)
                    # fuse nearest neighbors back in time
                    neighbor = gate.left_neighbors.get(q)
                    if gate.can_fuse(neighbor, max_qubits):
                        neighbor.fuse(gate)
        # create a circuit and assign the new queue
        circuit = self._shallow_copy()
        circuit.queue = queue.from_fused()
        return circuit

    def _eager_execute(self, state):
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
        for gate in self.queue:
            # create gate cache before compilation
            _ = gate.cache
        self._compiled_execute = K.compile(self._execute_for_compile)

    def _execute(self, initial_state=None):
        """Performs all circuit gates on the state vector."""
        self._final_state = None
        state = self.get_initial_state(initial_state)

        if self._compiled_execute is None:
            state = self._eager_execute(state)
        else:
            state, callback_results = self._compiled_execute(state)
            for callback, results in callback_results.items():
                callback.extend(results)

        self._final_state = self.state_cls.from_tensor(state, self.nqubits)
        return self._final_state

    def _device_execute(self, initial_state=None):
        """Executes circuit on the specified device and checks for OOM errors."""
        device = K.default_device
        try:
            with K.device(device):
                state = self._execute(initial_state=initial_state)
        except K.oom_error:
            raise_error(RuntimeError, f"State does not fit in {device} memory."
                                       "Please switch the execution device to a "
                                       "different one using ``qibo.set_device``.")
        return state

    def _device(self):
        from qibo.backends.numpy import NumpyBackend
        return NumpyBackend.DummyModule()

    def _repeated_execute(self, nreps, initial_state=None):
        results = []
        initial_state = self.get_initial_state(initial_state)
        for _ in range(nreps):
            with self._device():
                state = K.copy(initial_state)
            state = self._device_execute(state)
            if self.measurement_gate is None:
                results.append(state.tensor)
            else:
                state.measure(self.measurement_gate, nshots=1)
                results.append(state.measurements[0])
                del(state)

        with self._device():
            results = K.stack(results, axis=0)
        if self.measurement_gate is None:
            return results
        state = self.state_cls(self.nqubits)
        state.set_measurements(self.measurement_gate.qubits, results, self.measurement_tuples)
        return state

    def execute(self, initial_state=None, nshots=None):
        """Propagates the state through the circuit applying the corresponding gates.

        If channels are found within the circuits gates then Qibo will perform
        the simulation by repeating the circuit execution ``nshots`` times.
        If the circuit contains measurements the corresponding noisy measurement
        result will be returned, otherwise the final state vectors will be
        collected to a ``(nshots, 2 ** nqubits)`` tensor and returned.
        The latter usage is memory intensive and not recommended.
        If the circuit is created with the ``density_matrix = True`` flag and
        contains channels, then density matrices will be used instead of
        repeated execution.
        Note that some channels (:class:`qibo.abstractions.gates.KrausChannel`) can
        only be simulated using density matrices and not repeated execution.
        For more details on noise simulation with and without density matrices
        we refer to :ref:`How to perform noisy simulation? <noisy-example>`

        Args:
            initial_state (array): Initial state vector as a numpy array of shape ``(2 ** nqubits,)``.
                A Tensorflow tensor with shape ``nqubits * (2,)`` is also allowed
                allowed as an initial state but must have the `dtype` of the circuit.
                If ``initial_state`` is ``None`` the ``|000...0>`` state will be used.
            nshots (int): Number of shots to sample if the circuit contains
                measurement gates.
                If ``nshots`` is ``None`` the measurement gates will be ignored.

        Returns:
            A :class:`qibo.abstractions.states.AbstractState` object which
            holds the final state vector as a tensor of shape ``(2 ** nqubits,)``
            or the final density matrix as a tensor of shpae
            ``(2 ** nqubits, 2 ** nqubits)``.
            If ``nshots`` is given and the circuit contains measurements
            the returned circuit object also contains the measured bitstrings.
        """
        if nshots is not None and self.repeated_execution:
            self._final_state = None
            return self._repeated_execute(nshots, initial_state)

        state = self._device_execute(initial_state)
        if self.measurement_gate is not None and nshots is not None:
            with self._device():
                state.measure(self.measurement_gate, nshots, self.measurement_tuples)
        return state

    @property
    def final_state(self):
        """Final state as a tensor of shape ``(2 ** nqubits,)``.

        The circuit has to be executed at least once before accessing this
        property, otherwise a ``ValueError`` is raised. If the circuit is
        executed more than once, only the last final state is returned.
        """
        if self._final_state is None:
            raise_error(RuntimeError, "Cannot access final state before the "
                                      "circuit is executed.")
        return self._final_state

    def get_initial_state(self, state=None):
        """"""
        if state is None:
            state = self.state_cls.zero_state(self.nqubits)
        elif not isinstance(state, self.state_cls):
            state = self.state_cls.from_tensor(state, self.nqubits)
        return state.tensor


class DensityMatrixCircuit(Circuit):
    """Backend implementation of :class:`qibo.abstractions.circuit.AbstractCircuit`.

    Performs simulation using density matrices. Can be initialized using the
    ``density_matrix=True`` flag and supports the use of channels.
    For more information on the use of density matrices we refer to the
    :ref:`Using density matrices? <densitymatrix-example>` example.

    Example:
        .. testcode::

            from qibo import models, gates
            c = models.Circuit(2, density_matrix=True)
            c.add(gates.H(0))
            c.add(gates.PauliNoiseChannel(1, px=0.2))

    Args:
        nqubits (int): Total number of qubits in the circuit.
    """

    def __init__(self, nqubits):
        super(DensityMatrixCircuit, self).__init__(nqubits)
        self.density_matrix = True
        self.state_cls = states.MatrixState

    def get_initial_state(self, state=None):
        # Allow using state vectors as initial states but transform them
        # to the equivalent density matrix
        if state is not None and tuple(state.shape) == (2 ** self.nqubits,):
            state = states.VectorState.from_tensor(state, self.nqubits)
            return state.to_density_matrix().tensor
        return super().get_initial_state(state)
