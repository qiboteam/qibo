# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import math
import joblib
from qibo import K
from qibo import gates as gate_module
from qibo.abstractions import gates
from qibo.abstractions.circuit import AbstractCircuit
from qibo.config import raise_error, get_threads
from qibo.core import callbacks, circuit, measurements, states
from qibo.tensorflow.distutils import DistributedQueues
from typing import Dict, List, Optional, Set, Tuple, Union


class DistributedCircuit(circuit.Circuit):
    """Distributed implementation of :class:`qibo.abstractions.circuit.AbstractCircuit` in Tensorflow.

    Uses multiple `accelerator` devices (GPUs) for applying gates to the state vector.
    The full state vector is saved in the given `memory device` (usually the CPU)
    during the simulation. A gate is applied by splitting the state to pieces
    and copying each piece to an accelerator device that is used to perform the
    matrix multiplication. An `accelerator` device can be used more than once
    resulting to logical devices that are more than the physical accelerators in
    the system.

    Distributed circuits currently do not support native tensorflow gates,
    compilation and callbacks.

    Example:
        ::

            from qibo.models import Circuit
            # The system has two GPUs and we would like to use each GPU twice
            # resulting to four total logical accelerators
            accelerators = {'/GPU:0': 2, '/GPU:1': 2}
            # Define a circuit on 32 qubits to be run in the above GPUs keeping
            # the full state vector in the CPU memory.
            c = Circuit(32, accelerators, memory_device="/CPU:0")

    Args:
        nqubits (int): Total number of qubits in the circuit.
        accelerators (dict): Dictionary that maps device names to the number of
            times each device will be used.
            The total number of logical devices must be a power of 2.
        memory_device (str): Name of the device where the full state will be
            saved (usually the CPU).
    """

    def __init__(self,
                 nqubits: int,
                 accelerators: Dict[str, int],
                 memory_device: str = "/CPU:0"):
        super().__init__(nqubits)
        self.init_kwargs.update({"accelerators": accelerators,
                                 "memory_device": memory_device})
        self.ndevices = sum(accelerators.values())
        self.nglobal = float(math.log2(self.ndevices))
        if not (self.nglobal.is_integer() and self.nglobal > 0):
            raise_error(ValueError, "Number of calculation devices should be a power "
                                    "of 2 but is {}.".format(self.ndevices))
        self.nglobal = int(self.nglobal)
        self.nlocal = self.nqubits - self.nglobal

        self.memory_device = memory_device
        self.calc_devices = accelerators
        self.queues = DistributedQueues(self, gate_module)

    def set_nqubits(self, gate):
        AbstractCircuit.set_nqubits(self, gate)

    def on_qubits(self, *q):
        if self.queues.queues:
            raise_error(RuntimeError, "Cannot use distributed circuit as a "
                                      "subroutine after it was executed.")
        return super().on_qubits(*q)

    def copy(self, deep: bool = True):
        if not deep:
            raise_error(ValueError, "Non-deep copy is not allowed for distributed "
                                    "circuits because they modify gate objects.")
        return super().copy(deep)

    def _fuse_copy(self):
        return self.copy(deep=True)

    def fuse(self):
        if self.queues.queues:
            raise_error(RuntimeError, "Cannot fuse distributed circuit after "
                                      "its first execution.")
        return super().fuse()

    def with_noise(self, noise_map, measurement_noise=None):
        raise_error(NotImplementedError, "Distributed circuit does not support "
                                         "density matrices yet.")

    def _add(self, gate: gates.Gate):
        """Adds a gate in the circuit (inherited from :class:`qibo.abstractions.circuit.AbstractCircuit`).

        Also checks that there are sufficient qubits to use as global.
        """
        if K.name != "custom":
            raise_error(NotImplementedError, "Distributed circuit does not "
                                             "support native tensorflow gates.")
        if isinstance(gate, gates.KrausChannel):
            raise_error(NotImplementedError, "Distributed circuits do not "
                                             "support channels.")
        elif (self.nqubits - len(gate.target_qubits) < self.nglobal and
              not isinstance(gate, (gates.M, gates.VariationalLayer))):
            raise_error(ValueError, "Insufficient qubits to use for global in "
                                    "distributed circuit.")
        return super()._add(gate)

    def compile(self):
        """"""
        raise_error(RuntimeError, "Cannot compile circuit that uses custom operators.")

    def _device_job(self, state, gates):
        for gate in gates:
            state = gate(state)
        return state

    def _joblib_execute(self, state, queues: List[List["BackendGate"]]):
        """Executes gates in ``accelerators`` in parallel.

        Args:
            queues: List that holds the gates to be applied by each accelerator.
                Has shape ``(ndevices, ngates_i)`` where ``ngates_i`` is the
                number of gates to be applied by accelerator ``i``.
        """
        def device_job(ids, device):
            for i in ids:
                with K.device(device):
                    piece = self._device_job(state.pieces[i], queues[i])
                    state.pieces[i].assign(piece)
                    del(piece)

        pool = joblib.Parallel(n_jobs=len(self.calc_devices),
                               prefer="threads")
        pool(joblib.delayed(device_job)(ids, device)
             for device, ids in self.queues.device_to_ids.items())

    def _swap(self, state, global_qubit: int, local_qubit: int):
        m = self.queues.qubits.reduced_global[global_qubit]
        m = self.nglobal - m - 1
        t = 1 << m
        for g in range(self.ndevices // 2):
            i = ((g >> m) << (m + 1)) + (g & (t - 1))
            local_eff = self.queues.qubits.reduced_local[local_qubit]
            with K.device(self.memory_device):
                K.op.swap_pieces(state.pieces[i], state.pieces[i + t],
                                 local_eff, self.nlocal, get_threads())

    def _revert_swaps(self, state, swap_pairs: List[Tuple[int, int]]):
        for q1, q2 in swap_pairs:
            if q1 not in self.queues.qubits.set:
                q1, q2 = q2, q1
            self._swap(state, q1, q2)

    def _special_gate_execute(self, state, gate: Union["BackendGate"]):
        """Executes special gates on ``memory_device``.

        Currently special gates are ``Flatten`` or ``CallbackGate``.
        This method calculates the full state vector because special gates
        are not implemented for state pieces.
        """
        with K.device(self.memory_device):
            # Reverse all global SWAPs that happened so far
            self._revert_swaps(state, reversed(gate.swap_reset))
            full_state = state.tensor
            if isinstance(gate, gates.CallbackGate):
                gate(full_state)
            else:
                full_state = gate(full_state)
                state.assign_pieces(full_state)
            # Redo all global SWAPs that happened so far
            self._revert_swaps(state, gate.swap_reset)

    def _execute(self, initial_state=None):
        """Performs all circuit gates on the state vector."""
        self._final_state = None
        state = self.get_initial_state(initial_state)
        if self.measurement_gate is not None:
            self.measurement_gate.device = self.memory_device

        special_gates = iter(self.queues.special_queue)
        for i, queues in enumerate(self.queues.queues):
            if queues:  # standard gate
                self._joblib_execute(state, queues)
            else: # special gate
                gate = next(special_gates)
                if isinstance(gate, tuple): # SWAP global-local qubit
                    self._swap(state, *gate)
                else:
                    self._special_gate_execute(state, gate)
        for gate in special_gates: # pragma: no cover
            self._special_gate_execute(state, gate)

        self._final_state = state
        return state

    def _device_execute(self, initial_state=None):
        """Executes circuit and checks for OOM errors."""
        try:
            return self._execute(initial_state)
        except K.oom_error:
            raise_error(RuntimeError, "State does not fit in memory during distributed "
                                      "execution. Please create a new circuit with "
                                      "different device configuration and try again.")

    def execute(self, initial_state=None, nshots=None):
        """Equivalent to :meth:`qibo.core.circuit.Circuit.execute`.

        Returns:
            A :class:`qibo.core.states.DistributedState` object corresponding
            to the final state of execution. Note that this state contains the
            full state vector scattered to pieces and does not create a
            single tensor unless the user explicitly calls the ``tensor``
            property. This avoids creating multiple copies of large states in
            CPU memory.
        """
        return super().execute(initial_state=initial_state, nshots=nshots)

    def get_initial_state(self, state=None):
        """"""
        if not self.queues.queues and self.queue:
            self.queues.set(self.queue)

        if state is None:
            return states.DistributedState.zero_state(self)
        elif isinstance(state, states.DistributedState):
            state.circuit = self
            return state
        elif isinstance(state, K.tensor_types):
            state = super().get_initial_state(state)
            return states.DistributedState.from_tensor(state, self)

        raise_error(TypeError, "Initial state type {} is not supported by "
                               "distributed circuits.".format(type(state)))
