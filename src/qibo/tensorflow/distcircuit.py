# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
import joblib
from qibo.config import raise_error
from qibo.base import gates
from qibo import gates as gate_module
from qibo.tensorflow import callbacks, circuit, measurements
from qibo.tensorflow import distutils as utils
from qibo.tensorflow import custom_operators as op
from typing import Dict, List, Optional, Set, Tuple, Union
InitStateType = Union[np.ndarray, tf.Tensor, utils.DistributedState]
OutputType = Union[utils.DistributedState, measurements.CircuitResult]


class TensorflowDistributedCircuit(circuit.TensorflowCircuit):
    """Distributed implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

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
        super(TensorflowDistributedCircuit, self).__init__(nqubits)
        self._init_kwargs.update({"accelerators": accelerators,
                                  "memory_device": memory_device})
        self.ndevices = sum(accelerators.values())
        self.nglobal = float(np.log2(self.ndevices))
        if not (self.nglobal.is_integer() and self.nglobal > 0):
            raise_error(ValueError, "Number of calculation devices should be a power "
                                    "of 2 but is {}.".format(self.ndevices))
        self.nglobal = int(self.nglobal)
        self.nlocal = self.nqubits - self.nglobal

        self.memory_device = memory_device
        self.calc_devices = accelerators
        self.queues = utils.DistributedQueues(self, gate_module)

    def _set_nqubits(self, gate):
        # Do not set ``gate.nqubits`` during gate addition because this will
        # be set by ``self.queues`` when creating the gates on each device.
        if gate._nqubits is not None:
            raise_error(ValueError, "Attempting to add gate with preset number of "
                                    "qubits in distributed circuit.")

    def copy(self, deep: bool = True) -> "TensorflowDistributedCircuit":
        if not deep:
            raise_error(ValueError, "Non-deep copy is not allowed for distributed "
                                    "circuits because they modify gate objects.")
        return super(TensorflowDistributedCircuit, self).copy(deep)

    def _fuse_copy(self) -> "TensorflowDistributedCircuit":
        return self.copy(deep=True)

    def fuse(self) -> "TensorflowDistributedCircuit":
        if self.queues.queues:
            raise_error(RuntimeError, "Cannot fuse distributed circuit after "
                                      "its first execution.")
        return super(TensorflowDistributedCircuit, self).fuse()

    def with_noise(self, noise_map, measurement_noise=None):
        raise_error(NotImplementedError, "Distributed circuit does not support "
                                         "density matrices yet.")

    def _add(self, gate: gates.Gate):
        """Adds a gate in the circuit (inherited from :class:`qibo.base.circuit.BaseCircuit`).

        Also checks that there are sufficient qubits to use as global.
        """
        if not isinstance(gate, gate_module.TensorflowGate):
            raise_error(NotImplementedError, "Distributed circuit does not support "
                                             "native tensorflow gates.")
        if isinstance(gate, gates.VariationalLayer):
            gate._prepare()
        elif (self.nqubits - len(gate.target_qubits) < self.nglobal and
              not isinstance(gate, gates.M)):
            raise_error(ValueError, "Insufficient qubits to use for global in "
                                    "distributed circuit.")
        super(TensorflowDistributedCircuit, self)._add(gate)

    def compile(self):
        """"""
        raise_error(RuntimeError, "Cannot compile circuit that uses custom operators.")

    def _device_execute(self, state: tf.Tensor, gates: List["TensorflowGate"]) -> tf.Tensor:
        for gate in gates:
            state = gate(state)
        return state

    def _joblib_execute(self, state: utils.DistributedState,
                        queues: List[List["TensorflowGate"]]):
        """Executes gates in ``accelerators`` in parallel.

        Args:
            queues: List that holds the gates to be applied by each accelerator.
                Has shape ``(ndevices, ngates_i)`` where ``ngates_i`` is the
                number of gates to be applied by accelerator ``i``.
        """
        def _device_job(ids, device):
            for i in ids:
                with tf.device(device):
                    piece = self._device_execute(state.pieces[i], queues[i])
                    state.pieces[i].assign(piece)
                    del(piece)

        pool = joblib.Parallel(n_jobs=len(self.calc_devices),
                               prefer="threads")
        pool(joblib.delayed(_device_job)(ids, device)
             for device, ids in self.queues.device_to_ids.items())

    def _swap(self, state: utils.DistributedState, global_qubit: int, local_qubit: int):
        m = self.queues.qubits.reduced_global[global_qubit]
        m = self.nglobal - m - 1
        t = 1 << m
        for g in range(self.ndevices // 2):
            i = ((g >> m) << (m + 1)) + (g & (t - 1))
            local_eff = self.queues.qubits.reduced_local[local_qubit]
            with tf.device(self.memory_device):
                op.swap_pieces(state.pieces[i], state.pieces[i + t],
                               local_eff, self.nlocal)

    def _revert_swaps(self, state: utils.DistributedState, swap_pairs: List[Tuple[int, int]]):
        for q1, q2 in swap_pairs:
            if q1 not in self.queues.qubits.set:
                q1, q2 = q2, q1
            self._swap(state, q1, q2)

    def _special_gate_execute(self, state: utils.DistributedState,
                              gate: Union["TensorflowGate"]):
        """Executes special gates on ``memory_device``.

        Currently special gates are ``Flatten`` or ``CallbackGate``.
        This method calculates the full state vector because special gates
        are not implemented for state pieces.
        """
        with tf.device(self.memory_device):
            # Reverse all global SWAPs that happened so far
            self._revert_swaps(state, reversed(gate.swap_reset))
            full_state = state.vector
            if isinstance(gate, gates.CallbackGate):
                gate(full_state)
            else:
                full_state = gate(full_state)
                state.assign_vector(full_state)
            # Redo all global SWAPs that happened so far
            self._revert_swaps(state, gate.swap_reset)

    def _execute(self, initial_state: Optional[InitStateType] = None,
                 nshots: Optional[int] = None) -> OutputType:
        """Performs ``circuit.execute``."""
        state = self.get_initial_state(initial_state)

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
        if self.measurement_gate is None or nshots is None:
            return state

        with tf.device(self.memory_device):
            samples = self.measurement_gate(state.vector, nshots, samples_only=True,
                                            is_density_matrix=self.using_density_matrix)
            self.measurement_gate_result = measurements.GateResult(
                self.measurement_gate.qubits, state, decimal_samples=samples)
            result = measurements.CircuitResult(
                self.measurement_tuples, self.measurement_gate_result)
        return result

    def execute(self, initial_state: Optional[InitStateType] = None,
                nshots: Optional[int] = None) -> OutputType:
        """Equivalent to :meth:`qibo.tensorflow.circuit.TensorflowCircuit.execute`.

        If measurements are not specified this returns a
        :class:`qibo.tensorflow.distutils.DistributedState` instead of a
        ``tf.Tensor``. This avoids creating multiple copies of large states in
        the CPU memory.
        """
        oom_error = tf.python.framework.errors_impl.ResourceExhaustedError
        try:
            return self._execute(initial_state=initial_state, nshots=nshots)
        except oom_error:
            raise_error(RuntimeError, "State does not fit in memory during distributed "
                                      "execution. Please create a new circuit with "
                                      "different device configuration and try again.")

    def get_initial_state(
          self, state: Optional[Union[InitStateType, str]] = None
          ) -> tf.Tensor:
        """"""
        if not self.queues.queues and self.queue:
            self.queues.set(self.queue)
        if state is None:
            return utils.DistributedState.default(self)
        elif isinstance(state, str):
            return getattr(utils.DistributedState, state)(self)
        elif isinstance(state, utils.DistributedState):
            return state
        full_state = super(TensorflowDistributedCircuit,
                           self).get_initial_state(state)
        return utils.DistributedState.from_vector(full_state, self)
