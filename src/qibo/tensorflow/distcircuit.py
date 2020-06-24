# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import copy
import numpy as np
import tensorflow as tf
import joblib
from qibo.config import DTYPES
from qibo.base import gates
from qibo.tensorflow import callbacks, circuit, measurements
from qibo.tensorflow import custom_operators as op
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


class DeviceQueues:
    """Data structure that holds gate queues for each accelerator device.

    For a distributed simulation we have to swap global qubits multiple times.
    For each global qubit configuration a several gates can be applied to the
    state forming a gate group. Once all gates in the group are applied the
    global qubits are swapped and we proceed to the next gate group.

    ``DeviceQueues`` holds the following data that define the gate groups and
    corresponding global qubits:
    * ``ndevices``: Number of logical accelerator devices.
    * ``nglobal``: Number of global qubits (= log2(ndevices)).
    * ``global_qubits_lists``: List of sorted global qubit lists. Each list
        corresponds to a gate group.
    * ``global_qubits_sets``: List of global qubit sets. This follows ``global_qubits_lists``
        but a set is used to allow O(1) search.
    * ``device_to_ids``: Dictionary that maps device (str) to list of piece indices.
        When a device is used multiple times then it is responsible for updating
        multiple state pieces. The list of indices specifies which pieces the device
        will update.
    * ``ids_to_device``: Inverse dictionary of ``device_to_ids``.
    * ``queues``: List of length ``ndevices``. For each device we store the queues
        of each gate group in a list of lists.
        For example ``queues[2][1]`` gives the queue (list) of the 1st gate group
        to be run in the 2nd device.
    * ``special_queue``: List with special gates than run on the full state vector
        on ``memory_device``. Special gates have no target qubits and currently
        are the ``CallbackGate`` and ``Flatten``.
    """

    def __init__(self, calc_devices: Dict[str, int]):
        self.ndevices = sum(calc_devices.values())
        self.nglobal = int(np.log2(self.ndevices))

        self.queues = [[] for _ in range(self.ndevices)]
        self.special_queue = []

        self.global_qubits_lists = []
        self.global_qubits_sets = []

        self.device_to_ids = {d: v for d, v in self._ids_gen(calc_devices)}
        self.ids_to_device = self.ndevices * [None]
        for device, ids in self.device_to_ids.items():
            for i in ids:
                self.ids_to_device[i] = device

    @staticmethod
    def _ids_gen(calc_devices) -> Tuple[str, List[int]]:
        """Generator of device piece indices."""
        start = 0
        for device, n in calc_devices.items():
            stop = start + n
            yield device, list(range(start, stop))
            start = stop

    def append(self, qubits):
        """Appends a new global qubit lists.

        Appending a global qubit lists defines a new group.

        Args:
            qubits: Any iterable that contains the global qubit ids.
        """
        self.global_qubits_sets.append(set(qubits))
        self.global_qubits_lists.append(sorted(qubits))

    def __len__(self) -> int:
        """Number of different gate groups."""
        return len(self.global_qubits_lists)

    def _create_reduced_gate(self, iq: int, gate: gates.Gate) -> gates.Gate:
        """Creates a copy of a gate for specific device application.

        Target and control qubits are modified according to the local qubits of
        the circuit when this gate will be applied.

        Args:
            iq (int): Index of the ``self.global_qubits_lists`` that corresponds
                to the gate's gate group.
            gate: The :class:`qibo.base.gates.Gate` object of the gate to copy.

        Returns:
            A :class:`qibo.base.gates.Gate` object with the proper target and
            control qubit indices for device-specific application.
        """
        def reduction_number(q: int) -> int:
            for i, gq in enumerate(self.global_qubits_lists[iq]):
                if gq > q:
                    return i
            return i + 1

        calc_gate = copy.copy(gate)
        # Recompute the target/control indices considering only local qubits.
        calc_gate.target_qubits = tuple(q - reduction_number(q)
                                        for q in calc_gate.target_qubits)
        calc_gate.control_qubits = tuple(q - reduction_number(q)
                                         for q in calc_gate.control_qubits
                                         if q not in self.global_qubits_sets[iq])
        calc_gate.original_gate = gate
        return calc_gate

    def create(self, queues: List[List[gates.Gate]], nlocal: int):
        """Creates the gate objects for each device and stores them in ``self.queues``.

        Args:
            queues (list): List of gate queues that defines the gate groups using
                general (non device specific gates).
            nlocal: Number of local qubits in the circuit (``= nqubits - nglobal``).
        """
        if len(queues) != len(self):
            raise ValueError("Global qubits lists are {} while given queues "
                             "are {}.".format(len(self), len(queues)))
        for iq, queue in enumerate(queues):
            if not self.global_qubits_lists[iq]:
                assert len(queue) == 1
                gate = queue[0]
                gate.nqubits = self.nglobal + nlocal
                self.special_queue.append(gate)
            else:
                for i in range(self.ndevices):
                    self.queues[i].append([])
                for gate in queue:
                    for device, ids in self.device_to_ids.items():
                        calc_gate = self._create_reduced_gate(iq, gate)
                        # Gate matrix should be constructed in the calculation
                        # device otherwise device parallelization will break
                        with tf.device(device):
                            calc_gate.nqubits = nlocal
                        for i in ids:
                            flag = True
                            # If there are control qubits that are global then
                            # the gate should not be applied by all devices
                            for control in (set(gate.control_qubits) &
                                            self.global_qubits_sets[iq]):
                                ic = self.global_qubits_lists[iq].index(control)
                                ic = self.nglobal - ic - 1
                                flag = bool((i // (2 ** ic)) % 2)
                                if not flag:
                                    break
                            if flag:
                                self.queues[i][-1].append(calc_gate)


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
        self.nglobal = np.log2(self.ndevices)
        if not (self.nglobal.is_integer() and self.nglobal > 0):
            raise ValueError("Number of calculation devices should be a power "
                             "of 2 but is {}.".format(self.ndevices))
        self.nglobal = int(self.nglobal)

        self.memory_device = memory_device
        self.calc_devices = accelerators

        self.device_queues = DeviceQueues(accelerators)
        self.pieces = None
        self._global_qubits = None
        self._construct_shapes()

    def _construct_shapes(self):
        """Useful shapes for the simulation."""
        dtype = DTYPES.get('DTYPEINT')
        n = self.nqubits - self.nglobal
        self.device_shape = tf.cast((self.ndevices, 2 ** n), dtype=dtype)
        self.full_shape = tf.cast((2 ** self.nqubits,), dtype=dtype)
        self.tensor_shape = self.nqubits * (2,)

        self.local_full_shape = tf.cast((2 ** n,), dtype=dtype)
        self.local_tensor_shape = n * (2,)

    @property
    def global_qubits(self) -> List[int]:
        """Returns the global qubits IDs in a sorted list.

        The global qubits are used to split the state to multiple pieces.
        Gates that have global qubits as their target qubits cannot be applied
        using the accelerators. In order to apply such gates we have to swap
        the target global qubit with a different (local) qubit.
        """
        if self._global_qubits is None:
            raise ValueError("Cannot access global qubits before being set.")
        return sorted(self._global_qubits)

    @global_qubits.setter
    def global_qubits(self, x: Sequence[int]):
        """Sets the current global qubits.

        At the same time the ``transpose_order`` and ``reverse_transpose_order``
        lists are set. These lists are used in order to transpose the state pieces
        when we want to swap global qubits.
        """
        if len(x) != self.nglobal:
            raise ValueError("Invalid number of global qubits {} for using {} "
                             "calculation devices.".format(len(x), self.ndevices))
        self._global_qubits = set(x)
        local_qubits = [i for i in range(self.nqubits) if i not in self._global_qubits]

        self.transpose_order = list(sorted(self._global_qubits)) + local_qubits
        self.reverse_transpose_order = self.nqubits * [0]
        for i, v in enumerate(self.transpose_order):
            self.reverse_transpose_order[v] = i

    def _set_nqubits(self, gate):
        # Do not set ``gate.nqubits`` during gate addition because this will
        # be set by the ``set_gates`` method once all gates are known.
        pass

    def with_noise(self, noise_map, measurement_noise=None):
        raise NotImplementedError("Distributed circuit does not support "
                                  "density matrices yet.")

    def _add(self, gate: gates.Gate):
        """Adds a gate in the circuit (inherited from :class:`qibo.base.circuit.BaseCircuit`).

        Also checks that there are sufficient qubits to use as global.
        """
        if isinstance(gate, gates.VariationalLayer):
            gate._prepare()
        elif (self.nqubits - len(gate.target_qubits) < self.nglobal and
              not isinstance(gate, gates.M)):
            raise ValueError("Insufficient qubits to use for global in "
                             "distributed circuit.")
        super(TensorflowDistributedCircuit, self)._add(gate)

    def set_gates(self):
        """Prepares gates for device-specific gate execution.

        Each gate has to be recreated in the device that will be executed to
        allow parallel execution. The global qubit lists and gate groups are
        also specified here.
        A gate group is identified by looping through the circuit's gate queue
        and adding gates in the group until the number of global becomes ``nglobal``.
        Once this happens no more gates can be added in the group. In order to
        apply new gates some global qubits have to be swapped to global and a
        new gate group will be defined for the new global qubit configuration.

        The final global qubit lists and gate queues that are used for execution
        are storred in ``self.device_queues`` which is a ``DeviceQueues`` object.
        """
        if not self.queue:
            raise RuntimeError("No gates available to set for distributed run.")

        all_qubits = set(range(self.nqubits))
        queues = [[]]

        global_qubits = set(all_qubits)
        queue = iter(self.queue)
        try:
            # Loop through the gate queue to define gate groups and
            # global qubit lists
            gate = next(queue)
            # special gates are gates without target qubits
            # (eg. ``CallbackGate`` or ``Flatten``)
            special_gates = []
            while True:
                target_qubits = set(gate.target_qubits)
                if not target_qubits:
                    special_gates.append(gate)
                else:
                    global_qubits -= target_qubits
                while len(global_qubits) > self.nglobal and not special_gates:
                    queues[-1].append(gate)
                    gate = next(queue)
                    target_qubits = set(gate.target_qubits)
                    if not target_qubits:
                        special_gates.append(gate)
                    else:
                        global_qubits -= target_qubits

                if len(global_qubits) == self.nglobal:
                    queues[-1].append(gate)
                    if not special_gates:
                        gate = next(queue)
                        target_qubits = set(gate.target_qubits)
                        if not target_qubits:
                            special_gates.append(gate)
                        while not target_qubits & global_qubits and not special_gates:
                            queues[-1].append(gate)
                            gate = next(queue)
                            target_qubits = set(gate.target_qubits)
                            if not target_qubits:
                                special_gates.append(gate)

                elif len(global_qubits) > self.nglobal:
                    global_qubits = list(sorted(global_qubits))[:self.nglobal]

                else:
                    # must be len(global_qubits) < self.nglobal
                    free_qubits = list(sorted(target_qubits))
                    global_qubits |= set(free_qubits[self.nglobal - len(global_qubits):])

                self.device_queues.append(global_qubits)
                if special_gates:
                    self.device_queues.append([])
                    queues.append([special_gates.pop()])
                    gate = next(queue)

                queues.append([])
                global_qubits = set(all_qubits)

        except StopIteration:
            if len(global_qubits) > self.nglobal:
                global_qubits = list(sorted(global_qubits))[:self.nglobal]
            if target_qubits:
                self.device_queues.append(global_qubits)

        self.device_queues.create(queues, nlocal=self.nqubits - self.nglobal)

    def compile(self):
        raise RuntimeError("Cannot compile circuit that uses custom operators.")

    def _device_execute(self, state: tf.Tensor, gates: List["TensorflowGate"]) -> tf.Tensor:
        for gate in gates:
            state = gate(state)
        return state

    # Old casting on CPU after runs finish. Not used because it leads to
    # GPU memory errors
    #def _cast_results(self, results: List[List[tf.Tensor]]):
    #    i = 0
    #    for result in results:
    #        for s in result:
    #            self.pieces[i].assign(s)
    #            i += 1

    def _joblib_execute(self, group: int):
        """Executes gates in ``accelerators`` in parallel."""
        def _device_job(ids, device):
            for i in ids:
                with tf.device(device):
                    state = self._device_execute(
                        self.pieces[i], self.device_queues.queues[i][group])
                    self.pieces[i].assign(state)
                    del(state)

        pool = joblib.Parallel(n_jobs=len(self.calc_devices),
                               prefer="threads")
        pool(joblib.delayed(_device_job)(ids, device)
             for device, ids in self.device_queues.device_to_ids.items())

    def _special_gate_execute(self, ispecial: int):
        """Executes special gates (``Flatten`` or ``CallbackGate``) on ``memory_device``.

        These gates require the full state vector (cannot be executed in the state pieces).
        """
        state = self._merge()
        gate = self.device_queues.special_queue[ispecial]
        if isinstance(gate, gates.CallbackGate):
            gate(state)
        else:
            state = gate(state)
            self._split(state)

    def execute(self,
                initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None,
                nshots: Optional[int] = None,
                ) -> Union[tf.Tensor, measurements.CircuitResult]:
        """Same as the ``execute`` method of :class:`qibo.tensorflow.circuit.TensorflowCircuit`."""
        if not self.device_queues.global_qubits_lists:
            self.set_gates()
        self.global_qubits = self.device_queues.global_qubits_lists[0]
        self._cast_initial_state(initial_state)

        ispecial = 0
        for iall, global_qubits in enumerate(self.device_queues.global_qubits_lists):
            if global_qubits:  # standard gate
                if iall > 0:
                    self._swap(global_qubits)
                self._joblib_execute(iall - ispecial)
            else: # special gate
                self._special_gate_execute(ispecial)
                ispecial += 1

        state = self.final_state
        if self.measurement_gate is None or nshots is None:
            return state

        with tf.device(self.memory_device):
            samples = self.measurement_gate(state, nshots, samples_only=True,
                                            is_density_matrix=self.using_density_matrix)
            self.measurement_gate_result = measurements.GateResult(
                self.measurement_gate.qubits, state, decimal_samples=samples)
            result = measurements.CircuitResult(
                self.measurement_tuples, self.measurement_gate_result)
        return result

    @property
    def final_state(self) -> tf.Tensor:
        """Final state as a Tensorflow tensor of shape ``(2 ** nqubits,)``.

        The circuit has to be executed at least once before accessing this
        property, otherwise a ``ValueError`` is raised. If the circuit is
        executed more than once, only the last final state is returned.
        """
        if self.pieces is None:
            raise RuntimeError("Cannot access the state tensor before being set.")
        return self._merge()

    def _default_global_qubits(self) -> List[int]:
        """Returns a list with the last qubits to cast them as global."""
        return list(range(self.nglobal))

    def _default_initial_piece(self) -> tf.Tensor:
        """Returns the 0th piece for the |000...0> state."""
        zeros = tf.zeros(2 ** (self.nqubits - self.nglobal), dtype=DTYPES.get('DTYPECPX'))
        return op.initial_state(zeros)

    def _create_pieces(self):
        """Creates the state pieces as ``tf.Variable``s stored in the ``memory_device``."""
        n = 2 ** (self.nqubits - self.nglobal)
        with tf.device(self.memory_device):
            self.pieces = [tf.Variable(tf.zeros(n, dtype=DTYPES.get('DTYPECPX')))
                           for _ in range(self.ndevices)]

    def _default_initial_state(self) -> tf.Tensor:
        """Assigns the default |000...0> state to the state pieces."""
        self._create_pieces()
        with tf.device(self.memory_device):
            self.pieces[0].assign(self._default_initial_piece())

    def _cast_initial_state(self, initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None) -> tf.Tensor:
        """Checks and casts initial state given by user."""
        if self._global_qubits is None:
            self.global_qubits = self._default_global_qubits()

        if initial_state is None:
            return self._default_initial_state()

        state = super(TensorflowDistributedCircuit, self)._cast_initial_state(initial_state)
        if self.pieces is None:
            self._create_pieces()
        self._split(state)

    def _split(self, state: tf.Tensor):
        """Splits a given state vector and assigns it to the ``tf.Variable`` pieces.

        Args:
            state (tf.Tensor): Full state vector as a tensor of shape ``(2 ** nqubits)``.
        """
        with tf.device(self.memory_device):
            state = tf.reshape(state, self.device_shape)
            pieces = [state[i] for i in range(self.ndevices)]
            new_state = tf.zeros(self.device_shape, dtype=DTYPES.get('DTYPECPX'))
            new_state = op.transpose_state(pieces, new_state, self.nqubits, self.transpose_order)
            for i in range(self.ndevices):
                self.pieces[i].assign(new_state[i])

    def _merge(self) -> tf.Tensor:
        """Merges the current ``tf.Variable`` pieces to a full state vector.

        Returns:
            state (tf.Tensor): Full state vector as a tensor of shape ``(2 ** nqubits)``.
        """
        new_global_qubits = list(range(self.nglobal))
        if self.global_qubits == new_global_qubits:
            with tf.device(self.memory_device):
                state = tf.concat([x[tf.newaxis] for x in self.pieces], axis=0)
        else:
            state = self._swap(new_global_qubits)
        return tf.reshape(state, self.full_shape)

    def _swap(self, new_global_qubits: Sequence[int]):
        """Changes the list of global qubits in order to apply gates.

        This is done by `transposing` the state vector so that global qubits
        hold the first ``nglobal`` indices at all times.
        """
        order = list(self.reverse_transpose_order)
        self.global_qubits = new_global_qubits
        order = [order[v] for v in self.transpose_order]
        with tf.device(self.memory_device):
            state = tf.zeros(self.device_shape, dtype=DTYPES.get('DTYPECPX'))
            state = op.transpose_state(self.pieces, state, self.nqubits, order)
            for i in range(self.ndevices):
                self.pieces[i].assign(state[i])
        return state
