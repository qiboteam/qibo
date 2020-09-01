import copy
import numpy as np
import tensorflow as tf
from qibo.base import gates
from qibo.tensorflow import custom_operators as op
from qibo.config import DTYPES, raise_error
from typing import Dict, List, Optional, Sequence, Tuple


class DistributedQubits:
    """Data structure that holds lists related to global qubit IDs.

    Holds the following data:
    * ``list``: Sorted list with the ids of global qubits.
    * ``set``: Same as ``list`` but in a set to allow O(1) search.
    * ``local``: Sorted list with the ids of local qubits.
    * ``reduced_global``: Map from global qubit ids to their reduced value.
        The reduced value is the effective id in a hypothetical circuit
        that does not contain the local qubits.
    * ``reduced_local``: Map from local qubit ids to their reduced value.
    * ``transpose_order``: Order of indices used to split a full state vector
        to state pieces.
    * ``reverse_tranpose_order``: Order of indices used to merge state pieces
        to a full state vector.
    """

    def __init__(self, qubits: Sequence[int], nqubits: int):
        self.set = set(qubits)
        self.list = sorted(qubits)
        self.local = [q for q in range(nqubits) if q not in self.set]
        self.reduced_global = {q: self.list.index(q) for q in self.list}
        self.reduced_local = {q: q - self.reduction_number(q)
                              for q in self.local}

        self.transpose_order = self.list + self.local
        self.reverse_transpose_order = nqubits * [0]
        for i, v in enumerate(self.transpose_order):
            self.reverse_transpose_order[v] = i

    def reduction_number(self, q: int) -> int:
        """Calculates the effective id in a circuit without the global qubits."""
        for i, gq in enumerate(self.list):
            if gq > q:
                return i
        return i + 1


class DistributedBase:
    """Base class for ``DistributedQueues`` and ``DistributedState``.

    Holds a reference to the parent ``DistributedCircuit`` and reads from it
    the following properties:
    * ``nqubits``: Total number of qubits in the circuit.
    * ``ndevices``: Number of logical accelerator devices.
    * ``nglobal``: Number of global qubits (= log2(ndevices)).
    * ``nlocal``: Number of local qubits (= nqubits - nglobal).
    """

    def __init__(self, circuit):
        self.circuit = circuit

    @property
    def nqubits(self):
        return self.circuit.nqubits

    @property
    def nglobal(self):
        return self.circuit.nglobal

    @property
    def nlocal(self):
        return self.circuit.nlocal

    @property
    def ndevices(self):
        return self.circuit.ndevices


class DistributedQueues(DistributedBase):
    """Data structure that holds gate queues for each accelerator device.

    For a distributed simulation we have to swap global qubits multiple times.
    For each global qubit configuration a several gates can be applied to the
    state forming a gate group. Once all gates in the group are applied the
    global qubits are swapped and we proceed to the next gate group.

    Holds the following data (in addition to ``DistributedBase``):
    * ``gate_module``: Gate module used by the circuit. This is used to create
        new SWAP gates when needed.
    * ``device_to_ids``: Dictionary that maps device (str) to list of piece indices.
        When a device is used multiple times then it is responsible for updating
        multiple state pieces. The list of indices specifies which pieces the device
        will update.
    * ``ids_to_device``: Inverse dictionary of ``device_to_ids``.
    * ``queues``: Nested list of shape ``(ngroups, ndevices, group size)``.
        For example ``queues[2][1]`` gives the gate queue of the second gate
        group to be run in the first device.
        If ``gate[i]`` is an empty list it means that this the i-th group
        consists of a special gate to be run on ``memory_device``.
    * ``special_queue``: List with special gates than run on the full state vector
        on ``memory_device``. Special gates have no target qubits and can be
        ``CallbackGate``, ``Flatten`` or SWAPs between local and global qubits.
    """

    def __init__(self, circuit, gate_module):
        super(DistributedQueues, self).__init__(circuit)
        self.gate_module = gate_module
        self.queues = []
        self.special_queue = []
        self.qubits = None

        # List that holds the global-local SWAP pairs so that we can reset them
        # in the end
        self.swaps_list = []

        self.device_to_ids = {d: v for d, v in self._ids(circuit.calc_devices)}
        self.ids_to_device = self.ndevices * [None]
        for device, ids in self.device_to_ids.items():
            for i in ids:
                self.ids_to_device[i] = device

    def set(self, queue: List[gates.Gate]):
        """Prepares gates for device-specific gate execution.

        Each gate has to be recreated in the device that will be executed to
        allow parallel execution. This method creates the gate groups that
        contain these device gates.
        A gate group is identified by looping through the circuit's gate queue
        and adding gates in the group until the number of global becomes ``nglobal``.
        Once this happens no more gates can be added in the group. In order to
        apply new gates some global qubits have to be swapped to global and a
        new gate group will be defined for the new global qubit configuration.

        This method also creates the ``DistributedQubits`` object holding the
        global qubits list.
        """
        if not queue:
            raise_error(RuntimeError, "No gates available to set for distributed run.")

        counter = self.count(queue, self.nqubits)
        if self.qubits is None:
            self.qubits = DistributedQubits(counter.argsort()[:self.nglobal],
                                            self.nqubits)
        transformed_queue = self.transform(queue, counter)
        self.create(transformed_queue)

    def _ids(self, calc_devices: Dict[str, int]) -> Tuple[str, List[int]]:
        """Generator of device piece indices."""
        start = 0
        for device, n in calc_devices.items():
            stop = start + n
            yield device, list(range(start, stop))
            start = stop

    def _create_device_gate(self, gate: gates.Gate) -> gates.Gate:
        """Creates a copy of a gate for specific device application.

        Target and control qubits are modified according to the local qubits of
        the circuit when this gate will be applied.

        Args:
            gate: The :class:`qibo.base.gates.Gate` object of the gate to copy.

        Returns:
            A :class:`qibo.base.gates.Gate` object with the proper target and
            control qubit indices for device-specific application.
        """
        devgate = copy.copy(gate)
        # Recompute the target/control indices considering only local qubits.
        new_target_qubits = tuple(q - self.qubits.reduction_number(q)
                                  for q in devgate.target_qubits)
        new_control_qubits = tuple(q - self.qubits.reduction_number(q)
                                   for q in devgate.control_qubits
                                   if q not in self.qubits.set)
        devgate.set_targets_and_controls(new_target_qubits, new_control_qubits)
        devgate.original_gate = gate
        devgate.device_gates = set()
        return devgate

    @staticmethod
    def count(queue: List[gates.Gate], nqubits: int) -> np.ndarray:
        """Counts how many gates target each qubit.

        Args:
            queue: List of gates.
            nqubits: Number of total qubits in the circuit.

        Returns:
            Array of integers with shape (nqubits,) with the number of gates
            for each qubit id.
        """
        counter = np.zeros(nqubits, dtype=np.int32)
        for gate in queue:
            for qubit in gate.target_qubits:
                counter[qubit] += 1
        return counter

    def _transform(self, queue: List[gates.Gate],
                   remaining_queue: List[gates.Gate],
                   counter: np.ndarray) -> List[gates.Gate]:
        """Helper recursive method for ``transform``."""
        new_remaining_queue = []
        for gate in remaining_queue:
            if gate.is_special_gate:
                gate.swap_reset = list(self.swaps_list)

            global_targets = set(gate.target_qubits) & self.qubits.set
            accept = isinstance(gate, gates.SWAP) and len(global_targets) == 1
            accept = accept or not global_targets
            for skipped_gate in new_remaining_queue:
                accept = accept and skipped_gate.commutes(gate)
                if not accept:
                    break
            if accept:
                queue.append(gate)
                for q in gate.target_qubits:
                    counter[q] -= 1
            else:
                new_remaining_queue.append(gate)

        if not new_remaining_queue:
            return queue

        # Find which qubits to swap
        gate = new_remaining_queue[0]
        target_set = set(gate.target_qubits)
        global_targets = target_set & self.qubits.set
        if isinstance(gate, gates.SWAP): # pragma: no cover
            # special case of swap on two global qubits
            assert len(global_targets) == 2
            global_targets.remove(target_set.pop())

        available_swaps = (q for q in counter.argsort()
                           if q not in self.qubits.set | target_set)
        qubit_map = {}
        for q in global_targets:
            qs = next(available_swaps)
            # Update qubit map that holds the swaps
            qubit_map[q] = qs
            qubit_map[qs] = q
            # Keep SWAPs in memory to reset them in the end
            self.swaps_list.append((min(q, qs), max(q, qs)))
            # Add ``SWAP`` gate in ``queue``.
            queue.append(self.gate_module.SWAP(q, qs))
            #  Modify ``counter`` to take into account the swaps
            counter[q], counter[qs] = counter[qs], counter[q]

        # Modify gates to take into account the swaps
        for gate in new_remaining_queue:
            new_target_qubits = tuple(qubit_map[q] if q in qubit_map else q
                                       for q in gate.target_qubits)
            new_control_qubits = tuple(qubit_map[q] if q in qubit_map else q
                                        for q in gate.control_qubits)
            gate.set_targets_and_controls(new_target_qubits, new_control_qubits)

        return self._transform(queue, new_remaining_queue, counter)

    def transform(self, queue: List[gates.Gate],
                  counter: Optional[np.ndarray] = None) -> List[gates.Gate]:
        """Transforms gate queue to be compatible with distributed simulation.

        Adds SWAP gates between global and local qubits so that no gates are
        applied to global qubits.

        Args:
            queue (list): Original gate queue.
            counter (np.ndarray): Counter of how many gates target each qubit.
                If ``None`` this is calculated using the ``count`` method.

        Returns:
            List of gates that have the same effect as the original queue but
            are compatible with distributed run (do not have global qubits as
            targets).
        """
        if counter is None:
            counter = self.count(queue, self.nqubits)
        new_queue = self._transform([], queue, counter)
        new_queue.extend((self.gate_module.SWAP(*p)
                          for p in reversed(self.swaps_list)))
        return new_queue

    def create(self, queue: List[gates.Gate]):
        """Creates the queues for each accelerator device.

        Args:
            queue (list): List of gates compatible with distributed run.
            If the original ``queue`` contains gates that target global qubits
            then ``transform` should be used to obtain a compatible queue.
        """
        for gate in queue:
            if not gate.target_qubits: # special gate
                gate.nqubits = self.nqubits
                self.special_queue.append(gate)
                self.queues.append([])

            elif set(gate.target_qubits) & self.qubits.set: # global swap gate
                global_qubits = set(gate.target_qubits) & self.qubits.set
                if not isinstance(gate, gates.SWAP):
                    raise_error(ValueError, "Only SWAP gates are supported for "
                                            "global qubits.")
                if len(global_qubits) > 1:
                    raise_error(ValueError, "SWAPs between global qubits are not allowed.")

                global_qubit = global_qubits.pop()
                local_qubit = gate.target_qubits[0]
                if local_qubit == global_qubit:
                    local_qubit = gate.target_qubits[1]

                self.special_queue.append((global_qubit, local_qubit))
                self.queues.append([])

            else:
                if not self.queues or not self.queues[-1]:
                    self.queues.append([[] for _ in range(self.ndevices)])

                for device, ids in self.device_to_ids.items():
                    devgate = self._create_device_gate(gate)
                    # Gate matrix should be constructed in the calculation
                    # device otherwise device parallelization will break
                    devgate.device = device
                    devgate.nqubits = self.nlocal

                    for i in ids:
                        flag = True
                        # If there are control qubits that are global then
                        # the gate should not be applied by all devices
                        for control in (set(gate.control_qubits) &
                                        self.qubits.set):
                            ic = self.qubits.list.index(control)
                            ic = self.nglobal - ic - 1
                            flag = bool((i // (2 ** ic)) % 2)
                            if not flag:
                                break
                        if flag:
                            self.queues[-1][i].append(devgate)
                            if isinstance(gate, gates.ParametrizedGate):
                                gate.device_gates.add(devgate)


class DistributedState(DistributedBase):
    """Data structure that holds the pieces of a state vector.

    This is created automatically by
    :class:`qibo.tensorflow.distcircuit.TensorflowDistributedCircuit`
    which uses state pieces instead of the full state vector tensor to allow
    distribution to multiple devices.
    Using the ``DistributedState`` instead of the full state vector as a
    ``tf.Tensor`` avoids creating two copies of the state in the CPU memory
    and allows simulation of one more qubit.

    The full state vector can be accessed using the ``state.vector`` or
    ``state.numpy()`` methods of the ``DistributedState``.
    The ``DistributedState`` supports indexing as a standard array.

    Holds the following data:
    * self.pieces: List of length ``ndevices`` holding ``tf.Variable``s with
      the state pieces.
    * self.qubits: The ``DistributedQubits`` object created by the
      ``DistributedQueues`` of the circuit.
    * self.shapes: Dictionary containing tensors that are useful for reshaping
      the state when splitting/merging the pieces.
    """

    def __init__(self, circuit: "DistributedCircuit"):
        super(DistributedState, self).__init__(circuit)
        self.device = circuit.memory_device
        self.qubits = circuit.queues.qubits
        self.dtype = DTYPES.get('DTYPECPX')

        # Create pieces
        n = 2 ** (self.nqubits - self.nglobal)
        with tf.device(self.device):
            self.pieces = [tf.Variable(tf.zeros(n, dtype=self.dtype))
                           for _ in range(self.ndevices)]

        dtype = DTYPES.get('DTYPEINT')
        self.shapes = {
            "full": tf.cast((2 ** self.nqubits,), dtype=dtype),
            "device": tf.cast((len(self.pieces), n), dtype=dtype),
            "tensor": self.nqubits * (2,)
            }

        self.bintodec = {
            "global": 2 ** np.arange(self.nglobal - 1, -1, -1),
            "local": 2 ** np.arange(self.nlocal - 1, -1, -1)
            }

    @classmethod
    def default(cls, circuit: "DistributedCircuit"):
      """Creates the |000...0> state for default initialization."""
      state = cls(circuit)
      with tf.device(state.device):
          op.initial_state(state.pieces[0])
      return state

    @classmethod
    def ones(cls, circuit: "DistributedCircuit"):
      """Creates the |+++...+> state for adiabatic evolution initialization."""
      state = cls(circuit)
      with tf.device(state.device):
          norm = tf.cast(2 ** float(state.nqubits / 2.0), dtype=state.dtype)
          state.pieces = [tf.Variable(tf.ones_like(p) / norm)
                          for p in state.pieces]
      return state

    @classmethod
    def from_vector(cls, full_state: tf.Tensor, circuit: "DistributedCircuit"):
        """Initializes pieces from a given full state vector."""
        state = cls(circuit)
        state.assign_vector(full_state)
        return state

    def assign_vector(self, full_state: tf.Tensor):
        """Splits a full state vector and assigns it to the ``tf.Variable`` pieces.

        Args:
            full_state (tf.Tensor): Full state vector as a tensor of shape
                ``(2 ** nqubits)``.
        """
        with tf.device(self.device):
            full_state = tf.reshape(full_state, self.shapes["device"])
            pieces = [full_state[i] for i in range(self.ndevices)]
            new_state = tf.zeros(self.shapes["device"], dtype=self.dtype)
            new_state = op.transpose_state(pieces, new_state, self.nqubits,
                                           self.qubits.transpose_order)
            for i in range(self.ndevices):
                self.pieces[i].assign(new_state[i])

    @property
    def vector(self) -> tf.Tensor:
        """Returns the full state vector as a ``tf.Tensor`` of shape ``(2 ** nqubits,)``.

        This is done by merging the state pieces to a single tensor.
        Using this method will double memory usage.
        """
        if self.qubits.list == list(range(self.nglobal)):
            with tf.device(self.device):
                state = tf.concat([x[tf.newaxis] for x in self.pieces], axis=0)
                state = tf.reshape(state, self.shapes["full"])
        elif self.qubits.list == list(range(self.nlocal, self.nqubits)):
            with tf.device(self.device):
                state = tf.concat([x[:, tf.newaxis] for x in self.pieces], axis=1)
                state = tf.reshape(state, self.shapes["full"])
        else: # fall back to the transpose op
            with tf.device(self.device):
                state = tf.zeros(self.shapes["full"], dtype=self.dtype)
                state = op.transpose_state(self.pieces, state, self.nqubits,
                                           self.qubits.reverse_transpose_order)
        return state

    def __len__(self) -> int:
        return 2 ** self.nqubits

    def __getitem__(self, key):
      """Implements indexing of the distributed state without the full vector."""
      if isinstance(key, slice):
          return [self[i] for i in range(*key.indices(len(self)))]

      elif isinstance(key, list):
          return [self[i] for i in key]

      elif isinstance(key, int):
          binary_index = bin(key)[2:].zfill(self.nqubits)
          binary_index = np.array([int(x) for x in binary_index])

          global_ids = binary_index[self.qubits.list]
          global_ids = global_ids.dot(self.bintodec["global"])
          local_ids = binary_index[self.qubits.local]
          local_ids = local_ids.dot(self.bintodec["local"])
          return self.pieces[global_ids][local_ids]

      else:
          raise_error(TypeError, "Unknown index type {}.".format(type(key)))

    def __array__(self) -> np.ndarray:
        return self.vector.numpy()

    def numpy(self) -> np.ndarray:
        return self.vector.numpy()
