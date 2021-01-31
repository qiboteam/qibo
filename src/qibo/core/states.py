from qibo import K
from qibo.config import raise_error, get_threads
from qibo.core import measurements
from qibo.abstractions.states import AbstractState


class VectorState(AbstractState):

    @property
    def shape(self):
        return (self.nstates,)

    @AbstractState.tensor.setter
    def tensor(self, x):
        if not isinstance(x, K.tensor_types):
            raise_error(TypeError, "Initial state type {} is not recognized."
                                    "".format(type(x)))
        try:
            shape = tuple(x.shape)
            if self._nqubits is None:
                self.nqubits = int(K.np.log2(shape[0]))
        except ValueError: # happens when using TensorFlow compiled mode
            shape = None
        if shape is not None and shape != self.shape:
            raise_error(ValueError, "Invalid tensor shape {} for state of {} "
                                    "qubits.".format(shape, self.nqubits))
        self._tensor = K.cast(x)

    def __array__(self):
        return K.qnp.cast(self.tensor)

    def numpy(self):
        return self.__array__()

    @classmethod
    def zstate(cls, nqubits):
        state = cls(nqubits)
        is_matrix = isinstance(state, MatrixState)
        state.tensor = K.initial_state(nqubits, is_matrix)
        return state

    @classmethod
    def xstate(cls, nqubits):
        state = cls(nqubits)
        shape = K.cast(state.nstates, dtype='DTYPEINT')
        state.tensor = K.ones(shape) / K.cast(K.qnp.sqrt(state.nstates))
        return state

    @classmethod
    def random(cls, nqubits):
        raise_error(NotImplementedError)

    def to_density_matrix(self):
        matrix = K.outer(self.tensor, K.conj(self.tensor))
        return MatrixState.from_tensor(matrix, nqubits=self.nqubits)

    def traceout(self, qubits=None, measurement_gate=None):
        if qubits is None and measurement_gate is None:
            raise_error(ValueError)

        if qubits is not None:
            if measurement_gate is not None:
                raise_error(ValueError)
            unmeasured_qubits = [i for i in range(self.nqubits)
                                 if i not in qubits]
            if isinstance(self, MatrixState):
                from qibo.abstractions.callbacks import PartialTrace
                qubits = set(unmeasured_qubits)
                return PartialTrace.einsum_string(qubits, self.nqubits,
                                                  measuring=True)
            return unmeasured_qubits

        if not measurement_gate.is_prepared:
            measurement_gate.set_nqubits(self.tensor)
        if isinstance(self, MatrixState):
            return measurement_gate.traceout
        return measurement_gate.unmeasured_qubits

    def probabilities(self, qubits=None, measurement_gate=None):
        unmeasured_qubits = self.traceout(qubits, measurement_gate)
        shape = self.nqubits * (2,)
        state = K.reshape(K.square(K.abs(self.tensor)), shape)
        return K.sum(state, axis=unmeasured_qubits)

    def measure(self, gate, nshots, registers=None):
        self.measurements = gate(self, nshots)
        if registers is not None:
            self.measurements = measurements.CircuitResult(
                registers, self.measurements)

    def set_measurements(self, qubits, samples, registers=None):
        self.measurements = measurements.GateResult(qubits, decimal_samples=samples)
        if registers is not None:
            self.measurements = measurements.CircuitResult(
                    registers, self.measurements)

    def _get_measurements(self, mode="samples", binary=True, registers=False):
        if isinstance(self.measurements, measurements.GateResult):
            return getattr(self.measurements, mode)(binary)
        elif isinstance(self.measurements, measurements.CircuitResult):
            return getattr(self.measurements, mode)(binary, registers)
        raise_error(RuntimeError, "Measurements are not available.")

    def samples(self, binary=True, registers=False):
        return self._get_measurements("samples", binary, registers)

    def frequencies(self, binary=True, registers=False):
        return self._get_measurements("frequencies", binary, registers)

    def apply_bitflips(self, p0, p1=None):
        if self.measurements is None:
            raise_error(RuntimeError, "Measurements are not available.")
        self.measurements = self.measurements.apply_bitflips(p0, p1)
        return self


class MatrixState(VectorState):

    @property
    def shape(self):
        return (self.nstates, self.nstates)

    @classmethod
    def xstate(cls, nqubits):
        state = VectorState.xstate(nqubits)
        return state.to_density_matrix()

    @classmethod
    def random(cls, nqubits):
        raise_error(NotImplementedError)

    def to_density_matrix(self):
        raise_error(RuntimeError, "State is already a density matrix.")

    def probabilities(self, qubits=None, measurement_gate=None):
        traceout = self.traceout(qubits, measurement_gate)
        shape = 2 * self.nqubits * (2,)
        state = K.einsum(traceout, K.reshape(self.tensor, shape))
        return K.cast(state, dtype='DTYPE')


class DistributedState(VectorState):
    """Data structure that holds the pieces of a state vector.

    This is created automatically by
    :class:`qibo.tensorflow.distcircuit.DistributedCircuit`
    which uses state pieces instead of the full state vector tensor to allow
    distribution to multiple devices.
    Using the ``DistributedState`` instead of the full state vector as a tensor
    avoids creating two copies of the state in the CPU memory and allows
    simulation of one more qubit.

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

    def __init__(self, circuit):
        from qibo.tensorflow.distcircuit import DistributedCircuit
        super().__init__(circuit.nqubits)
        self.circuit_cls = DistributedCircuit
        self._circuit = None
        self.device = None
        self.qubits = None
        self.circuit = circuit

        # Create pieces
        n = 2 ** (self.nqubits - self.nglobal)
        with K.device(self.device):
            self.pieces = [K.optimization.Variable(K.zeros(n))
                           for _ in range(self.ndevices)]

        self.shapes = {
            "full": K.cast((2 ** self.nqubits,), dtype='DTYPEINT'),
            "device": K.cast((len(self.pieces), n), dtype='DTYPEINT'),
            "tensor": self.nqubits * (2,)
            }

        self.bintodec = {
            "global": 2 ** K.np.arange(self.nglobal - 1, -1, -1),
            "local": 2 ** K.np.arange(self.nlocal - 1, -1, -1)
            }

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, c):
        if not isinstance(c, self.circuit_cls):
            raise_error(TypeError, "Circuit of unsupported type {} was given to "
                                   "distributed state.")
        self._circuit = c
        self.device = c.memory_device
        self.qubits = c.queues.qubits

    @property
    def nglobal(self):
        return self.circuit.nglobal

    @property
    def nlocal(self):
        return self.circuit.nlocal

    @property
    def ndevices(self):
        return self.circuit.ndevices

    @property
    def tensor(self):
        """Returns the full state vector as a tensor of shape ``(2 ** nqubits,)``.

        This is done by merging the state pieces to a single tensor.
        Using this method will double memory usage.
        """
        if self.qubits.list == list(range(self.nglobal)):
            with K.device(self.device):
                state = K.concatenate([x[K.newaxis] for x in self.pieces], axis=0)
                state = K.reshape(state, self.shapes["full"])
        elif self.qubits.list == list(range(self.nlocal, self.nqubits)):
            with K.device(self.device):
                state = K.concatenate([x[:, K.newaxis] for x in self.pieces], axis=1)
                state = K.reshape(state, self.shapes["full"])
        else: # fall back to the transpose op
            with K.device(self.device):
                state = K.zeros(self.shapes["full"])
                state = K.op.transpose_state(self.pieces, state, self.nqubits,
                                             self.qubits.reverse_transpose_order,
                                             get_threads())
        return state

    @tensor.setter
    def tensor(self, x):
        raise_error(NotImplementedError, "Tensor setter is not supported by "
                                         "distributed states for memory "
                                         "efficiency.")

    @property
    def dtype(self):
        return self.pieces[0].dtype

    def assign_pieces(self, full_state):
        """Splits a full state vector and assigns it to the ``tf.Variable`` pieces.

        Args:
            full_state (array): Full state vector as a tensor of shape
                ``(2 ** nqubits)``.
        """
        with K.device(self.device):
            full_state = K.reshape(full_state, self.shapes["device"])
            pieces = [full_state[i] for i in range(self.ndevices)]
            new_state = K.zeros(self.shapes["device"])
            new_state = K.op.transpose_state(pieces, new_state, self.nqubits,
                                             self.qubits.transpose_order,
                                             get_threads())
            for i in range(self.ndevices):
                self.pieces[i].assign(new_state[i])

    def __getitem__(self, key):
      """Implements indexing of the distributed state without the full vector."""
      if isinstance(key, slice):
          return [self[i] for i in range(*key.indices(len(self)))]

      elif isinstance(key, list):
          return [self[i] for i in key]

      elif isinstance(key, int):
          binary_index = bin(key)[2:].zfill(self.nqubits)
          binary_index = K.np.array([int(x) for x in binary_index],
                                    dtype=K.np.int64)

          global_ids = binary_index[self.qubits.list]
          global_ids = global_ids.dot(self.bintodec["global"])
          local_ids = binary_index[self.qubits.local]
          local_ids = local_ids.dot(self.bintodec["local"])
          return self.pieces[global_ids][local_ids]

      else:
          raise_error(TypeError, "Unknown index type {}.".format(type(key)))

    @classmethod
    def from_tensor(cls, full_state, circuit):
        state = cls(circuit)
        state.assign_pieces(full_state)
        return state

    @classmethod
    def zstate(cls, circuit):
      state = cls(circuit)
      with K.device(state.device):
          piece = K.initial_state(nqubits=state.nlocal)
          state.pieces[0] = K.optimization.Variable(piece, dtype=piece.dtype)
      return state

    @classmethod
    def xstate(cls, circuit):
      state = cls(circuit)
      with K.device(state.device):
          norm = K.cast(2 ** float(state.nqubits / 2.0), dtype=state.dtype)
          state.pieces = [K.optimization.Variable(K.ones_like(p) / norm)
                          for p in state.pieces]
      return state

    def copy(self):
        raise_error(NotImplementedError)
