from qibo import K
from qibo.config import raise_error
from qibo.core import measurements
from qibo.abstractions.states import AbstractState


class VectorState(AbstractState):

    @property
    def shape(self):
        return (self.nstates,)

    @property
    def dtype(self):
        return self.tensor.dtype

    @AbstractState.tensor.setter
    def tensor(self, x):
        if not isinstance(x, K.Tensor):
            if isinstance(x, K.qnp.tensor_types):
                x = K.cast(x)
            else:
                raise_error(TypeError, "Initial state type {} is not recognized."
                                        "".format(type(x)))
        if x.dtype != K.dtypes('DTYPECPX'):
            x = K.cast(x)
        shape = tuple(x.shape)
        if self._nqubits is None:
            self.nqubits = int(K.np.log2(shape[0]))
        if shape != self.shape:
            raise_error(ValueError, "Invalid tensor shape {} for state of {} "
                                    "qubits.".format(shape, self.nqubits))
        self._tensor = x

    def __array__(self):
        return K.to_numpy(self.tensor)

    def numpy(self):
        return self.__array__()

    @classmethod
    def zero_state(cls, nqubits):
        state = cls(nqubits)
        is_matrix = isinstance(state, MatrixState)
        state.tensor = K.initial_state(nqubits, is_matrix)
        return state

    @classmethod
    def plus_state(cls, nqubits):
        state = cls(nqubits)
        shape = K.cast((state.nstates,), dtype='DTYPEINT')
        state.tensor = K.ones(shape) / K.cast(K.qnp.sqrt(state.nstates))
        return state

    def to_density_matrix(self):
        matrix = K.outer(self.tensor, K.conj(self.tensor))
        return MatrixState.from_tensor(matrix, nqubits=self.nqubits)

    def check_measured_qubits(func): # pylint: disable=E0213
        """Decorator for checking list of measured qubits for probability calculation."""
        def wrapper(self, qubits=None, measurement_gate=None):
            if qubits is None:
                if measurement_gate is None:
                    raise_error(ValueError, "Either ``qubits`` or ``measurement_gates`` "
                                            "should be given to calculate measurement "
                                            "probabilities.")
                if not measurement_gate.is_prepared:
                    measurement_gate._set_nqubits(self.tensor)
                qubits = measurement_gate.target_qubits
            elif measurement_gate is not None:
                raise_error(ValueError, "Cannot calculate measurement "
                                        "probabilities if both ``qubits`` and "
                                        "``measurement_gate`` are given."
                                        "Please specify only one of them.")
            return func(self, qubits=set(qubits)) # pylint: disable=E1102
        return wrapper

    @check_measured_qubits
    def probabilities(self, qubits=None, measurement_gate=None):
        unmeasured_qubits = tuple(i for i in range(self.nqubits)
                                  if i not in qubits)
        state = K.reshape(K.square(K.abs(self.tensor)), self.nqubits * (2,))
        return K.sum(state, axis=unmeasured_qubits)

    def measure(self, gate, nshots, registers=None):
        self.measurements = gate(self, nshots)
        if registers is not None:
            self.measurements = measurements.MeasurementRegistersResult(
                registers, self.measurements)

    def set_measurements(self, qubits, samples, registers=None):
        self.measurements = measurements.MeasurementResult(qubits)
        self.measurements.decimal = samples
        if registers is not None:
            self.measurements = measurements.MeasurementRegistersResult(
                    registers, self.measurements)

    def measurement_getter(func): # pylint: disable=E0213
        """Decorator for defining the ``samples`` and ``frequencies`` methods."""
        def wrapper(self, binary=True, registers=False):
            name = func.__name__ # pylint: disable=E1101
            if isinstance(self.measurements, measurements.MeasurementResult):
                return getattr(self.measurements, name)(binary)
            elif isinstance(self.measurements, measurements.MeasurementRegistersResult):
                return getattr(self.measurements, name)(binary, registers)
            raise_error(RuntimeError, "Measurements are not available.")
        return wrapper

    @measurement_getter
    def samples(self, binary=True, registers=False): # pragma: no cover
        pass

    @measurement_getter
    def frequencies(self, binary=True, registers=False): # pragma: no cover
        pass

    def apply_bitflips(self, p0, p1=None):
        if self.measurements is None:
            raise_error(RuntimeError, "Measurements are not available.")
        self.measurements = self.measurements.apply_bitflips(p0, p1)
        return self

    def expectation(self, hamiltonian, normalize=False):
        statec = K.conj(self.tensor)
        hstate = hamiltonian @ self.tensor
        ev = K.real(K.sum(statec * hstate))
        if normalize:
            norm = K.sum(K.square(K.abs(self.tensor)))
            ev = ev / norm
        return ev


class MatrixState(VectorState):

    @property
    def shape(self):
        return (self.nstates, self.nstates)

    @classmethod
    def plus_state(cls, nqubits):
        state = VectorState.plus_state(nqubits)
        return state.to_density_matrix()

    def to_density_matrix(self):
        raise_error(RuntimeError, "State is already a density matrix.")

    @VectorState.check_measured_qubits
    def probabilities(self, qubits=None, measurement_gate=None):
        order = (tuple(sorted(qubits)) +
                 tuple(i for i in range(self.nqubits) if i not in qubits))
        order = order + tuple(i + self.nqubits for i in order)
        shape = 2 * (2 ** len(qubits), 2 ** (self.nqubits - len(qubits)))

        state = K.reshape(self.tensor, 2 * self.nqubits * (2,))
        state = K.reshape(K.transpose(state, order), shape)
        state = K.einsum("abab->a", state)

        return K.reshape(K.cast(state, dtype='DTYPE'), len(qubits) * (2,))

    def expectation(self, hamiltonian, normalize=False):
        ev = K.real(K.trace(hamiltonian @ self.tensor))
        if normalize:
            norm = K.real(K.trace(self.tensor))
            ev = ev / norm
        return ev


class DistributedState(VectorState):
    """Data structure that holds the pieces of a state vector.

    This is created automatically by
    :class:`qibo.core.distcircuit.DistributedCircuit`
    which uses state pieces instead of the full state vector tensor to allow
    distribution to multiple devices.
    Using the ``DistributedState`` instead of the full state vector as a tensor
    avoids creating two copies of the state in the CPU memory and allows
    simulation of one more qubit.

    The full state vector can be accessed using the ``state.vector`` or
    ``state.numpy()`` methods of the ``DistributedState``.
    The ``DistributedState`` supports indexing as a standard array.
    """

    def __init__(self, circuit):
        from qibo.core.distcircuit import DistributedCircuit
        super().__init__(circuit.nqubits)
        self.circuit_cls = DistributedCircuit
        if not isinstance(circuit, self.circuit_cls):
            raise_error(TypeError, "Circuit of unsupported type {} was given to "
                                   "distributed state.".format(type(circuit)))
        self.circuit = circuit
        # List of length ``ndevices`` holding ``tf.Variable``s with the state pieces
        self.pieces = None

        # Dictionaries containing tensors that are useful for reshaping
        # the state when splitting or merging the pieces.
        n = self.nstates // 2 ** self.nglobal
        self.shapes = {
            "full": K.cpu_cast((self.nstates,), dtype='DTYPEINT'),
            "device": K.cpu_cast((self.ndevices, n), dtype='DTYPEINT'),
            "tensor": self.nqubits * (2,)
            }

        self.bintodec = {
            "global": 2 ** K.np.arange(self.nglobal - 1, -1, -1),
            "local": 2 ** K.np.arange(self.nlocal - 1, -1, -1)
            }

    @property
    def qubits(self):
        # ``DistributedQubits`` object created by the ``DistributedQueues`` of the circuit.
        return self.circuit.queues.qubits

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
    def dtype(self):
        return self.pieces[0].dtype

    @property
    def tensor(self):
        """Returns the full state vector as a tensor of shape ``(2 ** nqubits,)``.

        This is done by merging the state pieces to a single tensor.
        Using this method will double memory usage.
        """
        if self.qubits.list == list(range(self.nglobal)):
            with K.on_cpu():
                tensor = K.concatenate([x[K.newaxis] for x in self.pieces], axis=0)
                tensor = K.reshape(tensor, self.shapes["full"])
        elif self.qubits.list == list(range(self.nlocal, self.nqubits)):
            with K.on_cpu():
                tensor = K.concatenate([x[:, K.newaxis] for x in self.pieces], axis=1)
                tensor = K.reshape(tensor, self.shapes["full"])
        else: # fall back to the transpose op
            with K.on_cpu():
                tensor = K.zeros(self.shapes["full"])
                tensor = K.transpose_state(self.pieces, tensor, self.nqubits,
                                           self.qubits.reverse_transpose_order)
        return tensor

    @tensor.setter
    def tensor(self, x):
        raise_error(NotImplementedError, "Tensor setter is not supported by "
                                         "distributed states for memory "
                                         "efficiency.")

    def create_pieces(self):
        n = 2 ** self.nlocal
        with K.on_cpu():
            self.pieces = [K.cpu_tensor(K.zeros(n)) for _ in range(self.ndevices)]

    def assign_pieces(self, tensor):
        """Splits a full state vector and assigns it to the ``tf.Variable`` pieces.

        Args:
            tensor (array): Full state vector as a tensor of shape ``(2 ** nqubits,)``.
        """
        if self.pieces is None:
            self.create_pieces()

        with K.on_cpu():
            tensor = K.reshape(tensor, self.shapes["device"])
            pieces = [tensor[i] for i in range(self.ndevices)]
            new_tensor = K.zeros(self.shapes["device"])
            new_tensor = K.transpose_state(pieces, new_tensor, self.nqubits,
                                           self.qubits.transpose_order)
            for i in range(self.ndevices):
                K.cpu_assign(self, i, new_tensor[i])

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
    def zero_state(cls, circuit):
        state = cls(circuit)
        state.create_pieces()
        with K.on_cpu():
            piece = K.initial_state(nqubits=state.nlocal)
            state.pieces[0] = K.cpu_tensor(piece, dtype=state.dtype)
        return state

    @classmethod
    def plus_state(cls, circuit):
        state = cls(circuit)
        with K.on_cpu():
            n = K.cast(2 ** state.nlocal, dtype=K.dtypes('DTYPEINT'))
            norm = K.cast(2 ** float(state.nqubits / 2.0))
            state.pieces = [K.cpu_tensor(K.ones(n) / norm)
                            for _ in range(state.ndevices)]
        return state

    def copy(self):
        new = self.__class__(self.circuit)
        new.pieces = self.pieces
        new.measurements = self.measurements
        return new

    @VectorState.check_measured_qubits
    def probabilities(self, qubits=None, measurement_gate=None):
        unmeasured_qubits = tuple(i for i in range(self.nqubits)
                                  if i not in qubits)
        with K.on_cpu():
            state = K.reshape(K.square(K.abs(self.tensor)), self.nqubits * (2,))
            return K.sum(state, axis=unmeasured_qubits)
