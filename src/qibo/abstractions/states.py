import math
from abc import ABC, abstractmethod
from qibo.config import raise_error


class AbstractState(ABC):
    """Abstract class for quantum states returned by model execution.

    Args:
        nqubits (int): Optional number of qubits in the state.
            If ``None`` then the number is calculated automatically from the
            tensor representation of the state.
    """

    def __init__(self, nqubits=None):
        self._nqubits = None
        self._tensor = None
        self.nstates = None
        self.measurements = None
        if nqubits is not None:
            self.nqubits = nqubits

    @property
    def nqubits(self):
        """Number of qubits in the state."""
        if self._nqubits is None:
            raise_error(AttributeError, "State number of qubits not available.")
        return self._nqubits

    def __len__(self):
        """Number of components in the state's tensor representation."""
        if self._nqubits is None:
            raise_error(AttributeError, "State number of qubits not available.")
        return self.nstates

    @nqubits.setter
    def nqubits(self, n):
        self._nqubits = n
        self.nstates = 2 ** n

    @property
    @abstractmethod
    def shape(self): # pragma: no cover
        """Shape of the state's tensor representation."""
        raise_error(NotImplementedError)

    @property
    @abstractmethod
    def dtype(self): # pragma: no cover
        """Type of state's tensor representation."""
        raise_error(NotImplementedError)

    @property
    def tensor(self):
        """Tensor representation of the state in the computational basis."""
        if self._tensor is None:
            raise_error(AttributeError, "State tensor not available.")
        return self._tensor

    @tensor.setter
    def tensor(self, x):
        nqubits = int(math.log2(len(x)))
        if self._nqubits is None:
            self.nqubits = nqubits
        elif self._nqubits != nqubits:
            raise_error(ValueError, "Cannot assign tensor of length {} to "
                                    "state with {} qubits."
                                    "".format(len(x), self.nqubits))
        self._tensor = x

    @abstractmethod
    def __array__(self): # pragma: no cover
        """State's tensor representation as an array."""
        raise_error(NotImplementedError)

    @abstractmethod
    def numpy(self): # pragma: no cover
        """State's tensor representation as a numpy array."""
        raise_error(NotImplementedError)

    def state(self, numpy=False):
        """State's tensor representation as an backend tensor.

        Args:
            numpy (bool): If ``True`` the returned tensor will be a numpy array,
                otherwise it will follow the backend tensor type.
                Default is ``False``.
        """
        if numpy:
            return self.numpy()
        return self.tensor

    def __getitem__(self, i):
        if isinstance(i, int) and i >= self.nstates:
            raise_error(IndexError, "State index {} out of range.".format(i))
        return self.tensor[i]

    @classmethod
    def from_tensor(cls, x, nqubits=None):
        """Constructs state from a tensor.

        Args:
            x: Tensor representation of the state in the computational basis.
            nqubits (int): Optional number of qubits in the state.
                If ``None`` it is calculated automatically from the tensor
                representation shape.
        """
        obj = cls(nqubits)
        obj.tensor = x
        return obj

    @classmethod
    @abstractmethod
    def zero_state(cls, nqubits): # pragma: no cover
        """Constructs the |00...0> state.

        Args:
            nqubits (int): Number of qubits in the state.
        """
        raise_error(NotImplementedError)

    @classmethod
    @abstractmethod
    def plus_state(cls, nqubits): # pragma: no cover
        """Constructs the |++...+> state.

        Args:
            nqubits (int): Number of qubits in the state.
        """
        raise_error(NotImplementedError)

    def copy(self):
        """Creates a copy of the state.

        Note that this does not create a deep copy. The new state references
        to the same tensor representation for memory efficiency.
        """
        new = self.__class__(self._nqubits)
        if self._tensor is not None:
            new.tensor = self.tensor
        new.measurements = self.measurements
        return new

    @abstractmethod
    def to_density_matrix(self): # pragma: no cover
        """Transforms a pure quantum state to its density matrix form.

        Returns:
            A :class:`qibo.abstractions.states.AbstractState` object that
            contains the state in density matrix form.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def probabilities(self, qubits=None, measurement_gate=None): # pragma: no cover
        """Calculates measurement probabilities by tracing out qubits.

        Exactly one of the following arguments should be given.

        Args:
            qubits (list, set): Set of qubits that are measured.
            measurement_gate (:class:`qibo.abstractions.gates.M`): Measurement
                gate that contains the measured qubit details.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def measure(self, gate, nshots, registers=None): # pragma: no cover
        """Measures the state using a measurement gate.

        Args:
            gate (:class:`qibo.abstractions.gates.M`): Measurement gate to use
                for measuring the state.
            nshots (int): Number of measurement shots.
            registers (dict): Dictionary that maps register names to the
                corresponding tuples of qubit ids.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def set_measurements(self, qubits, samples, registers=None): # pragma: no cover
        """Sets the state's measurements using decimal samples.

        Args:
            qubits (tuple): Measured qubit ids.
            samples (Tensor): Tensor with decimal samples of the measurement
                results.
            registers (dict): Dictionary that maps register names to the
                corresponding tuples of qubit ids.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def samples(self, binary=True, registers=False): # pragma: no cover
        """Returns raw measurement samples.

        Args:
            binary (bool): Return samples in binary or decimal form.
            registers (bool): Group samples according to registers.
                Can be used only if ``registers`` were given when calling
                :meth:`qibo.abstractions.states.AbstractState.measure`.

        Returns:
            If `binary` is `True`
                samples are returned in binary form as a tensor
                of shape `(nshots, n_measured_qubits)`.
            If `binary` is `False`
                samples are returned in decimal form as a tensor
                of shape `(nshots,)`.
            If `registers` is `True`
                samples are returned in a `dict` where the keys are the register
                names and the values are the samples tensors for each register.
            If `registers` is `False`
                a single tensor is returned which contains samples from all the
                measured qubits, independently of their registers.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def frequencies(self, binary=True, registers=False): # pragma: no cover
        """Returns the frequencies of measured samples.

        Args:
            binary (bool): Return frequency keys in binary or decimal form.
            registers (bool): Group frequencies according to registers.
                Can be used only if ``registers`` were given when calling
                :meth:`qibo.abstractions.states.AbstractState.measure`.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If `binary` is `True`
                the keys of the `Counter` are in binary form, as strings of
                0s and 1s.
            If `binary` is `False`
                the keys of the `Counter` are integers.
            If `registers` is `True`
                a `dict` of `Counter` s is returned where keys are the name of
                each register.
            If `registers` is `False`
                a single `Counter` is returned which contains samples from all
                the measured qubits, independently of their registers.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_bitflips(self, p0, p1=None): # pragma: no cover
        """Applies bitflip noise to the measured samples.

        Args:
            p0: Bitflip probability map. Can be:
                A dictionary that maps each measured qubit to the probability
                that it is flipped, a list or tuple that has the same length
                as the tuple of measured qubits or a single float number.
                If a single float is given the same probability will be used
                for all qubits.
            p1: Probability of asymmetric bitflip. If ``p1`` is given, ``p0``
                will be used as the probability for 0->1 and ``p1`` as the
                probability for 1->0. If ``p1`` is ``None`` the same probability
                ``p0`` will be used for both bitflips.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def expectation(self, hamiltonian, normalize=False): # pragma: no cover
        """Calculates Hamiltonian expectation value with respect to the state.

        Args:
            hamiltonian (`qibo.abstractions.hamiltonians.Hamiltonian`):
                Hamiltonian object to calculate the expectation value of.
            normalize (bool): Normalize the result by dividing with the norm of
                the state. Default is ``False``.
        """
        raise_error(NotImplementedError)
