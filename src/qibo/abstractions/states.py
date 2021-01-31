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

    @property
    @abstractmethod
    def shape(self):
        """Shape of the state's tensor representation."""
        raise_error(NotImplementedError)

    @nqubits.setter
    def nqubits(self, n):
        self._nqubits = n
        self.nstates = 2 ** n

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

    @property
    def dtype(self):
        """Type of state's tensor representation."""
        return self.tensor.dtype

    @abstractmethod
    def __array__(self):
        """State's tensor representation as an array."""
        raise_error(NotImplementedError)

    @abstractmethod
    def numpy(self):
        """State's tensor representation as a numpy array."""
        raise_error(NotImplementedError)

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
    def zstate(cls, nqubits):
        """Constructs the |00...0> state.

        Args:
            nqubits (int): Number of qubits in the state.
        """
        raise_error(NotImplementedError)

    @classmethod
    @abstractmethod
    def xstate(cls, nqubits):
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
    def to_density_matrix(self):
        """Transforms a pure quantum state to its density matrix form."""
        raise_error(NotImplementedError)

    @abstractmethod
    def probabilities(self, qubits=None, measurement_gate=None):
        """"""
        raise_error(NotImplementedError)

    @abstractmethod
    def measure(self, gate, nshots, registers=None):
        raise_error(NotImplementedError)

    @abstractmethod
    def samples(self, binary=True, registers=False):
        raise_error(NotImplementedError)

    @abstractmethod
    def frequencies(self, binary=True, registers=False):
        raise_error(NotImplementedError)

    @abstractmethod
    def apply_bitflips(self, p0, p1=None):
        raise_error(NotImplementedError)
