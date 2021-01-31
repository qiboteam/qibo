import math
from abc import ABC, abstractmethod
from qibo.config import raise_error


class AbstractState(ABC):

    def __init__(self, nqubits=None):
        self._nqubits = None
        self._tensor = None
        self.nstates = None
        if nqubits is not None:
            self.nqubits = nqubits

    @property
    def nqubits(self):
        if self._nqubits is None:
            raise_error(AttributeError, "State number of qubits not available.")
        return self._nqubits

    def __len__(self):
        if self._nqubits is None:
            raise_error(AttributeError, "State number of qubits not available.")
        return self.nstates

    @property
    @abstractmethod
    def shape(self):
        raise_error(NotImplementedError)

    @nqubits.setter
    def nqubits(self, n):
        self._nqubits = n
        self.nstates = 2 ** n

    @property
    def tensor(self):
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
        return self.tensor.dtype

    @abstractmethod
    def __array__(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def numpy(self):
        raise_error(NotImplementedError)

    def __getitem__(self, i):
        if isinstance(i, int) and i >= self.nstates:
            raise_error(IndexError, "State index {} out of range.".format(i))
        return self.tensor[i]

    @classmethod
    def from_tensor(cls, x, nqubits=None):
        obj = cls(nqubits)
        obj.tensor = x
        return obj

    @classmethod
    @abstractmethod
    def zstate(cls, nqubits):
        raise_error(NotImplementedError)

    @classmethod
    @abstractmethod
    def xstate(cls, nqubits):
        raise_error(NotImplementedError)

    @classmethod
    @abstractmethod
    def random(cls, nqubits):
        raise_error(NotImplementedError)

    @abstractmethod
    def to_density_matrix(self):
        raise_error(NotImplementedError)

    @abstractmethod
    def probabilities(self):
        raise_error(NotImplementedError)
