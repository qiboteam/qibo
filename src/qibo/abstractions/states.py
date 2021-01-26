import math
from abc import ABC, abstractmethod
from qibo.config import raise_error


class AbstractState(ABC):

    def __init__(self, nqubits=None):
        self._nqubits = None
        self._vector = None
        self._matrix = None
        self.nstates = None

    @property
    def nqubits(self):
        if self._nqubits is None:
            raise_error(AttributeError, "State number of qubits not available.")
        return self._nqubits

    def __len__(self):
        if self._nqubits is None:
            raise_error(AttributeError, "State number of qubits not available.")
        return self.nstates

    @nqubits.setter
    def nqubits(self, n):
        self._nqubits = n
        self.nstates = 2 ** n

    @property
    def vector(self):
        if self._vector is None:
            raise_error(AttributeError, "State vector not available.")
        return self._vector

    @property
    def matrix(self):
        if self._matrix is None:
            raise_error(AttributeError, "Density matrix not available.")
        return self._matrix

    @vector.setter
    def vector(self, x):
        nqubits = int(math.log2(len(x)))
        if self._nqubits is None:
            self.nqubits = nqubits
        elif self._nqubits != nqubits:
            raise_error(ValueError, "Cannot assign vector of length {} to "
                                    "state with {} qubits."
                                    "".format(len(x), self.nqubits))
        self._vector = x

    @matrix.setter
    def matrix(self, x):
        nqubits = int(math.log2(len(x)))
        if self._nqubits is None:
            self.nqubits = nqubits
        elif self._nqubits != nqubits:
            raise_error(ValueError, "Cannot assign matrix of length {} to "
                                    "state with {} qubits."
                                    "".format(len(x), self.nqubits))
        self._matrix = x

    @abstractmethod
    def to_matrix(self):
        raise_error(NotImplementedError)

    @classmethod
    def from_vector(cls, x, nqubits=None):
        obj = cls()
        obj.vector = x
        return obj

    @classmethod
    def from_matrix(cls, x, nqubits=None):
        obj = cls()
        obj.matrix = x
        return obj

    @classmethod
    @abstractmethod
    def default(cls, nqubits, is_matrix=False):
        raise_error(NotImplementedError)

    @classmethod
    @abstractmethod
    def ones(cls, nqubits, is_matrix=False):
        raise_error(NotImplementedError)

    @classmethod
    @abstractmethod
    def random(cls, nqubits, is_matrix=False):
        raise_error(NotImplementedError)

    @property
    def tensor(self):
        if self._matrix is not None:
            return self.matrix
        return self.vector

    @property
    @abstractmethod
    def shape(self):
        raise_error(NotImplementedError)

    def __getitem__(self, i):
        return self.tensor[i]
