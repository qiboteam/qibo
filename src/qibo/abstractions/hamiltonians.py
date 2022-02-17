from abc import ABC, abstractmethod
from qibo.config import log, raise_error


class AbstractHamiltonian(ABC):
    """Qibo abstraction for Hamiltonian objects."""

    def __init__(self):
        self._nqubits = None

    @property
    def nqubits(self):
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n):
        if not isinstance(n, int):
            raise_error(RuntimeError, "nqubits must be an integer but is "
                                      "{}.".format(type(n)))
        if n < 1:
            raise_error(ValueError, "nqubits must be a positive integer but is "
                                    "{}".format(n))
        self._nqubits = n

    @abstractmethod
    def eigenvalues(self, k=6): # pragma: no cover
        """Computes the eigenvalues for the Hamiltonian.

        Args:
            k (int): Number of eigenvalues to calculate if the Hamiltonian
                was created using a sparse matrix. This argument is ignored
                if the Hamiltonian was created using a dense matrix.
                See :meth:`qibo.backends.abstract.AbstractBackend.eigvalsh` for 
                more details.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def eigenvectors(self, k=6): # pragma: no cover
        """Computes a tensor with the eigenvectors for the Hamiltonian.

        Args:
            k (int): Number of eigenvalues to calculate if the Hamiltonian
                was created using a sparse matrix. This argument is ignored
                if the Hamiltonian was created using a dense matrix.
                See :meth:`qibo.backends.abstract.AbstractBackend.eigh` for
                more details.
        """
        raise_error(NotImplementedError)

    def ground_state(self):
        """Computes the ground state of the Hamiltonian.

        Uses :meth:`qibo.abstractions.hamiltonians.AbstractHamiltonian.eigenvectors`
        and returns eigenvector corresponding to the lowest energy.
        """
        return self.eigenvectors()[:, 0]

    @abstractmethod
    def exp(self, a): # pragma: no cover
        """Computes a tensor corresponding to exp(-1j * a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def expectation(self, state, normalize=False): # pragma: no cover
        """Computes the real expectation value for a given state.

        Args:
            state (array): the expectation state.
            normalize (bool): If ``True`` the expectation value is divided
                with the state's norm squared.

        Returns:
            Real number corresponding to the expectation value.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def __add__(self, o): # pragma: no cover
        """Add operator."""
        raise_error(NotImplementedError)

    def __radd__(self, o):
        """Right operator addition."""
        return self.__add__(o)

    @abstractmethod
    def __sub__(self, o): # pragma: no cover
        """Subtraction operator."""
        raise_error(NotImplementedError)

    @abstractmethod
    def __rsub__(self, o): # pragma: no cover
        """Right subtraction operator."""
        raise_error(NotImplementedError)

    @abstractmethod
    def __mul__(self, o): # pragma: no cover
        """Multiplication to scalar operator."""
        raise_error(NotImplementedError)

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)

    @abstractmethod
    def __matmul__(self, o): # pragma: no cover
        """Matrix multiplication with other Hamiltonians or state vectors."""
        raise_error(NotImplementedError)


class MatrixHamiltonian(AbstractHamiltonian):
    """Abstract Hamiltonian based on full matrix representation."""

    def __init__(self, nqubits, matrix=None):
        super().__init__()
        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    @property
    def matrix(self):
        """Returns the full matrix representation.

        Can be a dense ``(2 ** nqubits, 2 ** nqubits)`` array or a sparse
        matrix, depending on how the Hamiltonian was created.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, m):
        shape = tuple(m.shape)
        if shape != 2 * (2 ** self.nqubits,):
            raise_error(ValueError, "The Hamiltonian is defined for {} qubits "
                                    "while the given matrix has shape {}."
                                    "".format(self.nqubits, shape))
        self._matrix = m


class SymbolicHamiltonian(AbstractHamiltonian):
    """Abstract Hamiltonian based on symbolic representation.

    Unlike :class:`qibo.abstractions.hamiltonians.MatrixHamiltonian` this
    object does not create the full ``(2 ** nqubits, 2 ** nqubits)``
    Hamiltonian matrix leading to more efficient calculations.
    Note that the matrix is required and will be created automatically if
    specific methods, such as ``.eigenvectors()`` or ``.exp()`` are called.
    """

    def __init__(self, ground_state=None):
        super().__init__()
        self._dense = None
        self._ground_state = ground_state

    @property
    def dense(self):
        """Creates the equivalent :class:`qibo.abstractions.hamiltonians.MatrixHamiltonian`."""
        if self._dense is None:
            log.warning("Calculating the dense form of a symbolic Hamiltonian. "
                        "This operation is memory inefficient.")
            self.dense = self.calculate_dense()
        return self._dense

    @dense.setter
    def dense(self, hamiltonian):
        assert isinstance(hamiltonian, MatrixHamiltonian)
        self._dense = hamiltonian
        self._eigenvalues = hamiltonian._eigenvalues
        self._eigenvectors = hamiltonian._eigenvectors
        self._exp = hamiltonian._exp

    @abstractmethod
    def calculate_dense(self): # pragma: no cover
        raise_error(NotImplementedError)

    @property
    def matrix(self):
        """Returns the full ``(2 ** nqubits, 2 ** nqubits)`` matrix representation."""
        return self.dense.matrix

    def eigenvalues(self, k=6):
        return self.dense.eigenvalues(k)

    def eigenvectors(self, k=6):
        return self.dense.eigenvectors(k)

    def ground_state(self):
        if self._ground_state is None:
            log.warning("Ground state for this Hamiltonian was not given.")
            return self.eigenvectors()[:, 0]
        return self._ground_state()

    def exp(self, a):
        return self.dense.exp(a)

    @abstractmethod
    def circuit(self, dt, accelerators=None, memory_device="/CPU:0"): # pragma: no cover
        raise_error(NotImplementedError)
