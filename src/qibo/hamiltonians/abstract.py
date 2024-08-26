from abc import abstractmethod

from qibo.config import raise_error


class AbstractHamiltonian:
    """Qibo abstraction for Hamiltonian objects."""

    def __init__(self):
        self._nqubits = None

    @property
    def nqubits(self):
        return self._nqubits

    @nqubits.setter
    def nqubits(self, n):
        if not isinstance(n, int):
            raise_error(RuntimeError, f"nqubits must be an integer but is {type(n)}.")
        if n < 1:
            raise_error(ValueError, f"nqubits must be a positive integer but is {n}")
        self._nqubits = n

    @abstractmethod
    def eigenvalues(self, k=6):  # pragma: no cover
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
    def eigenvectors(self, k=6):  # pragma: no cover
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

        Uses :meth:`qibo.hamiltonians.AbstractHamiltonian.eigenvectors`
        and returns eigenvector corresponding to the lowest energy.
        """
        return self.eigenvectors()[:, 0]

    @abstractmethod
    def exp(self, a):  # pragma: no cover
        """Computes a tensor corresponding to :math:`\\exp(-i \\, a \\, H)`.

        Args:
            a (complex): Complex number to multiply Hamiltonian before exponentiation.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def expectation(self, state, normalize=False):  # pragma: no cover
        """Computes the real expectation value for a given state.

        Args:
            state (ndarray): state in which to calculate the expectation value.
            normalize (bool, optional): If ``True``, the expectation value
                :math:`\\ell_{2}`-normalized. Defaults to ``False``.

        Returns:
            float: real number corresponding to the expectation value.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def expectation_from_samples(self, freq, qubit_map=None):  # pragma: no cover
        """Computes the expectation value of a diagonal observable,
        given computational-basis measurement frequencies.

        Args:
            freq (collections.Counter): the keys are the observed values in binary form
                and the values the corresponding frequencies, that is the number
                of times each measured value/bitstring appears.
            qubit_map (tuple): Mapping between frequencies and qubits.
                If ``None``, then defaults to
                :math:`[1, \\, 2, \\, \\cdots, \\, \\mathrm{len}(\\mathrm{key})]`.
                Defaults to ``None``.

        Returns:
            float: real number corresponding to the expectation value.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def __add__(self, o):  # pragma: no cover
        """Add operator."""
        raise_error(NotImplementedError)

    def __radd__(self, o):
        """Right operator addition."""
        return self.__add__(o)

    @abstractmethod
    def __sub__(self, o):  # pragma: no cover
        """Subtraction operator."""
        raise_error(NotImplementedError)

    @abstractmethod
    def __rsub__(self, o):  # pragma: no cover
        """Right subtraction operator."""
        raise_error(NotImplementedError)

    @abstractmethod
    def __mul__(self, o):  # pragma: no cover
        """Multiplication to scalar operator."""
        raise_error(NotImplementedError)

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)

    @abstractmethod
    def __matmul__(self, o):  # pragma: no cover
        """Matrix multiplication with other Hamiltonians or state vectors."""
        raise_error(NotImplementedError)
