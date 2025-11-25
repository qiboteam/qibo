from abc import abstractmethod
from typing import Dict, Optional, Tuple

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

    @property
    @abstractmethod
    def matrix(self):  # pragma: no cover
        """Return the full matrix representation.

        For :math:`n` qubits, can be a dense :math:`2^{n} \\times 2^{n}` array or a sparse
        matrix, depending on how the Hamiltonian was created.
        """
        pass

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
    def expectation(self, circuit, nshots: Optional[int] = None):  # pragma: no cover
        """Computes the expectation value for a given circuit.

        Args:
            circuit (Circuit): circuit to calculate the expectation value from.
                If the circuit has already been executed, this will just make use of the cached
                result, otherwise it will execute the circuit.
            nshots (int, optional): number of shots to calculate the expectation value, if ``None``
                it will try to compute the exact expectation value (if possible). Defaults to ``None``.

        Returns:
            float: The expectation value.
        """
        raise_error(NotImplementedError)

    def expectation_from_state(self, state: "ndarray", normalize: bool = False):
        """Compute the expectation value starting from a quantum state.

        Args:
            state (ndarray): the quantum state.
            normalize (bool): whether to normalize the input state. Defaults to ``False``.
        Returns:
            (float) the expectation value.
        """
        if len(state.shape) == 2:
            return self.backend.calculate_expectation_density_matrix(  # pylint: disable=no-member
                self.matrix, state, normalize
            )
        return self.backend.calculate_expectation_state(  # pylint: disable=no-member
            self.matrix, state, normalize
        )

    @abstractmethod
    def expectation_from_samples(
        self,
        frequencies: Dict[str | int, int],
        qubit_map: Optional[Tuple[int, ...]] = None,
    ):  # pragma: no cover
        pass

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
