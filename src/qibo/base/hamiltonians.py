from qibo.config import raise_error


class Hamiltonian(object):
    """This class implements the abstract Hamiltonian operator.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape
            ``(2 ** nqubits, 2 ** nqubits)``.
    """

    NUMERIC_TYPES = None

    def __init__(self, nqubits, matrix):
        if not isinstance(nqubits, int):
            raise_error(RuntimeError, "nqubits must be an integer but is "
                                            "{}.".format(type(nqubits)))
        if nqubits < 0:
            raise_error(ValueError, "nqubits must be a positive integer but is "
                                    "{}".format(nqubits))
        shape = tuple(matrix.shape)
        if shape != 2 * (2 ** nqubits,):
            raise_error(ValueError, "The Hamiltonian is defined for {} qubits "
                                    "while the given matrix has shape {}."
                                    "".format(nqubits, shape))

        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    def _calculate_eigenvalues(self): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    def _calculate_eigenvectors(self): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    def _calculate_exp(self, a): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    def _eye(self, n=None): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    def eigenvalues(self):
        """Computes the eigenvalues for the Hamiltonian."""
        if self._eigenvalues is None:
            self._eigenvalues = self._calculate_eigenvalues()
        return self._eigenvalues

    def eigenvectors(self):
        """Computes a tensor with the eigenvectors for the Hamiltonian."""
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self._calculate_eigenvectors()
        return self._eigenvectors

    def exp(self, a):
        """Computes a tensor corresponding to exp(-1j * a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        if self._exp.get("a") != a:
            self._exp["a"] = a
            self._exp["result"] = self._calculate_exp(a)
        return self._exp.get("result")

    def expectation(self, state, normalize=False): # pragma: no cover
        """Computes the real expectation value for a given state.

        Args:
            state (array): the expectation state.
            normalize (bool): If ``True`` the expectation value is divided
                with the state's norm squared.

        Returns:
            Real number corresponding to the expectation value.
        """
        # abstract method
        raise_error(NotImplementedError)

    def __add__(self, o):
        """Add operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be added.")
            new_matrix = self.matrix + o.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, self.matrix + o * self._eye())
        else:
            raise_error(NotImplementedError, "Hamiltonian addition to {} not "
                                             "implemented.".format(type(o)))

    def __radd__(self, o):
        """Right operator addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be subtracted.")
            new_matrix = self.matrix - o.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, self.matrix - o * self._eye())
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))

    def __rsub__(self, o):
        """Right subtraction operator."""
        if isinstance(o, self.__class__): # pragma: no cover
            # impractical case because it will be handled by `__sub__`
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be added.")
            new_matrix = o.matrix - self.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, o * self._eye() - self.matrix)
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, self.NUMERIC_TYPES):
            new_matrix = self.matrix * o
            r = self.__class__(self.nqubits, new_matrix)
            if self._eigenvalues is not None:
                if o.real >= 0:
                    r._eigenvalues = o * self._eigenvalues
                else:
                    r._eigenvalues = o * self._eigenvalues[::-1]
            if self._eigenvectors is not None:
                if o.real > 0:
                    r._eigenvectors = self._eigenvectors
                elif o == 0:
                    r._eigenvectors = self._eye(int(self._eigenvectors.shape[0]))
            return r
        else:
            raise_error(NotImplementedError, "Hamiltonian multiplication to {} "
                                             "not implemented.".format(type(o)))

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)

    def __matmul__(self, o): # pragma: no cover
        """Matrix multiplication with other Hamiltonians or state vectors."""
        # abstract method
        raise_error(NotImplementedError)
