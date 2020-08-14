from qibo import gates
from qibo.config import raise_error


class Hamiltonian(object):
    """Abstract Hamiltonian operator using full matrix representation.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape
            ``(2 ** nqubits, 2 ** nqubits)``.
    """
    NUMERIC_TYPES = None
    ARRAY_TYPES = None
    K = None # calculation backend (numpy or TensorFlow)

    def __init__(self, nqubits, matrix):
        if not isinstance(nqubits, int):
            raise_error(RuntimeError, "nqubits must be an integer but is "
                                            "{}.".format(type(nqubits)))
        if nqubits < 1:
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

    def _calculate_exp(self, a): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    def eigenvalues(self):
        """Computes the eigenvalues for the Hamiltonian."""
        if self._eigenvalues is None:
            self._eigenvalues = self.K.linalg.eigvalsh(self.matrix)
        return self._eigenvalues

    def eigenvectors(self):
        """Computes a tensor with the eigenvectors for the Hamiltonian."""
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.K.linalg.eigh(self.matrix)
        return self._eigenvectors

    def exp(self, a):
        """Computes a tensor corresponding to exp(-1j * a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        if self._exp.get("a") != a:
            self._exp["a"] = a
            self._exp["result"] = self._calculate_exp(a) # pylint: disable=E1111
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

    def _eye(self, n=None):
        if n is None:
            n = int(self.matrix.shape[0])
        return self.K.eye(n, dtype=self.matrix.dtype)

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

    def _real(self, o):
        """Calculates real part of number or tensor."""
        return o.real

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, self.NUMERIC_TYPES) or isinstance(o, self.ARRAY_TYPES):
            new_matrix = self.matrix * o
            r = self.__class__(self.nqubits, new_matrix)
            if self._eigenvalues is not None:
                if self._real(o) >= 0:
                    r._eigenvalues = o * self._eigenvalues
                else:
                    r._eigenvalues = o * self._eigenvalues[::-1]
            if self._eigenvectors is not None:
                if self._real(o) > 0:
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

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            new_matrix = self.K.matmul(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.ARRAY_TYPES):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return self.K.matmul(self.matrix, o[:, self.K.newaxis])[:, 0]
            elif rank == 2: # matrix
                return self.K.matmul(self.matrix, o)
            else:
                raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                        "rank-{} tensor.".format(rank))
        else:
            raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                             "implemented.".format(type(o)))


class LocalHamiltonian(object):
    """Local Hamiltonian operator used for Trotterized time evolution.

    The Hamiltonian represented is a sum of two-qubit interaction terms as the
    example analyzed in Section 4.1 of
    `arXiv:1901.05824 <https://arxiv.org/abs/1901.05824>`_.
    Periodic boundary conditions are assumed.

    Args:
        terms (list): List of :class:`qibo.base.hamiltonians.Hamiltonian`
            objects that correspond to the local operators. The total
            Hamiltonian is the sum of all the terms in the list.

    Example:
        ::

            from qibo import matrices, hamiltonians
            # Create local term for critical TFIM Hamiltonian
            matrix = -np.kron(matrices.Z, matrices.Z) - np.kron(matrices.X, matrices.I)
            term = hamiltonians.Hamiltonian(2, matrix)
            # Create a ``LocalHamiltonian`` object corresponding to a critical TFIM for 5 qubits
            h = hamiltonians.LocalHamiltonian(5 * [term])
    """

    def __init__(self, terms):
        for term in terms:
            if not issubclass(type(term), Hamiltonian):
                raise_error(TypeError, "Invalid term type {}.".format(type(term)))
            if term.nqubits != 2:
                raise_error(ValueError, "LocalHamiltonian terms should target "
                                        "one or two qubits but targets {}."
                                        "".format(term.nqubits))
        self.nqubits = len(terms)
        self.terms = terms
        self._dt = None
        self._circuit = None

    def _terms(self, dt, odd=False):
        """Generates exponentials of the Trotter decomposition.

        Args:
            dt (float): Time step to use for the decomposition.
            odd (bool): If ``True`` it generates the exponentials for Hodd,
                otherwise for Heven (see paper for details).
        """
        for term in self.terms[int(odd)::2]:
            yield term.exp(dt / (2.0 - float(odd)))

    def _allterms(self, dt):
        """Generates all terms for the 2nd order Trotter decomposition."""
        for term in self._terms(dt):
            yield term
        for term in self._terms(dt, odd=True):
            yield term
        for term in self._terms(dt):
            yield term

    def _create_circuit(self, dt):
        """Creates circuit that implements the Trotterized evolution."""
        from qibo.models import Circuit
        # even gates
        even = lambda: (gates.Unitary(term, 2 * i, (2 * i + 1) % self.nqubits)
                        for i, term in enumerate(self._terms(dt)))
        # odd gates
        odd = (gates.Unitary(term, 2 * i + 1, (2 * i + 2) % self.nqubits)
               for i, term in enumerate(self._terms(dt, odd=True)))

        self._circuit = Circuit(self.nqubits)
        self._circuit.add(even())
        self._circuit.add(odd)
        self._circuit.add(even())

    def circuit(self, dt):
        """Circuit implementing second order Trotter time step.

        Args:
            dt (float): Time step to use for Trotterization.

        Returns:
            :class:`qibo.base.circuit.BaseCircuit` that implements a single
            time step of the second order Trotterized evolution.
        """
        if self._circuit is None:
            self._dt = dt
            self._create_circuit(dt)
        elif dt != self._dt:
            self._dt = dt
            self._circuit.set_parameters(list(self._allterms(dt)))
        return self._circuit
