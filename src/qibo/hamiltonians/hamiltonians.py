from qibo.config import raise_error, log
from qibo.hamiltonians.abstract import AbstractHamiltonian

class Hamiltonian(AbstractHamiltonian):
    """Hamiltonian based on a dense or sparse matrix representation.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape ``(2 ** nqubits, 2 ** nqubits)``.
            Sparse matrices based on ``scipy.sparse`` for numpy/qibojit backends
            or on ``tf.sparse`` for the tensorflow backend are also
            supported.
    """
    def __init__(self, nqubits, matrix=None, **kwargs):
        super().__init__(**kwargs)
        if not (isinstance(matrix, self.backend.tensor_types) or self.backend.issparse(matrix)):
            raise_error(TypeError, "Matrix of invalid type {} given during "
                                   "Hamiltonian initialization"
                                   "".format(type(matrix)))

        matrix = self.backend.cast(matrix)

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

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map):
        """Creates a ``Hamiltonian`` from a symbolic Hamiltonian.

        We refer to the :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
        example for more details.

        Args:
            symbolic_hamiltonian (sympy.Expr): The full Hamiltonian written
                with symbols.
            symbol_map (dict): Dictionary that maps each symbol that appears in
                the Hamiltonian to a pair of (target, matrix).

        Returns:
            A :class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian` object
            that implements the Hamiltonian represented by the given symbolic
            expression.
        """
        log.warning("`Hamiltonian.from_symbolic` and the use of symbol maps is "
                    "deprecated. Please use `SymbolicHamiltonian` and Qibo symbols "
                    "to construct Hamiltonians using symbols.")
        return SymbolicHamiltonian(symbolic_hamiltonian, symbol_map)

    def eigenvalues(self, k=6):
        if self._eigenvalues is None:
            self._eigenvalues = self.backend.calculate_eigenvalues(self.matrix, k)
        return self._eigenvalues

    def eigenvectors(self, k=6):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.backend.calculate_eigenvectors(self.matrix, k)
        return self._eigenvectors

    def exp(self, a):
        if self._exp.get("a") != a:
            self._exp["a"] = a
            self._exp["result"] = self.backend.calculate_exp(a, self._eigenvectors,
                                                             self._eigenvalues, self.matrix)
        return self._exp.get("result")

class SymbolicHamiltonian(AbstractHamiltonian):
    #TODO: update docstring
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian`.

    Calculations using symbolic Hamiltonians are either done directly using
    the given ``sympy`` expression as it is (``form``) or by parsing the
    corresponding ``terms`` (which are :class:`qibo.core.terms.SymbolicTerm`
    objects). The latter approach is more computationally costly as it uses
    a ``sympy.expand`` call on the given form before parsing the terms.
    For this reason the ``terms`` are calculated only when needed, for example
    during Trotterization.
    The dense matrix of the symbolic Hamiltonian can be calculated directly
    from ``form`` without requiring ``terms`` calculation (see
    :meth:`qibo.core.hamiltonians.SymbolicHamiltonian.calculate_dense` for details).

    Args:
        form (sympy.Expr): Hamiltonian form as a ``sympy.Expr``. Ideally the
            Hamiltonian should be written using Qibo symbols.
            See :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
            example for more details.
        symbol_map (dict): Dictionary that maps each ``sympy.Symbol`` to a tuple
            of (target qubit, matrix representation). This feature is kept for
            compatibility with older versions where Qibo symbols were not available
            and may be deprecated in the future.
            It is not required if the Hamiltonian is constructed using Qibo symbols.
            The symbol_map can also be used to pass non-quantum operator arguments
            to the symbolic Hamiltonian, such as the parameters in the
            :meth:`qibo.hamiltonians.MaxCut` Hamiltonian.
        ground_state (Callable): Function with no arguments that returns the
            ground state of this Hamiltonian. This is useful in cases where
            the ground state is trivial and is used for initialization,
            for example the easy Hamiltonian in adiabatic evolution,
            however we would like to avoid constructing and diagonalizing the
            full Hamiltonian matrix only to find the ground state.
    """

    def __init__(self, form=None, symbol_map={}, ground_state=None):
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
        assert isinstance(hamiltonian, Hamiltonian)
        self._dense = hamiltonian
        self._eigenvalues = hamiltonian._eigenvalues
        self._eigenvectors = hamiltonian._eigenvectors
        self._exp = hamiltonian._exp

    # @abstractmethod
    # def calculate_dense(self): # pragma: no cover
    #     raise_error(NotImplementedError)

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

    # @abstractmethod
    # def circuit(self, dt, accelerators=None, memory_device="/CPU:0"): # pragma: no cover
    #     raise_error(NotImplementedError)