from qibo.config import raise_error, log
from qibo.hamiltonians.abstract import Hamiltonian as AbstractHamiltonian
from qibo.hamiltonians.symbolic import SymbolicHamiltonian

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
        #TODO: get rid of tensor types
        # if not (isinstance(matrix, K.tensor_types) or self.backend.issparse(matrix)):
        if not self.backend.issparse(matrix):
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
            self._eigenvalues = self.backend.compute_eigenvalues(self.matrix, k)
        return self._eigenvalues

    def eigenvectors(self, k=6):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.backend.compute_eigenvectors(self.matrix, k)
        return self._eigenvectors