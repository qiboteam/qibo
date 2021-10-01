import sympy
from qibo import gates, matrices, K
from qibo.config import raise_error


class Symbol(sympy.Symbol):
    """Qibo specialization for ``sympy`` symbols.

    These symbols can be used to create :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.

    Args:
        q (int): Target qubit id.
        matrix (np.ndarray): 2x2 matrix represented by this symbol.
        name (str): Name of the symbol which defines how it is represented in
            symbolic expressions.
    """

    def __new__(cls, q, matrix=None, name="Symbol", commutative=False):
        name = "{}{}".format(name, q)
        return super().__new__(cls=cls, name=name, commutative=commutative)

    def __init__(self, q, matrix=None, name="Symbol", commutative=False):
        self.target_qubit = q
        self._gate = None
        if not (matrix is None or isinstance(matrix, K.qnp.numeric_types) or
                isinstance(matrix, K.qnp.tensor_types)):
            raise_error(TypeError, "Invalid type {} of symbol matrix."
                                   "".format(type(matrix)))
        self.matrix = matrix

    @property
    def gate(self):
        """Qibo gate that implements the action of the symbol on states."""
        if self._gate is None:
            self._gate = self.calculate_gate()
        return self._gate

    def calculate_gate(self):
        return gates.Unitary(self.matrix, self.target_qubit)


class PauliSymbol(Symbol):

    def __new__(cls, q, commutative=False):
        matrix = getattr(matrices, cls.__name__)
        return super().__new__(cls, q, matrix, cls.__name__, commutative)

    def __init__(self, q, commutative=False):
        name = self.__class__.__name__
        matrix = getattr(matrices, name)
        super().__init__(q, matrix, name, commutative)

    def calculate_gate(self):
        name = self.__class__.__name__
        return getattr(gates, name)(self.target_qubit)


class X(PauliSymbol):
    """Qibo symbol for the Pauli-X operator.

    Args:
        q (int): Target qubit id.
    """
    pass


class Y(PauliSymbol):
    """Qibo symbol for the Pauli-X operator.

    Args:
        q (int): Target qubit id.
    """
    pass


class Z(PauliSymbol):
    """Qibo symbol for the Pauli-X operator.

    Args:
        q (int): Target qubit id.
    """
    pass
