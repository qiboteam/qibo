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

    def __new__(cls, q, matrix=None, name="Symbol"):
        name = "{}{}".format(name, q)
        return super().__new__(cls=cls, name=name, commutative=False)

    def __init__(self, q, matrix=None, name="Symbol"):
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

    def __new__(cls, q):
        return super().__new__(cls=cls, q=q, name=cls.__name__)

    def __init__(self, q):
        self.target_qubit = q
        self._gate = None
        self.matrix = getattr(matrices, self.__class__.__name__)

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


class SymbolicTerm(list):
    """Helper method for parsing symbolic Hamiltonian terms.

    Each :class:`qibo.symbols.SymbolicTerm` corresponds to a term in the
    Hamiltonian.

    Example:
        ::

            from qibo.symbols import X, Y, SymbolicTerm
            sham = X(0) * X(1) + 2 * Y(0) * Y(1)
            termsdict = sham.as_coefficients_dict()
            sterms = [SymbolicTerm(c, f) for f, c in termsdict.items()]

    Args:
        coefficient (complex): Complex number coefficient of the underlying
            term in the Hamiltonian.
        factors (sympy.Expr): Sympy expression for the underlying term.
        symbol_map (dict): Dictionary that maps symbols in the given ``factors``
            expression to tuples of (target qubit id, matrix).
            This is required only if the expression is not created using Qibo
            symbols and to keep compatibility with older versions where Qibo
            symbols were not available.
    """

    def __init__(self, coefficient, factors, symbol_map=None):
        super().__init__()
        ordered_factors = []
        if factors != 1:
            ordered_factors = factors.as_ordered_factors()

        for factor in ordered_factors:
            if isinstance(factor, sympy.Pow):
                factor, pow = factor.args
                assert isinstance(pow, sympy.Integer)
                assert isinstance(factor, sympy.Symbol)
                pow = int(pow)
            else:
                pow = 1

            if symbol_map is not None and factor in symbol_map:
                q, matrix = symbol_map.get(factor)
                factor = Symbol(q, matrix, name=factor.name)

            if isinstance(factor, sympy.Symbol):
                if isinstance(factor.matrix, K.qnp.tensor_types):
                    self.extend(pow * [factor])
                else:
                    coefficient *= factor.matrix
            elif factor == sympy.I:
                coefficient *= 1j
            else: # pragma: no cover
                raise_error(TypeError, "Cannot parse factor {}.".format(factor))

        self.coefficient = complex(coefficient)
