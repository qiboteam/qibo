import sympy
from qibo import gates, matrices, K
from qibo.config import raise_error


class Symbol(sympy.Symbol):
    """Qibo specialization for ``sympy`` symbols.

    These symbols can be used to create :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
    See :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
    for more details.

    Example:
        .. testcode::

            from qibo import hamiltonians
            from qibo.symbols import X, Y, Z
            # construct a XYZ Hamiltonian on two qubits using Qibo symbols
            form = X(0) * X(1) + Y(0) * Y(1) + Z(0) * Z(1)
            ham = hamiltonians.SymbolicHamiltonian(form)

    Args:
        q (int): Target qubit id.
        matrix (np.ndarray): 2x2 matrix represented by this symbol.
        name (str): Name of the symbol which defines how it is represented in
            symbolic expressions.
        commutative (bool): If ``True`` the constructed symbols commute with
            each other. Default is ``False``.
            This argument should be used with caution because quantum operators
            are not commutative objects and therefore switching this to ``True``
            may lead to wrong results. It is useful for improving performance
            in symbolic calculations in cases where the user is sure that
            the operators participating in the Hamiltonian form are commuting
            (for example when the Hamiltonian consists of Z terms only).
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

    def full_matrix(self, nqubits):
        """Calculates the full dense matrix corresponding to the symbol as part of a bigger system.

        Args:
            nqubits (int): Total number of qubits in the system.

        Returns:
            Matrix of dimension (2^nqubits, 2^nqubits) composed of the Kronecker
            product between identities and the symbol's single-qubit matrix.
        """
        from qibo.hamiltonians import multikron
        matrix_list = self.target_qubit * [matrices.I]
        matrix_list.append(self.matrix)
        n = nqubits - self.target_qubit - 1
        matrix_list.extend(matrices.I for _ in range(n))
        return multikron(matrix_list)


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


class I(PauliSymbol):
    """Qibo symbol for the identity operator.

    Args:
        q (int): Target qubit id.
    """
    pass


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
