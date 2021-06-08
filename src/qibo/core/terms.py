import sympy
from qibo import gates, K
from qibo.config import raise_error


class HamiltonianTerm:

    def __init__(self, matrix, *q):
        self.target_qubits = tuple(q)
        self._gate = None
        if not (matrix is None or isinstance(matrix, K.qnp.numeric_types) or
                isinstance(matrix, K.qnp.tensor_types)):
            raise_error(TypeError, "Invalid type {} of symbol matrix."
                                   "".format(type(matrix)))
        self._matrix = matrix

    @property
    def matrix(self):
        return self._matrix

    @property
    def gate(self):
        """Qibo gate that implements the action of the term on states."""
        if self._gate is None:
            self._gate = gates.Unitary(self.matrix, *self.target_qubits)
        return self._gate

    def exp(self, dt):
        return K.qnp.expm(-1j * dt * self.matrix)

    def expgate(self, dt):
        return gates.Unitary(self.exp(dt), *self.target_qubits)

    def __mul__(self, x):
        return self.__class__(x * self.matrix, *self.target_qubits)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __call__(self, state, density_matrix=False):
        if density_matrix:
            self.gate.density_matrix = True
            return self.gate.density_matrix_half_call(state)
        return self.gate(state) # pylint: disable=E1102


class SymbolicTerm(HamiltonianTerm):
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

    def __init__(self, coefficient, factors=[], matrix_map={}):
        self.coefficient = complex(coefficient)
        self.factors = factors
        self.matrix_map = matrix_map
        self._matrix = None
        self._gate = None
        self.target_qubits = tuple(sorted(self.matrix_map.keys()))

    @classmethod
    def from_factors(cls, coefficient, factors, symbol_map=None):
        if factors == 1:
            return cls(coefficient)

        _factors = []
        _matrix_map = {}
        for factor in factors.as_ordered_factors():
            if isinstance(factor, sympy.Pow):
                factor, pow = factor.args
                assert isinstance(pow, sympy.Integer)
                assert isinstance(factor, sympy.Symbol)
                pow = int(pow)
            else:
                pow = 1

            if symbol_map is not None and factor in symbol_map:
                from qibo.symbols import Symbol
                q, matrix = symbol_map.get(factor)
                factor = Symbol(q, matrix, name=factor.name)

            if isinstance(factor, sympy.Symbol):
                if isinstance(factor.matrix, K.qnp.tensor_types):
                    _factors.extend(pow * [factor])
                    q = factor.target_qubit
                    if q in _matrix_map:
                        _matrix_map[q].extend(pow * [factor.matrix])
                    else:
                        _matrix_map[q] = pow * [factor.matrix]
                else:
                    coefficient *= factor.matrix
            elif factor == sympy.I:
                coefficient *= 1j
            else: # pragma: no cover
                raise_error(TypeError, "Cannot parse factor {}.".format(factor))

        return cls(coefficient, _factors, _matrix_map)

    @property
    def matrix(self):
        """Calculates the full matrix corresponding to this term.

        Returns:
            Matrix as a ``np.ndarray`` of shape ``(2 ** ntargets, 2 ** ntargets)``
            where ``ntargets`` is the number of qubits included in the factors
            of this term.
        """
        if self._matrix is None:
            def matrices_product(matrices):
                if len(matrices) == 1:
                    return matrices[0]
                matrix = K.np.copy(matrices[0])
                for m in matrices[1:]:
                    matrix = matrix @ m
                return matrix

            self._matrix = self.coefficient
            for q in self.target_qubits:
                matrix = matrices_product(self.matrix_map.get(q))
                self._matrix = K.np.kron(self._matrix, matrix)
        return self._matrix

    def __mul__(self, x):
        new = self.__class__(self.coefficient, self.factors, self.matrix_map)
        new._matrix = self._matrix
        new._gate = self._gate
        new.coefficient *= x
        return new

    def __call__(self, state, density_matrix=False):
        for factor in self.factors:
            if density_matrix:
                factor.gate.density_matrix = True
                state = factor.gate.density_matrix_half_call(state)
            else:
                state = factor.gate(state)
        return self.coefficient * state
