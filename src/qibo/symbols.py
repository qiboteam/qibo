import sympy
from qibo import gates, matrices, K


class Symbol(sympy.Symbol):

    def __new__(cls, q, matrix=None, name="Symbol"):
        name = "{}{}".format(name, q)
        return super().__new__(cls=cls, name=name)

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
        if self._gate is None:
            self._gate = self.calculate_gate()
        return self._gate

    def calculate_gate(self):
        return gates.Unitary(self.matrix, *self.target_qubits)


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
    pass


class Y(PauliSymbol):
    pass


class Z(PauliSymbol):
    pass


class SymbolicTerm(list):

    def __init__(self, coefficient, factors):
        super().__init__()
        self.coefficient = coefficient

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

            if isinstance(factor, sympy.Symbol):
                if isinstance(factor.matrix, K.qnp.tensor_types):
                    self.extend(pow * [factor])
                else:
                    self.coefficient *= factor.matrix
            elif factor == sympy.I:
                self.coefficient *= 1j
            else:
                raise_error(TypeError, "Cannot parse factor {}.".format(factor))
