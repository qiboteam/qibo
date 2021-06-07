import itertools
import sympy
from qibo import gates, K
from qibo.config import log, raise_error, EINSUM_CHARS
from qibo.abstractions import hamiltonians, states


def multikron(matrix_list):
    """Calculates Kronecker product of a list of matrices.

    Args:
        matrices (list): List of matrices as ``np.ndarray``s.

    Returns:
        ``np.ndarray`` of the Kronecker product of all ``matrices``.
    """
    h = 1
    for m in matrix_list:
        h = K.np.kron(h, m)
    return h


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
        matrix = K.qnp.expm(-1j * dt * self.matrix)
        return gates.Unitary(matrix, *self.target_qubits)

    def __mul__(self, x):
        return self.__class__(x * self.matrix, *self.target_qubits)

    def __rmul__(self, x):
        return self.__mul__(x)

    def __call__(self, state, density_matrix=False):
        if density_matrix:
            self.gate.density_matrix = True
            return self.gate.density_matrix_half_call(state)
        return self.gate(state)


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


def parse_symbolic(hamiltonian, symbol_map):
    """Parses symbolic Hamiltonians defined using ``sympy``.

    This class should not be used by users.
    It is used internally to help creating
    :class:`qibo.abstractions.hamiltonians.Hamiltonian` and
    :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian` objects for Hamiltonians
    defined using symbols. For more information we refer to the
    :meth:`qibo.abstractions.hamiltonians.Hamiltonian.from_symbolic`
    and :meth:`qibo.abstractions.hamiltonians.TrotterHamiltonian.from_symbolic` methods.

    Args:
        symbolic_hamiltonian (sympy.Expr): The full Hamiltonian written with
            symbols.
        symbol_map (dict): Dictionary that maps each symbol to a pair of
            (target, matrix).

    Returns:
        terms (dict): Dictionary that maps tuple of qubit ids to tuple of the
            corresponding matrices. Each dictionary element corresponds to
            a Hamiltonian term.
    """
    if not issubclass(hamiltonian.__class__, sympy.Expr):
        raise_error(TypeError, "Symbolic Hamiltonian should be a `sympy` "
                               "expression but is {}."
                               "".format(type(hamiltonian)))
    if not isinstance(symbol_map, dict):
        raise_error(TypeError, "Symbol map must be a dictionary but is "
                               "{}.".format(type(symbol_map)))
    for k, v in symbol_map.items():
        if not isinstance(k, sympy.Symbol):
            raise_error(TypeError, "Symbol map keys must be `sympy.Symbol` "
                                   "but {} was found.".format(type(k)))
        if not isinstance(v, tuple):
            raise_error(TypeError, "Symbol map values must be tuples but "
                                   "{} was found.".format(type(v)))
        if len(v) != 2:
            raise_error(ValueError, "Symbol map values must be tuples of "
                                    "length 2 but length {} was found."
                                    "".format(len(v)))

    def check_symbolmap(s):
        """Checks if symbol exists in the given symbol map."""
        if s not in symbol_map:
            raise_error(ValueError, "Symbolic Hamiltonian contains symbol "
                                    "{} which does not exist in the symbol "
                                    "map.".format(s))

    symbolic = sympy.expand(hamiltonian)

    term_dict = symbolic.as_coefficients_dict()
    terms = {tuple(): (0,)}
    dtype = K.qnp.dtypes('DTYPECPX')
    if 1 in term_dict:
        terms[tuple()] += (dtype(term_dict.pop(1)),)
    for term, coeff in term_dict.items():
        targets, matrices = [], [dtype(coeff)]
        for factor in term.as_ordered_factors():
            if factor.is_symbol:
                check_symbolmap(factor)
                itarget = symbol_map[factor][0]
                ivalues = symbol_map[factor][1]
                if isinstance(ivalues, K.numeric_types):
                    matrices[0] *= ivalues
                else:
                    targets.append(itarget)
                    matrices.append(ivalues)
            elif isinstance(factor, sympy.Pow):
                base, pow = factor.args
                assert isinstance(pow, sympy.Integer)
                check_symbolmap(base)
                targets.append(symbol_map[base][0])
                matrix = symbol_map[base][1]
                for _ in range(int(pow) - 1):
                    matrix = matrix.dot(matrix)
                matrices.append(matrix)
            elif factor == sympy.I: # imaginary unit
                matrices[0] *= 1j
            else:
                raise_error(ValueError, f"Cannot parse factor {factor}.")
        targets, matrices = tuple(targets), tuple(matrices)
        if targets in terms:
            terms[targets] += matrices
        else:
            terms[targets] = matrices
    terms[tuple()] = sum(terms[tuple()]) # sum of constant terms
    return terms


def reduce_pairs(pair_sets, pair_map, free_targets):
    """Helper method for ``merge_one_qubit``.

    Finds the one and two qubit term merge map using an recursive procedure.

    Args:
        pair_sets (dict): Dictionary that maps each qubit id to a set of
            pairs that contain this qubit.
        pair_map (dict): Map from qubit id to the pair that this qubit will
            be merged with.
        free_targets (set): Set of qubit ids that are still not mapped to
            a pair in the ``pair_map``.

    Returns:
        pair_map (dict): The final map from qubit ids to pairs once the
            recursion finishes. If the returned map is ``None`` then the
            procedure failed and the merging is aborted.
    """
    def assign_target(target):
        """Assigns a pair to a qubit.

        This moves ``target`` from ``free_targets`` to ``pair_map``.
        """
        pair = pair_sets[target].pop()
        pair_map[target] = pair
        pair_sets.pop(target)
        target2 = pair[1] if pair[0] == target else pair[0]
        if target2 in pair_sets:
            pair_sets[target2].remove(pair)

    # Assign pairs to qubits that have a single available pair
    flag = True
    for target in set(free_targets):
        if target not in pair_sets or not pair_sets[target]:
            return None
        if len(pair_sets[target]) == 1:
            assign_target(target)
            free_targets.remove(target)
            flag = False
    # If all qubits were mapped to pairs return the result
    if not free_targets:
        return pair_map
    # If no qubits with a single available pair were found above, then
    # assign a pair randomly (not sure about this step!)
    if flag:
        target = free_targets.pop()
        assign_target(target)
    # Recurse
    return reduce_pairs(pair_sets, pair_map, free_targets)


def merge_one_qubit(terms, symbolic_terms):
    """Merges one-qubit matrices to the two-qubit terms for efficiency.

    This works for Hamiltonians with one and two qubit terms only.
    The two qubit terms should be sufficiently many so that every
    qubit appears as the first target at least once.

    Args:
        terms (dict): Dictionary that maps tuples of targets to the matrix
                      that acts on these on targets.

    Returns:
        The given ``terms`` dictionary updated so that one-qubit terms
        are merged to two-qubit ones.
    """
    one_qubit, two_qubit, pair_sets = dict(), dict(), dict()
    for targets, matrix in terms.items():
        assert len(targets) in {1, 2}
        if len(targets) == 1:
            one_qubit[targets[0]] = matrix
        else:
            two_qubit[targets] = matrix
            for t in targets:
                if t in pair_sets:
                    pair_sets[t].add(targets)
                else:
                    pair_sets[t] = {targets}

    free_targets = set(one_qubit.keys())
    pair_map = reduce_pairs(pair_sets, dict(), free_targets)
    if pair_map is None:
        log.info("Aborting merge of one and two-qubit terms during "
                 "TrotterHamiltonian creation because the two-qubit "
                 "terms are not sufficiently many.")
        return terms

    merged = dict()
    for target, pair in pair_map.items():
        two_qubit.pop(pair)
        if target == pair[0]:
            matrix = terms[pair]
        else:
            matrices = symbolic_terms[pair]
            pair = (pair[1], pair[0])
            matrix = 0
            for i in range(0, len(matrices), 3):
                matrix += matrices[i] * multikron(matrices[i + 2: i:-1])
        eye = K.np.eye(2, dtype=matrix.dtype)
        merged[pair] = K.np.kron(one_qubit[target], eye) + matrix
    merged.update(two_qubit)
    return merged
