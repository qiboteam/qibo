import itertools
from qibo import K
from qibo.config import log, raise_error, EINSUM_CHARS
from qibo.abstractions import hamiltonians, states


class Hamiltonian(hamiltonians.Hamiltonian):
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.Hamiltonian`."""

    def __new__(cls, nqubits, matrix, numpy=False):
        if not isinstance(matrix, K.tensor_types):
            raise_error(TypeError, "Matrix of invalid type {} given during "
                                   "Hamiltonian initialization"
                                   "".format(type(matrix)))
        if numpy:
            return NumpyHamiltonian(nqubits, matrix, numpy=True)
        else:
            return super().__new__(cls)

    def __init__(self, nqubits, matrix, numpy=False):
        assert not numpy
        self.K = K
        super().__init__(nqubits, self.K.cast(matrix), numpy=numpy)

    def eigenvalues(self):
        if self._eigenvalues is None:
            self._eigenvalues = self.K.eigvalsh(self.matrix)
        return self._eigenvalues

    def eigenvectors(self):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.K.eigh(self.matrix)
        return self._eigenvectors

    def exp(self, a):
        """Computes a tensor corresponding to exp(-1j * a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        if self._exp.get("a") != a:
            self._exp["a"] = a
            if self._eigenvectors is None:
                self._exp["result"] = self.K.expm(-1j * a * self.matrix)
            else:
                expd = self.K.diag(self.K.exp(-1j * a * self._eigenvalues))
                ud = self.K.transpose(self.K.conj(self._eigenvectors))
                self._exp["result"] = self.K.matmul(
                    self._eigenvectors, self.K.matmul(expd, ud))
        return self._exp.get("result")

    def expectation(self, state, normalize=False):
        if isinstance(state, states.AbstractState):
            ev = state.expectation(self, normalize)
        elif isinstance(state, K.tensor_types):
            shape = tuple(state.shape)
            if len(shape) == 1: # state vector
                from qibo.core.states import VectorState
                sobj = VectorState.from_tensor(state)
            elif len(shape) == 2: # density matrix
                from qibo.core.states import MatrixState
                sobj = MatrixState.from_tensor(state)
            else:
                raise_error(ValueError, "Cannot calculate Hamiltonian "
                                        "expectation value for state of shape "
                                        "{}.".format(shape))
            ev = sobj.expectation(self, normalize)
        else:
            raise_error(TypeError, "Cannot calculate Hamiltonian expectation "
                                   "value for state of type {}."
                                   "".format(type(state)))
        return ev

    def eye(self, n=None):
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
        elif isinstance(o, K.numeric_types):
            new_matrix = self.matrix + o * self.eye()
        else:
            raise_error(NotImplementedError, "Hamiltonian addition to {} not "
                                             "implemented.".format(type(o)))
        return self.__class__(self.nqubits, new_matrix)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be subtracted.")
            new_matrix = self.matrix - o.matrix
        elif isinstance(o, K.numeric_types):
            new_matrix = self.matrix - o * self.eye()
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))
        return self.__class__(self.nqubits, new_matrix)

    def __rsub__(self, o):
        """Right subtraction operator."""
        if isinstance(o, self.__class__): # pragma: no cover
            # impractical case because it will be handled by `__sub__`
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be added.")
            new_matrix = o.matrix - self.matrix
        elif isinstance(o, K.numeric_types):
            new_matrix = o * self.eye() - self.matrix
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))
        return self.__class__(self.nqubits, new_matrix)

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, self.K.Tensor):
            o = self.K.cast(o, dtype=self.matrix.dtype)
        if isinstance(o, K.numeric_types) or isinstance(o, K.tensor_types):
            new_matrix = self.matrix * o
            r = self.__class__(self.nqubits, new_matrix)
            if self._eigenvalues is not None:
                if K.qnp.cast(o).real >= 0:
                    r._eigenvalues = o * self._eigenvalues
                else:
                    r._eigenvalues = o * self._eigenvalues[::-1]
            if self._eigenvectors is not None:
                if K.qnp.cast(o).real > 0:
                    r._eigenvectors = self._eigenvectors
                elif o == 0:
                    r._eigenvectors = self.eye(int(self._eigenvectors.shape[0]))
            return r
        else:
            raise_error(NotImplementedError, "Hamiltonian multiplication to {} "
                                             "not implemented.".format(type(o)))

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            new_matrix = self.K.matmul(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)

        if isinstance(o, states.AbstractState):
            o = o.tensor
        if isinstance(o, K.tensor_types):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return self.K.matmul(self.matrix, o[:, self.K.newaxis])[:, 0]
            elif rank == 2: # matrix
                return self.K.matmul(self.matrix, o)
            else:
                raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                        "rank-{} tensor.".format(rank))

        raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                         "implemented.".format(type(o)))


class NumpyHamiltonian(Hamiltonian):

    def __new__(cls, nqubits, matrix, numpy=True):
        return hamiltonians.Hamiltonian.__new__(cls)

    def __init__(self, nqubits, matrix, numpy=True):
        assert numpy
        self.K = K.qnp
        hamiltonians.Hamiltonian.__init__(self, nqubits, self.K.cast(matrix),
                                          numpy=numpy)


class SymbolicHamiltonian:
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
    """
    import sympy
    from qibo import matrices

    def __init__(self, hamiltonian, symbol_map):
        if not issubclass(hamiltonian.__class__, self.sympy.Expr):
            raise_error(TypeError, "Symbolic Hamiltonian should be a `sympy` "
                                   "expression but is {}."
                                   "".format(type(hamiltonian)))
        if not isinstance(symbol_map, dict):
            raise_error(TypeError, "Symbol map must be a dictionary but is "
                                   "{}.".format(type(symbol_map)))
        for k, v in symbol_map.items():
            if not isinstance(k, self.sympy.Symbol):
                raise_error(TypeError, "Symbol map keys must be `sympy.Symbol` "
                                       "but {} was found.".format(type(k)))
            if not isinstance(v, tuple):
                raise_error(TypeError, "Symbol map values must be tuples but "
                                       "{} was found.".format(type(v)))
            if len(v) != 2:
                raise_error(ValueError, "Symbol map values must be tuples of "
                                        "length 2 but length {} was found."
                                        "".format(len(v)))
        self.symbolic = self.sympy.expand(hamiltonian)
        self.map = symbol_map

        term_dict = self.symbolic.as_coefficients_dict()
        self.constant = 0
        if 1 in term_dict:
            self.constant = self.matrices.dtype(term_dict.pop(1))
        self.terms = dict()
        target_ids = set()
        for term, coeff in term_dict.items():
            targets, matrices = [], [self.matrices.dtype(coeff)]
            for factor in term.as_ordered_factors():
                if factor.is_symbol:
                    self._check_symbolmap(factor)
                    itarget = self.map[factor][0]
                    ivalues = self.map[factor][1]
                    if isinstance(ivalues, K.numeric_types):
                        matrices[0] *= ivalues
                    else:
                        targets.append(itarget)
                        matrices.append(ivalues)
                elif isinstance(factor, self.sympy.Pow):
                    base, pow = factor.args
                    assert isinstance(pow, self.sympy.Integer)
                    self._check_symbolmap(base)
                    targets.append(self.map[base][0])
                    matrix = self.map[base][1]
                    for _ in range(int(pow) - 1):
                        matrix = matrix.dot(matrix)
                    matrices.append(matrix)
                else:
                    raise_error(ValueError, f"Cannot parse factor {factor}.")
            target_ids |= set(targets)
            targets, matrices = tuple(targets), tuple(matrices)
            if targets in self.terms:
                self.terms[targets] += matrices
            else:
                self.terms[targets] = matrices
        self.nqubits = max(target_ids) + 1

    def _check_symbolmap(self, s):
        """Checks if symbol exists in the given symbol map."""
        if s not in self.map:
            raise_error(ValueError, f"Symbolic Hamiltonian contains symbol {s} "
                                    "which does not exist in the symbol map.")

    @staticmethod
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

    def full_matrices(self):
        """Generator of matrices for each symbolic Hamiltonian term.

        Returns:
            Matrices of shape ``(2 ** nqubits, 2 ** nqubits)`` for each term in
            the given symbolic form. Here ``nqubits`` is the total number of
            qubits that the Hamiltonian acts on.
        """
        for targets, matrices in self.terms.items():
            matrix_list = self.nqubits * [self.matrices.I]
            n = len(targets)
            total = 0
            for i in range(0, len(matrices), n + 1):
                for t, m in zip(targets, matrices[i + 1: i + n + 1]):
                    matrix_list[t] = m
                total += matrices[i] * self.multikron(matrix_list)
            yield total

    def partial_matrices(self):
        """Generator of matrices for each symbolic Hamiltonian term.

        Returns:
            Matrices of shape ``(2 ** ntargets, 2 ** ntargets)`` for each term
            in the given symbolic form. Here ``ntargets`` is the number of
            qubits that the corresponding term acts on.
        """
        for targets, matrices in self.terms.items():
            n = len(targets)
            matrix = 0
            for i in range(0, len(matrices), n + 1):
                matrix += matrices[i] * self.multikron(
                    matrices[i + 1: i + n + 1])
            yield targets, matrix

    def dense_matrix(self):
        """Creates the full Hamiltonian matrix.

        Useful for creating :class:`qibo.abstractions.hamiltonians.Hamiltonian`
        object equivalent to the given symbolic Hamiltonian.

        Returns:
            Full Hamiltonian matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
        """
        matrix = sum(self.full_matrices())
        eye = K.np.eye(matrix.shape[0], dtype=matrix.dtype)
        return matrix + self.constant * eye

    def reduce_pairs(self, pair_sets, pair_map, free_targets):
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
        return self.reduce_pairs(pair_sets, pair_map, free_targets)

    def merge_one_qubit(self, terms):
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
        pair_map = self.reduce_pairs(pair_sets, dict(), free_targets)
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
                matrices = self.terms[pair]
                pair = (pair[1], pair[0])
                matrix = 0
                for i in range(0, len(matrices), 3):
                    matrix += matrices[i] * self.multikron(matrices[i + 2: i:-1])
            eye = K.np.eye(2, dtype=matrix.dtype)
            merged[pair] = K.np.kron(one_qubit[target], eye) + matrix
        merged.update(two_qubit)
        return merged

    def trotter_terms(self):
        """Creates a dictionary of targets and matrices.

        Useful for creating :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian`
        objects.

        Returns:
            terms (dict): Dictionary that maps tuples of targets to the matrix
                          that acts on these on targets.
            constant (float): The overall constant term of the Hamiltonian.
        """
        terms = {t: m for t, m in self.partial_matrices()}
        if tuple() in terms:
            constant = terms.pop(tuple()) + self.constant
        else:
            constant = self.constant
        if set(len(t) for t in terms.keys()) == {1, 2}:
            terms = self.merge_one_qubit(terms)
        return terms, constant


class TrotterHamiltonian(hamiltonians.TrotterHamiltonian):
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian`."""

    def __init__(self, *parts, ground_state=None):
        self.K = K
        super().__init__(*parts, ground_state=ground_state)

    @staticmethod
    def construct_terms(terms):
        """Helper method for `from_symbolic`.

        Constructs the term dictionary by using the same
        :class:`qibo.abstractions.hamiltonians.Hamiltonian` object for terms that
        have equal matrix representation. This is done for efficiency during
        the exponentiation of terms.

        Args:
            terms (dict): Dictionary that maps tuples of targets to the matrix
                          that acts on these on targets.

        Returns:
            terms (dict): Dictionary that maps tuples of targets to the
                          Hamiltonian term that acts on these on targets.
        """
        from qibo.hamiltonians import Hamiltonian
        unique_matrices = []
        hterms = {}
        for targets, matrix in terms.items():
            flag = True
            for m, h in unique_matrices:
                if K.np.array_equal(matrix, m):
                    ham = h
                    flag = False
                    break
            if flag:
                ham = Hamiltonian(len(targets), matrix, numpy=True)
                unique_matrices.append((matrix, ham))
            hterms[targets] = ham
        return hterms

    def make_compatible(self, o):
        if not isinstance(o, self.__class__):
            raise TypeError("Only ``TrotterHamiltonians`` can be made "
                            "compatible but {} was given.".format(type(o)))
        if self.is_compatible(o):
            return o

        normalizer = {}
        for targets in o.targets_map.keys():
            if len(targets) > 1:
                raise_error(NotImplementedError,
                            "Only non-interacting Hamiltonians can be "
                            "transformed using the `make_compatible` "
                            "method but the given Hamiltonian contains "
                            "a {} qubit term.".format(len(targets)))
            normalizer[targets[0]] = 0

        term_matrices = {}
        for targets in self.targets_map.keys():
            mats = []
            for target in targets:
                if target in normalizer:
                    normalizer[target] += 1
                    mats.append(o.targets_map[(target,)].matrix)
                else:
                    mats.append(None)
            term_matrices[targets] = tuple(mats)

        for v in normalizer.values():
            if v == 0:
                raise_error(ValueError, "Given non-interacting Hamiltonian "
                                        "cannot be made compatible.")

        new_terms = {}
        for targets, matrices in term_matrices.items():
            n = len(targets)
            s = K.np.zeros(2 * (2 ** n,), dtype=self.dtype)
            for i, (t, m) in enumerate(zip(targets, matrices)):
                matlist = n * [K.np.eye(2, dtype=self.dtype)]
                if m is not None:
                    matlist[i] = m / normalizer[t]
                    s += SymbolicHamiltonian.multikron(matlist)
            new_terms[targets] = self.term_class(n, s, numpy=True)

        new_parts = [{t: new_terms[t] for t in part.keys()}
                     for part in self.parts]
        return self.__class__(*new_parts, ground_state=o.ground_state_func)

    def calculate_dense_matrix(self):
        if 2 * self.nqubits > len(EINSUM_CHARS): # pragma: no cover
            # case not tested because it only happens in large examples
            raise_error(NotImplementedError, "Not enough einsum characters.")

        matrix = K.np.zeros(2 * self.nqubits * (2,), dtype=self.dtype)
        chars = EINSUM_CHARS[:2 * self.nqubits]
        for targets, term in self:
            tmat = term.matrix.reshape(2 * term.nqubits * (2,))
            n = self.nqubits - len(targets)
            emat = K.np.eye(2 ** n, dtype=self.dtype).reshape(2 * n * (2,))
            gen = lambda x: (chars[i + x] for i in targets)
            tc = "".join(itertools.chain(gen(0), gen(self.nqubits)))
            ec = "".join((c for c in chars if c not in tc))
            matrix += K.np.einsum(f"{tc},{ec}->{chars}", tmat, emat)
        return matrix.reshape(2 * (2 ** self.nqubits,))

    def expectation(self, state, normalize=False):
        return Hamiltonian.expectation(self, state, normalize)

    def __matmul__(self, state):
        if isinstance(state, states.AbstractState):
            state = state.tensor
        if not isinstance(state, K.tensor_types):
            raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                             "implemented.".format(type(state)))
        rank = len(tuple(state.shape))
        if rank != 1:
            raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                    "rank-{} tensor.".format(rank))
        result = self.K.zeros_like(state)
        for gate in self.terms():
            # Create copy of state so that the original is not modified
            statec = self.K.copy(state)
            result += gate(statec)
        return result
