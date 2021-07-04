import itertools
from qibo import K, gates
from qibo.config import log, raise_error, EINSUM_CHARS
from qibo.abstractions import hamiltonians, states
from qibo.core.symbolic import multikron


class Hamiltonian(hamiltonians.MatrixHamiltonian):
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.MatrixHamiltonian`.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape ``(2 ** nqubits, 2 ** nqubits)``.
        numpy (bool): If ``True`` the Hamiltonian is created using numpy as the
            calculation backend, otherwise the selected backend is used.
            Default option is ``numpy = False``.
    """
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
        matrix = self.K.cast(matrix)
        super().__init__(nqubits, matrix)

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map, numpy=False):
        """Creates a ``Hamiltonian`` from a symbolic Hamiltonian.

        We refer to the :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
        example for more details.

        Args:
            symbolic_hamiltonian (sympy.Expr): The full Hamiltonian written
                with symbols.
            symbol_map (dict): Dictionary that maps each symbol that appears in
                the Hamiltonian to a pair of (target, matrix).
            numpy (bool): If ``True`` the Hamiltonian is created using numpy as
                the calculation backend, otherwise the selected backend is used.
                Default option is ``numpy = False``.

        Returns:
            A :class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian` object
            that implements the Hamiltonian represented by the given symbolic
            expression.
        """
        # TODO: Remove ``numpy`` feature from Hamiltonians?
        log.warn("`Hamiltonian.from_symbolic` and the use of symbol maps is "
                 "deprecated. Please use `SymbolicHamiltonian` and Qibo symbols "
                 "to construct Hamiltonians using symbols.")
        return SymbolicHamiltonian(symbolic_hamiltonian, symbol_map)

    def eigenvalues(self):
        if self._eigenvalues is None:
            self._eigenvalues = self.K.eigvalsh(self.matrix)
        return self._eigenvalues

    def eigenvectors(self):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.K.eigh(self.matrix)
        return self._eigenvectors

    def exp(self, a):
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
        if isinstance(o, self.__class__):
            new_matrix = self.K.matmul(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)

        if isinstance(o, states.AbstractState):
            o = o.tensor
        if isinstance(o, K.tensor_types):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                print(type(self.matrix))
                print(type(o))
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
        return hamiltonians.MatrixHamiltonian.__new__(cls)

    def __init__(self, nqubits, matrix, numpy=True):
        assert numpy
        self.K = K.qnp
        matrix = K.to_numpy(matrix)
        hamiltonians.MatrixHamiltonian.__init__(self, nqubits, matrix)


class SymbolicHamiltonian(hamiltonians.SymbolicHamiltonian):
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian`.

    Args:
        form (sympy.Expr): Hamiltonian form as a ``sympy.Expr``. The Hamiltonian
            should be created using Qibo symbols.
            See :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
            example for more details.
        symbol_map (dict): Dictionary that maps each ``sympy.Symbol`` to the
            corresponding target qubit and matrix representation. This feature
            is kept for compatibility with older versions where Qibo symbols
            were not available and will be deprecated in the future.
            It is not required if the Hamiltonian is constructed using Qibo symbols.
    """

    def __init__(self, form, symbol_map=None):
        super().__init__()
        import sympy
        from qibo.symbols import SymbolicTerm
        if not issubclass(form.__class__, sympy.Expr):
            raise_error(TypeError, "Symbolic Hamiltonian should be a ``sympy`` "
                                   "expression but is {}.".format(type(form)))
        self.form = sympy.expand(form)
        termsdict = self.form.as_coefficients_dict()
        self.terms = [SymbolicTerm(c, f, symbol_map) for f, c in termsdict.items()]
        self.nqubits = max(factor.target_qubit for term in self.terms for factor in term) + 1
        self._dense = None

    def calculate_dense(self):
        matrix = 0
        for term in self.terms:
            kronlist = self.nqubits * [K.qnp.eye(2)]
            for factor in term:
                q = factor.target_qubit
                kronlist[q] = kronlist[q] @ factor.matrix
            matrix += term.coefficient * multikron(kronlist)
        return Hamiltonian(self.nqubits, matrix)

    def expectation(self, state, normalize=False):
        return Hamiltonian.expectation(self, state, normalize)

    def __add__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be added.")
            new_ham = self.__class__(self.form + o.form)
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense + o.dense
        elif isinstance(o, K.numeric_types):
            new_ham = self.__class__(self.form + o)
            if self._dense is not None:
                new_ham.dense = self.dense + o
        else:
            raise_error(NotImplementedError, "SymbolicHamiltonian addition to {} not "
                                             "implemented.".format(type(o)))
        return new_ham

    def __sub__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be subtracted.")
            new_ham = self.__class__(self.form - o.form)
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense - o.dense
        elif isinstance(o, K.numeric_types):
            new_ham = self.__class__(self.form - o)
            if self._dense is not None:
                new_ham.dense = self.dense - o
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))
        return new_ham

    def __rsub__(self, o):
        if isinstance(o, K.numeric_types):
            new_ham = self.__class__(o - self.form)
            if self._dense is not None:
                new_ham.dense = o - self.dense
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))
        return new_ham

    def __mul__(self, o):
        if not (isinstance(o, K.numeric_types) or isinstance(o, K.tensor_types)):
            raise_error(NotImplementedError, "Hamiltonian multiplication to {} "
                                             "not implemented.".format(type(o)))
        new_form = o * self.form
        new_ham = self.__class__(new_form)
        if self._dense is not None:
            new_ham.dense = o * self._dense
        return new_ham

    def apply_gates(self, state, density_matrix=False):
        total = 0
        for term in self.terms:
            temp_state = K.copy(state)
            for factor in term:
                if density_matrix:
                    factor.gate.density_matrix = True
                    temp_state = factor.gate.density_matrix_half_call(temp_state)
                else:
                    temp_state = factor.gate(temp_state)
            total += term.coefficient * temp_state
        return total

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            new_form = self.form * o.form
            new_ham = self.__class__(new_form)
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense @ o.dense
            return new_ham

        if isinstance(o, states.AbstractState):
            o = o.tensor
        if isinstance(o, K.tensor_types):
            rank = len(tuple(o.shape))
            if rank == 1: # state vector
                return self.apply_gates(o)
            elif rank == 2: # density matrix
                return self.apply_gates(o, density_matrix=True)
            else:
                raise_error(NotImplementedError, "Cannot multiply Hamiltonian with "
                                                 "rank-{} tensor.".format(rank))

        raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                         "implemented.".format(type(o)))


class TrotterHamiltonian(hamiltonians.TrotterHamiltonian):
    """Hamiltonian operator used for Trotterized time evolution.

    The Hamiltonian represented by this class has the form of Eq. (57) in
    `arXiv:1901.05824 <https://arxiv.org/abs/1901.05824>`_.

    Args:
        *parts (dict): Dictionary whose values are
            :class:`qibo.abstractions.hamiltonians.Hamiltonian` objects representing
            the h operators of Eq. (58) in the reference. The keys of the
            dictionary are tuples of qubit ids (int) that represent the targets
            of each h term.
        ground_state (Callable): Optional callable with no arguments that
            returns the ground state of this ``TrotterHamiltonian``. Specifying
            this method is useful if the ``TrotterHamiltonian`` is used as
            the easy Hamiltonian of the adiabatic evolution and its ground
            state is used as the initial condition.

    Example:
        ::

            from qibo import matrices, hamiltonians
            # Create h term for critical TFIM Hamiltonian
            matrix = -np.kron(matrices.Z, matrices.Z) - np.kron(matrices.X, matrices.I)
            term = hamiltonians.Hamiltonian(2, matrix)
            # TFIM with periodic boundary conditions is translationally
            # invariant and therefore the same term can be used for all qubits
            # Create even and odd Hamiltonian parts (Eq. (43) in arXiv:1901.05824)
            even_part = {(0, 1): term, (2, 3): term}
            odd_part = {(1, 2): term, (3, 0): term}
            # Create a ``TrotterHamiltonian`` object using these parts
            h = hamiltonians.TrotterHamiltonian(even_part, odd_part)

            # Alternatively the precoded TFIM model may be used with the
            # ``trotter`` flag set to ``True``
            h = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    """

    def __init__(self, *parts, ground_state=None):
        super().__init__()
        self.K = K
        self.dtype = None
        self.term_class = None
        # maps each distinct ``Hamiltonian`` term to the set of gates that
        # are associated with it
        self.expgate_sets = {}
        self.term_sets = {}
        self.targets_map = {}
        for part in parts:
            if not isinstance(part, dict):
                raise_error(TypeError, "``TrotterHamiltonian`` part should be "
                                       "dictionary but is {}."
                                       "".format(type(part)))
            for targets, term in part.items():
                if not issubclass(type(term), hamiltonians.AbstractHamiltonian):
                    raise_error(TypeError, "Invalid term type {}."
                                           "".format(type(term)))
                if len(targets) != term.nqubits:
                    raise_error(ValueError, "Term targets {} but supports {} "
                                            "qubits."
                                            "".format(targets, term.nqubits))

                if targets in self.targets_map:
                    raise_error(ValueError, "Targets {} are given in more than "
                                            "one term.".format(targets))
                self.targets_map[targets] = term
                if term in self.term_sets:
                    self.term_sets[term].add(targets)
                else:
                    self.term_sets[term] = {targets}
                    self.expgate_sets[term] = set()

                if self.term_class is None:
                    self.term_class = term.__class__
                elif term.__class__ != self.term_class:
                    raise_error(TypeError,
                                "Terms of different types {} and {} were "
                                "given.".format(term, self.term_class))
                if self.dtype is None:
                    self.dtype = term.matrix.dtype
                elif term.matrix.dtype != self.dtype: # pragma: no cover
                    raise_error(TypeError,
                                "Terms of different types {} and {} were "
                                "given.".format(term.matrix.dtype, self.dtype))
        self.parts = parts
        self.nqubits = len({t for targets in self.targets_map.keys()
                            for t in targets})
        self.nterms = sum(len(part) for part in self.parts)
        # Function that creates the ground state of this Hamiltonian
        # can be ``None``
        self.ground_state_func = ground_state
        # Circuit that implements on Trotter dt step
        self._circuit = None
        # List of gates that implement each Hamiltonian term. Useful for
        # calculating expectation
        self._terms = None
        # Define dense Hamiltonian attributes
        self._matrix = None
        self._dense = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    @classmethod
    def from_dictionary(cls, terms, ground_state=None):
        parts = cls._split_terms(terms)
        return cls(*parts, ground_state=ground_state)

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map, ground_state=None):
        """Creates a ``TrotterHamiltonian`` from a symbolic Hamiltonian.

        We refer to the :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
        example for more details.

        Args:
            symbolic_hamiltonian (sympy.Expr): The full Hamiltonian written
                with symbols.
            symbol_map (dict): Dictionary that maps each symbol that appears in
                the Hamiltonian to a pair of (target, matrix).
            ground_state (Callable): Optional callable with no arguments that
                returns the ground state of this ``TrotterHamiltonian``.
                See :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian` for more
                details.

        Returns:
            A :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian` object that
            implements the given symbolic Hamiltonian.
        """
        terms, constant = cls.symbolic_terms(symbolic_hamiltonian, symbol_map)
        terms = cls.construct_terms(terms)
        return cls.from_dictionary(terms, ground_state=ground_state) + constant

    @staticmethod
    def symbolic_terms(symbolic_hamiltonian, symbol_map):
        from qibo.core.symbolic import parse_symbolic, merge_one_qubit
        sterms = parse_symbolic(symbolic_hamiltonian, symbol_map)
        constant = sterms.pop(tuple())
        # Construct dictionary of terms with matrices of shape
        # ``(2 ** ntargets, 2 ** ntargets)`` for each term in the given
        # symbolic form. Here ``ntargets`` is the number of
        # qubits that the corresponding term acts on.
        terms = {}
        for targets, matrices in sterms.items():
            n = len(targets)
            matrix = 0
            for i in range(0, len(matrices), n + 1):
                matrix += matrices[i] * multikron(matrices[i + 1: i + n + 1])
            terms[targets] = matrix
        if set(len(t) for t in terms.keys()) == {1, 2}:
            terms = merge_one_qubit(terms, sterms)
        return terms, constant

    @staticmethod
    def _split_terms(terms):
        """Splits a dictionary of terms to multiple parts.

        Each qubit should not appear in more that one terms in each
        part to ensure commutation relations in the definition of
        :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian`.

        Args:
            terms (dict): Dictionary that maps tuples of targets to the matrix
                          that acts on these on targets.

        Returns:
            List of dictionary parts to be used for the creation of a
            ``TrotterHamiltonian``. The parts are such that no qubit appears
            twice in each part.
        """
        groups, singles = [set()], [set()]
        for targets in terms.keys():
            flag = True
            t = set(targets)
            for g, s in zip(groups, singles):
                if not t & s:
                    s |= t
                    g.add(targets)
                    flag = False
                    break
            if flag:
                groups.append({targets})
                singles.append(t)
        return [{k: terms[k] for k in g} for g in groups]

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

    def calculate_dense(self):
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
        matrix = matrix.reshape(2 * (2 ** self.nqubits,))
        return Hamiltonian(self.nqubits, matrix)

    def expectation(self, state, normalize=False):
        return Hamiltonian.expectation(self, state, normalize)

    def ground_state(self):
        """Computes the ground state of the Hamiltonian.

        If this method is needed it should be implemented efficiently for the
        particular Hamiltonian by passing the ``ground_state`` argument during
        initialization. If this argument is not passed then this method will
        diagonalize the full (dense) Hamiltonian matrix which is computationally
        and memory intensive.
        """
        if self.ground_state_func is None:
            log.info("Ground state function not available for ``TrotterHamiltonian``."
                     "Using dense Hamiltonian eigenvectors.")
            return self.eigenvectors()[:, 0]
        return self.ground_state_func()

    def is_compatible(self, o):
        """Checks if a ``TrotterHamiltonian`` has the same part structure.

        By part structure we mean that the target keys of the dictionaries
        contained in the ``self.parts`` list are the same for both Hamiltonians.
        Two :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian` objects can be
        added only when they are compatible (have the same part structure).
        When using Trotter decomposition to simulate adiabatic evolution then
        ``h0`` and ``h1`` should be compatible.

        Args:
            o: The Hamiltonian to check compatibility with.

        Returns:
            ``True`` if ``o`` has the same structure as ``self`` otherwise
            ``False``.
        """
        if isinstance(o, self.__class__):
            if len(self.parts) != len(o.parts):
                return False
            for part1, part2 in zip(self.parts, o.parts):
                if set(part1.keys()) != set(part2.keys()):
                    return False
            return True
        return False

    def make_compatible(self, o):
        """Makes given ``TrotterHamiltonian`` compatible to the current one.

        See :meth:`qibo.abstractions.hamiltonians.TrotterHamiltonian.is_compatible` for
        more details on how compatibility is defined in this context.
        The current method will be used automatically by
        :class:`qibo.evolution.AdiabaticEvolution` to make the ``h0`` and ``h1``
        Hamiltonians compatible if they are not.
        We note that in some applications making the Hamiltonians compatible
        manually instead of relying in this method may take better advantage of
        caching and lead to better execution performance.

        Args:
            o: The ``TrotterHamiltonian`` to make compatible to the current.
               Should be non-interacting (contain only one-qubit terms).

        Returns:
            A new :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian` object
            that is equivalent to ``o`` but has the same part structure as
            ``self``.
        """
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
                    s += multikron(matlist)
            new_terms[targets] = self.term_class(n, s, numpy=True)

        new_parts = [{t: new_terms[t] for t in part.keys()}
                     for part in self.parts]
        return self.__class__(*new_parts, ground_state=o.ground_state_func)

    def __iter__(self):
        """Helper iteration method to loop over the Hamiltonian terms."""
        for part in self.parts:
            for targets, term in part.items():
                yield targets, term

    def _create_circuit(self, dt, accelerators=None, memory_device="/CPU:0"):
        """Creates circuit that implements the Trotterized evolution."""
        from qibo.models import Circuit
        self._circuit = Circuit(self.nqubits, accelerators=accelerators,
                                memory_device=memory_device)
        self._circuit.check_initial_state_shape = False
        self._circuit.dt = None
        for part in itertools.chain(self.parts, self.parts[::-1]):
            for targets, term in part.items():
                gate = gates.Unitary(term.exp(dt / 2.0), *targets)
                self.expgate_sets[term].add(gate)
                self._circuit.add(gate)

    def terms(self):
        if self._terms is None:
            self._terms = [gates.Unitary(term.matrix, *targets)
                           for targets, term in self]
        return self._terms

    def circuit(self, dt, accelerators=None, memory_device="/CPU:0"):
        """Circuit implementing second order Trotter time step.

        Args:
            dt (float): Time step to use for Trotterization.

        Returns:
            :class:`qibo.abstractions.circuit.AbstractCircuit` that implements a single
            time step of the second order Trotterized evolution.
        """
        if self._circuit is None:
            self._create_circuit(dt, accelerators, memory_device)
        elif dt != self._circuit.dt:
            self._circuit.dt = dt
            self._circuit.set_parameters({
                gate: term.exp(dt / 2.0)
                for term, expgates in self.expgate_sets.items()
                for gate in expgates})
        return self._circuit

    def _scalar_op(self, op, o):
        """Helper method for implementing operations with scalars.

        Args:
            op (str): String that defines the operation, such as '__add__' or
                '__mul__'.
            o: Scalar to perform operation for.
        """
        new_parts = []
        new_terms = {term: getattr(term, op)(o) for term in self.expgate_sets.keys()}
        new_parts = ({targets: new_terms[term]
                      for targets, term in part.items()}
                     for part in self.parts)
        new = self.__class__(*new_parts)
        if self._dense is not None:
            new.dense = getattr(self.dense, op)(o)
        if self._circuit is not None:
            new._circuit = self._circuit
            new._circuit.dt = None
            new.expgate_sets = {new_terms[term]: gate_set
                              for term, gate_set in self.expgate_sets.items()}
        return new

    def _hamiltonian_op(self, op, o):
        """Helper method for implementing operations between local Hamiltonians.

        Args:
            op (str): String that defines the operation, such as '__add__'.
            o (:class:`qibo.abstractions.hamiltonians.TrotterHamiltonian`): Other local
                Hamiltonian to perform the operation.
        """
        if len(self.parts) != len(o.parts):
            raise_error(ValueError, "Cannot add local Hamiltonians if their "
                                    "parts are not compatible.")

        new_terms = {}
        def new_parts():
            for part1, part2 in zip(self.parts, o.parts):
                if set(part1.keys()) != set(part2.keys()):
                    raise_error(ValueError, "Cannot add local Hamiltonians "
                                            "if their parts are not "
                                            "compatible.")
                new_part = {}
                for targets in part1.keys():
                    term_tuple = (part1[targets], part2[targets])
                    if term_tuple not in new_terms:
                        new_terms[term_tuple] = getattr(part1[targets], op)(
                            part2[targets])
                    new_part[targets] = new_terms[term_tuple]
                yield new_part

        new = self.__class__(*new_parts())
        if self._circuit is not None:
            new.expgate_sets = {new_term: self.expgate_sets[t1]
                                for (t1, _), new_term in new_terms.items()}
            new._circuit = self._circuit
            new._circuit.dt = None
        return new

    def __add__(self, o):
        if isinstance(o, self.__class__):
            return self._hamiltonian_op("__add__", o)
        else:
            return self._scalar_op("__add__", o / self.nterms)

    def __sub__(self, o):
        if isinstance(o, self.__class__):
            return self._hamiltonian_op("__sub__", o)
        else:
            return self._scalar_op("__sub__", o / self.nterms)

    def __rsub__(self, o):
        return self._scalar_op("__rsub__", o / self.nterms)

    def __mul__(self, o):
        return self._scalar_op("__mul__", o)

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
