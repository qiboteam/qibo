import itertools
import numpy as np
from qibo import gates
from qibo.config import log, raise_error


class Hamiltonian:
    """Abstract Hamiltonian operator using full matrix representation.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape
            ``(2 ** nqubits, 2 ** nqubits)``.
        numpy (bool): If ``True`` the Hamiltonian is created using numpy as the
            calculation backend, otherwise TensorFlow is used.
            Default option is ``numpy = False``.
    """
    NUMERIC_TYPES = None
    ARRAY_TYPES = None
    K = None # calculation backend (numpy or TensorFlow)

    def __init__(self, nqubits, matrix, numpy=False):
        if not isinstance(nqubits, int):
            raise_error(RuntimeError, "nqubits must be an integer but is "
                                            "{}.".format(type(nqubits)))
        if nqubits < 1:
            raise_error(ValueError, "nqubits must be a positive integer but is "
                                    "{}".format(nqubits))
        shape = tuple(matrix.shape)
        if shape != 2 * (2 ** nqubits,):
            raise_error(ValueError, "The Hamiltonian is defined for {} qubits "
                                    "while the given matrix has shape {}."
                                    "".format(nqubits, shape))

        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map, numpy=False):  # pragma: no cover
        """Creates a ``Hamiltonian`` from a symbolic Hamiltonian.

        We refer to the :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
        example for more details.

        Args:
            symbolic_hamiltonian (sympy.Expr): The full Hamiltonian written
                with symbols.
            symbol_map (dict): Dictionary that maps each symbol that appears in
                the Hamiltonian to a pair of (target, matrix).
            numpy (bool): If ``True`` the Hamiltonian is created using numpy as
                the calculation backend, otherwise TensorFlow is used.
                Default option is ``numpy = False``.

        Returns:
            A :class:`qibo.base.hamiltonians.Hamiltonian` object that
            implements the given symbolic Hamiltonian.
        """
        # this method is defined for docs only.
        # It is properly implemented in `qibo.hamiltonians.Hamiltonian`.
        raise_error(NotImplementedError)

    def _calculate_exp(self, a): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    def eigenvalues(self):
        """Computes the eigenvalues for the Hamiltonian."""
        if self._eigenvalues is None:
            self._eigenvalues = self.K.linalg.eigvalsh(self.matrix)
        return self._eigenvalues

    def eigenvectors(self):
        """Computes a tensor with the eigenvectors for the Hamiltonian."""
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.K.linalg.eigh(self.matrix)
        return self._eigenvectors

    def ground_state(self):
        """Computes the ground state of the Hamiltonian.

        Uses the ``eigenvectors`` method and returns the lowest energy
        eigenvector.
        """
        return self.eigenvectors()[:, 0]

    def exp(self, a):
        """Computes a tensor corresponding to exp(-1j * a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        if self._exp.get("a") != a:
            self._exp["a"] = a
            self._exp["result"] = self._calculate_exp(a) # pylint: disable=E1111
        return self._exp.get("result")

    def expectation(self, state, normalize=False): # pragma: no cover
        """Computes the real expectation value for a given state.

        Args:
            state (array): the expectation state.
            normalize (bool): If ``True`` the expectation value is divided
                with the state's norm squared.

        Returns:
            Real number corresponding to the expectation value.
        """
        # abstract method
        raise_error(NotImplementedError)

    def _eye(self, n=None):
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
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, self.matrix + o * self._eye())
        else:
            raise_error(NotImplementedError, "Hamiltonian addition to {} not "
                                             "implemented.".format(type(o)))

    def __radd__(self, o):
        """Right operator addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be subtracted.")
            new_matrix = self.matrix - o.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, self.matrix - o * self._eye())
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))

    def __rsub__(self, o):
        """Right subtraction operator."""
        if isinstance(o, self.__class__): # pragma: no cover
            # impractical case because it will be handled by `__sub__`
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be added.")
            new_matrix = o.matrix - self.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, o * self._eye() - self.matrix)
        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))

    def _real(self, o):
        """Calculates real part of number or tensor."""
        return o.real

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, self.NUMERIC_TYPES) or isinstance(o, self.ARRAY_TYPES):
            new_matrix = self.matrix * o
            r = self.__class__(self.nqubits, new_matrix)
            if self._eigenvalues is not None:
                if self._real(o) >= 0:
                    r._eigenvalues = o * self._eigenvalues
                else:
                    r._eigenvalues = o * self._eigenvalues[::-1]
            if self._eigenvectors is not None:
                if self._real(o) > 0:
                    r._eigenvectors = self._eigenvectors
                elif o == 0:
                    r._eigenvectors = self._eye(int(self._eigenvectors.shape[0]))
            return r
        else:
            raise_error(NotImplementedError, "Hamiltonian multiplication to {} "
                                             "not implemented.".format(type(o)))

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            new_matrix = self.K.matmul(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.ARRAY_TYPES):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return self.K.matmul(self.matrix, o[:, self.K.newaxis])[:, 0]
            elif rank == 2: # matrix
                return self.K.matmul(self.matrix, o)
            else:
                raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                        "rank-{} tensor.".format(rank))
        else:
            raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                             "implemented.".format(type(o)))


class _SymbolicHamiltonian:
    """Parses symbolic Hamiltonians defined using ``sympy``.

    This class should not be used by users.
    It is used internally to help creating
    :class:`qibo.base.hamiltonians.Hamiltonian` and
    :class:`qibo.base.hamiltonians.TrotterHamiltonian` objects for Hamiltonians
    defined using symbols. For more information we refer to the
    :meth:`qibo.base.hamiltonians.Hamiltonian.from_symbolic`
    and :meth:`qibo.base.hamiltonians.TrotterHamiltonian.from_symbolic` methods.

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
                    targets.append(self.map[factor][0])
                    matrices.append(self.map[factor][1])
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
    def _multikron(matrix_list):
        """Calculates Kronecker product of a list of matrices.

        Args:
            matrices (list): List of matrices as ``np.ndarray``s.

        Returns:
            ``np.ndarray`` of the Kronecker product of all ``matrices``.
        """
        h = 1
        for m in matrix_list:
            h = np.kron(h, m)
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
                total += matrices[i] * self._multikron(matrix_list)
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
                matrix += matrices[i] * self._multikron(
                  matrices[i + 1: i + n + 1])
            yield targets, matrix

    def dense_matrix(self):
        """Creates the full Hamiltonian matrix.

        Useful for creating :class:`qibo.base.hamiltonians.Hamiltonian`
        object equivalent to the given symbolic Hamiltonian.

        Returns:
            Full Hamiltonian matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
        """
        matrix = sum(self.full_matrices())
        eye = np.eye(matrix.shape[0], dtype=matrix.dtype)
        return matrix + self.constant * eye

    def _merge_one_qubit(self, terms):
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
        # Split terms to one and two-qubit
        # Keep track of the first target in two-qubit terms
        one_qubit, two_qubit, first_targets = dict(), dict(), dict()
        for targets, matrix in terms.items():
            assert len(targets) in {1, 2}
            if len(targets) == 1:
                one_qubit[targets] = matrix
            else:
                t1, t2 = targets
                if t1 in first_targets and t2 not in first_targets:
                    # swap target order
                    # the matrix has to be recalculated as well!
                    t1, t2 = t2, t1
                    c, m1, m2 = self.terms[targets]
                    matrix = c * np.kron(m2, m1)
                two_qubit[(t1, t2)] = matrix
                first_targets[t1] = (t1, t2)

        merged = dict(two_qubit)
        for (t,), m in one_qubit.items():
            if t not in first_targets:
                log.info("Aborting merge of one and two-qubit terms during "
                         "TrotterHamiltonian creation because the two-qubit "
                         "terms are not sufficiently many.")
                return terms
            pair = first_targets[t]
            eye = np.eye(2, dtype=m.dtype)
            merged[pair] = np.kron(m, eye) + two_qubit[pair]
        return merged

    def trotter_terms(self):
        """Creates a dictionary of targets and matrices.

        Useful for creating :class:`qibo.base.hamiltonians.TrotterHamiltonian`
        objects.

        Returns:
            terms (dict): Dictionary that maps tuples of targets to the matrix
                          that acts on these on targets.
            constant (float): The overall constant term of the Hamiltonian.
        """
        terms = {t: m for t, m in self.partial_matrices()}
        if set(len(t) for t in terms.keys()) == {1, 2}:
            terms = self._merge_one_qubit(terms)
        return terms, self.constant


class TrotterHamiltonian(Hamiltonian):
    """Hamiltonian operator used for Trotterized time evolution.

    The Hamiltonian represented by this class has the form of Eq. (57) in
    `arXiv:1901.05824 <https://arxiv.org/abs/1901.05824>`_.

    Args:
        *parts (dict): Dictionary whose values are
            :class:`qibo.base.hamiltonians.Hamiltonian` objects representing
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
        self.dtype = None
        # maps each distinct ``Hamiltonian`` term to the set of gates that
        # are associated with it
        self.expgate_sets = {}
        targets_set = set()
        for part in parts:
            if not isinstance(part, dict):
                raise_error(TypeError, "``TrotterHamiltonian`` part should be "
                                       "dictionary but is {}."
                                       "".format(type(part)))
            for targets, term in part.items():
                if not issubclass(type(term), Hamiltonian):
                    raise_error(TypeError, "Invalid term type {}."
                                           "".format(type(term)))
                if len(targets) != term.nqubits:
                    raise_error(ValueError, "Term targets {} but supports {} "
                                            "qubits."
                                            "".format(targets, term.nqubits))

                if targets in targets_set:
                    raise_error(ValueError, "Targets {} are given in more than "
                                            "one term.".format(targets))
                targets_set.add(targets)
                if term not in self.expgate_sets:
                    self.expgate_sets[term] = set()

                if self.dtype is None:
                    self.dtype = term.matrix.dtype
                elif term.matrix.dtype != self.dtype:
                    raise_error(TypeError, "Terms of different types {} and {} "
                                            "were given.".format(
                                                term.matrix.dtype, self.dtype))
        self.parts = parts
        self.nqubits = len({t for targets in targets_set for t in targets})
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
                See :class:`qibo.base.hamiltonians.TrotterHamiltonian` for more
                details.

        Returns:
            A :class:`qibo.base.hamiltonians.TrotterHamiltonian` object that
            implements the given symbolic Hamiltonian.
        """
        from qibo.hamiltonians import Hamiltonian
        terms, constant = _SymbolicHamiltonian(
          symbolic_hamiltonian, symbol_map).trotter_terms()
        terms = {k: Hamiltonian(len(k), v, numpy=True)
                 for k, v in terms.items()}
        return cls.from_dictionary(terms, ground_state=ground_state) + constant

    @staticmethod
    def _split_terms(terms):
        """Splits a dictionary of terms to multiple parts.

        Each qubit should not appear in more that one terms in each
        part to ensure commutation relations in the definition of
        :class:`qibo.base.hamiltonians.TrotterHamiltonian`.

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

    def is_compatible(self, o):
        """Checks if a ``TrotterHamiltonian`` has the same part structure.

        ``TrotterHamiltonian``s with the same part structure can be add.

        Args:
            o: The second Hamiltonian to check.

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

        Args:
            o: The ``TrotterHamiltonian`` to make compatible to the current.
                Should be non-interacting (contain only one-qubit terms).

        Returns:
            A new :class:`qibo.base.hamiltonians.TrotterHamiltonian` object
            that is equivalent to ``o`` but has the same part structure as
            ``self``.
        """
        if not isinstance(o, self.__class__):
            raise TypeError("Only ``TrotterHamiltonians`` can be made "
                            "compatible but {} was given.".format(type(o)))
        if self.is_compatible(o):
            return o

        oterms = {}
        for part in o.parts:
            for t, m in part.items():
                if len(t) > 1:
                    raise_error(NotImplementedError,
                                "Only non-interacting Hamiltonians can be "
                                "transformed using the ``make_compatible`` "
                                "method.")
                oterms[t[0]] = m

        normalizer = {}
        for part in self.parts:
            for targets in part.keys():
                if targets[0] in normalizer:
                    normalizer[targets[0]] += 1
                else:
                    normalizer[targets[0]] = 1
        if set(normalizer.keys()) != set(oterms.keys()):
            raise_error(ValueError, "Given non-interacting Hamiltonian cannot "
                                    "be made compatible.")

        new_parts = []
        for part in self.parts:
            new_parts.append(dict())
            for targets in part.keys():
                if targets[0] in oterms:
                    n = len(targets)
                    h = oterms[targets[0]]
                    m = h.matrix
                    eye = np.eye(2 ** (n - 1), dtype=m.dtype)
                    m = np.kron(m, eye) / normalizer[targets[0]]
                    new_parts[-1][targets] = h.__class__(n, m, numpy=True)
        return self.__class__(*new_parts, ground_state=o.ground_state_func)

    def _calculate_dense_matrix(self, a): # pragma: no cover
        # abstract method
        raise_error(NotImplementedError)

    @property
    def dense(self):
        """Creates an equivalent Hamiltonian model that holds the full matrix.

        Returns:
            A :class:`qibo.base.hamiltonians.Hamiltonian` object that is
            equivalent to this local Hamiltonian.
        """
        if self._dense is None:
            from qibo import hamiltonians
            matrix = self._calculate_dense_matrix() # pylint: disable=E1111
            self.dense = hamiltonians.Hamiltonian(self.nqubits, matrix)
        return self._dense

    @dense.setter
    def dense(self, hamiltonian):
        self._dense = hamiltonian
        self._eigenvalues = hamiltonian._eigenvalues
        self._eigenvectors = hamiltonian._eigenvectors
        self._exp = hamiltonian._exp

    @property
    def matrix(self):
        return self.dense.matrix

    def eigenvalues(self):
        """Computes the eigenvalues for the Hamiltonian."""
        return self.dense.eigenvalues()

    def eigenvectors(self):
        """Computes a tensor with the eigenvectors for the Hamiltonian."""
        return self.dense.eigenvectors()

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

    def exp(self, a):
        """Computes a tensor corresponding to exp(-1j * a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        return self.dense.exp(a)

    def expectation(self, state, normalize=False): # pragma: no cover
        """Computes the real expectation value for a given state.

        Args:
            state (array): the expectation state.
            normalize (bool): If ``True`` the expectation value is divided
                with the state's norm squared.

        Returns:
            Real number corresponding to the expectation value.
        """
        # abstract method
        raise_error(NotImplementedError)

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
            :class:`qibo.base.circuit.BaseCircuit` that implements a single
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
            o (:class:`qibo.base.hamiltonians.TrotterHamiltonian`): Other local
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
        """Add operator."""
        if isinstance(o, self.__class__):
            return self._hamiltonian_op("__add__", o)
        else:
            return self._scalar_op("__add__", o / self.nterms)

    def __radd__(self, o):
        """Right operator addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            return self._hamiltonian_op("__sub__", o)
        else:
            return self._scalar_op("__sub__", o / self.nterms)

    def __rsub__(self, o):
        """Right subtraction operator."""
        return self._scalar_op("__rsub__", o / self.nterms)

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        return self._scalar_op("__mul__", o)

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)

    def __matmul__(self, state): # pragma: no cover
        """Matrix multiplication with state vectors."""
        # abstract method
        raise_error(NotImplementedError)


HAMILTONIAN_TYPES = (Hamiltonian, TrotterHamiltonian)
