import itertools
from qibo import K
from qibo.config import log, raise_error, EINSUM_CHARS
from qibo.abstractions import hamiltonians, states
from qibo.core.symbolic import multikron


class Hamiltonian(hamiltonians.Hamiltonian):
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.Hamiltonian`."""

    def __new__(cls, nqubits, matrix=None, numpy=False):
        if matrix is not None and not isinstance(matrix, K.tensor_types):
            raise_error(TypeError, "Matrix of invalid type {} given during "
                                   "Hamiltonian initialization"
                                   "".format(type(matrix)))
        if numpy:
            return NumpyHamiltonian(nqubits, matrix, numpy=True)
        else:
            return super().__new__(cls)

    def __init__(self, nqubits, matrix=None, numpy=False):
        assert not numpy
        self.K = K
        if matrix is not None:
            matrix = self.K.cast(matrix)
        super().__init__(nqubits, matrix, numpy=numpy)

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map, numpy=False):
        from qibo.core.symbolic import parse_symbolic
        terms = parse_symbolic(symbolic_hamiltonian, symbol_map)
        nqubits = max(set(q for targets in terms.keys() for q in targets)) + 1
        # Add matrices of shape ``(2 ** nqubits, 2 ** nqubits)`` for each term
        # in the given symbolic form. Here ``nqubits`` is the total number of
        # qubits that the Hamiltonian acts on.
        ham = cls(nqubits, matrix=None, numpy=numpy)
        ham.terms = terms
        return ham

    def calculate_dense_matrix(self):
        if self.terms is None:
            raise_error(ValueError, "Cannot construct Hamiltonian matrix "
                                    "because terms are not available.")
        matrix = 0
        constant = self.terms.pop(tuple())
        for targets, matrices in self.terms.items():
            matrix_list = self.nqubits * [K.np.eye(2)]
            n = len(targets)
            for i in range(0, len(matrices), n + 1):
                for t, m in zip(targets, matrices[i + 1: i + n + 1]):
                    matrix_list[t] = m
                matrix += matrices[i] * multikron(matrix_list)
        matrix += constant * K.np.eye(matrix.shape[0])
        self.terms[tuple()] = constant
        return matrix

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

    def term_gates(self):
        """Creates single-qubit unitary gates corresponding to each Hamiltonian term."""
        if self._gates is None:
            from qibo import gates
            self._gates = []
            constant = self.terms.pop(tuple())
            for targets, matrices in self.terms.items():
                n = len(targets)
                for i in range(0, len(matrices), n + 1):
                    self._gates.append([matrices[i]])
                    for t, m in zip(targets, matrices[i + 1: i + n + 1]):
                        self._gates[-1].append(gates.Unitary(m, t))
            self.terms[tuple()] = constant
        return self._gates

    def gate_matmul(self, state):
        """Multiplies the Hamiltonian to state term by term using unitary gates."""
        total = self.terms[tuple()] * state
        for gatelist in self.term_gates():
            temp_state = self.K.copy(state)
            for gate in gatelist[1:]:
                temp_state = gate(temp_state)
            total += gatelist[0] * temp_state
        return total

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
                if self._matrix is None: # TODO: Expand `gate_matmul` for density matrices
                    return self.gate_matmul(o)
                else:
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
        if matrix is not None:
            matrix = self.K.cast(matrix)
        hamiltonians.Hamiltonian.__init__(self, nqubits, matrix, numpy)


class TrotterHamiltonian(hamiltonians.TrotterHamiltonian):
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.TrotterHamiltonian`."""

    def __init__(self, *parts, ground_state=None):
        self.K = K
        super().__init__(*parts, ground_state=ground_state)

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map, ground_state=None):
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
                    s += multikron(matlist)
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
        for gate in self.terms(): # pylint: disable=E1120
            # TODO: Fix pylint here
            # Create copy of state so that the original is not modified
            statec = self.K.copy(state)
            result += gate(statec)
        return result
