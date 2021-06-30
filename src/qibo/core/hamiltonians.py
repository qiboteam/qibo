import itertools
from qibo import K, gates
from qibo.config import log, raise_error, EINSUM_CHARS
from qibo.abstractions import hamiltonians, states


class Hamiltonian(hamiltonians.MatrixHamiltonian):
    """Backend implementation of :class:`qibo.abstractions.hamiltonians.MatrixHamiltonian`.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape ``(2 ** nqubits, 2 ** nqubits)``.
    """

    def __init__(self, nqubits, matrix):
        if not isinstance(matrix, K.tensor_types):
            raise_error(TypeError, "Matrix of invalid type {} given during "
                                   "Hamiltonian initialization"
                                   "".format(type(matrix)))
        self.K = K
        matrix = self.K.cast(matrix)
        super().__init__(nqubits, matrix)

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map):
        """Creates a ``Hamiltonian`` from a symbolic Hamiltonian.

        We refer to the :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
        example for more details.

        Args:
            symbolic_hamiltonian (sympy.Expr): The full Hamiltonian written
                with symbols.
            symbol_map (dict): Dictionary that maps each symbol that appears in
                the Hamiltonian to a pair of (target, matrix).

        Returns:
            A :class:`qibo.abstractions.hamiltonians.SymbolicHamiltonian` object
            that implements the Hamiltonian represented by the given symbolic
            expression.
        """
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
                return self.K.matmul(self.matrix, o[:, self.K.newaxis])[:, 0]
            elif rank == 2: # matrix
                return self.K.matmul(self.matrix, o)
            else:
                raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                        "rank-{} tensor.".format(rank))

        raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                         "implemented.".format(type(o)))


class TrotterCircuit:

    def __init__(self, groups, dt, nqubits, accelerators, memory_device):
        from qibo.models import Circuit
        self.gates = {}
        self.dt = dt
        self.circuit = Circuit(nqubits, accelerators=accelerators, memory_device=memory_device)
        for group in itertools.chain(groups, groups[::-1]):
            gate = group.term.expgate(dt / 2.0)
            self.gates[gate] = group
            self.circuit.add(gate)

    def set(self, dt):
        if self.dt != dt:
            params = {gate: group.term.exp(dt / 2.0) for gate, group in self.gates.items()}
            self.dt = dt
            self.circuit.set_parameters(params)


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

    def __init__(self, form=None, symbol_map=None, ground_state=None):
        super().__init__(ground_state)
        self._dense = None
        self.terms = None # list of :class:`qibo.core.terms.HamiltonianTerm` objects
        self.constant = 0
        self.form = None
        self.trotter_circuit = None
        if form is not None:
            self.set_form(form, symbol_map)

    @classmethod
    def from_terms(cls, terms, ground_state=None):
        """Constructs a symbolic Hamiltonian directly from a list of terms."""
        ham = cls(ground_state=ground_state)
        ham.set_terms(terms)
        return ham

    def set_terms(self, terms):
        self.terms = terms
        self.nqubits = max(q for term in self.terms for q in term.target_qubits) + 1

    def set_form(self, form, symbol_map=None):
        import sympy
        from qibo.core.terms import SymbolicTerm
        if not issubclass(form.__class__, sympy.Expr):
            raise_error(TypeError, "Symbolic Hamiltonian should be a ``sympy`` "
                                   "expression but is {}.".format(type(form)))
        self.form = sympy.expand(form)
        terms = []
        for f, c in self.form.as_coefficients_dict().items():
            term = SymbolicTerm.from_factors(c, f, symbol_map)
            if term.target_qubits:
                terms.append(term)
            else:
                self.constant += term.coefficient
        self.set_terms(terms)

    def calculate_dense(self):
        if 2 * self.nqubits > len(EINSUM_CHARS): # pragma: no cover
            # case not tested because it only happens in large examples
            raise_error(NotImplementedError, "Not enough einsum characters.")

        matrix = 0
        chars = EINSUM_CHARS[:2 * self.nqubits]
        for term in self.terms:
            ntargets = len(term.target_qubits)
            tmat = K.np.reshape(term.matrix, 2 * ntargets * (2,))
            n = self.nqubits - ntargets
            emat = K.np.reshape(K.np.eye(2 ** n, dtype=tmat.dtype), 2 * n * (2,))
            gen = lambda x: (chars[i + x] for i in term.target_qubits)
            tc = "".join(itertools.chain(gen(0), gen(self.nqubits)))
            ec = "".join((c for c in chars if c not in tc))
            matrix += K.np.einsum(f"{tc},{ec}->{chars}", tmat, emat)
        matrix = K.np.reshape(matrix, 2 * (2 ** self.nqubits,))
        return Hamiltonian(self.nqubits, matrix) + self.constant

    def expectation(self, state, normalize=False):
        return Hamiltonian.expectation(self, state, normalize)

    def __add__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be added.")
            new_ham = self.__class__.from_terms(self.terms + o.terms)
            if self.form is not None and o.form is not None:
                new_ham.form = self.form + o.form
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense + o.dense

        elif isinstance(o, K.numeric_types):
            new_ham = self.__class__.from_terms(self.terms)
            new_ham.constant = self.constant + o
            if self.form is not None:
                new_ham.form = self.form + o
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
            new_terms = self.terms + [-1 * x for x in o.terms]
            new_ham = self.__class__.from_terms(new_terms)
            if self.form is not None and o.form is not None:
                new_ham.form = self.form - o.form
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense - o.dense

        elif isinstance(o, K.numeric_types):
            new_ham = self.__class__.from_terms(self.terms)
            new_ham.constant = self.constant - o
            if self.form is not None:
                new_ham.form = self.form - o
            if self._dense is not None:
                new_ham.dense = self.dense - o

        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))
        return new_ham

    def __rsub__(self, o):
        if isinstance(o, K.numeric_types):
            new_ham = self.__class__.from_terms([-1 * x for x in self.terms])
            new_ham.constant = o - self.constant
            if self.form is not None:
                new_ham.form = o - self.form
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
        o = complex(o)
        new_ham = self.__class__.from_terms([o * x for x in self.terms])
        new_ham.constant = self.constant * o
        if self.form is not None:
            new_ham.form = o * self.form
        if self._dense is not None:
            new_ham.dense = o * self._dense
        return new_ham

    def apply_gates(self, state, density_matrix=False):
        """Applies gates corresponding to the Hamiltonian terms to a given state.

        Helper method for ``__matmul__``.
        """
        total = 0
        for term in self.terms:
            total += term(K.copy(state), density_matrix)
        if self.constant:
            total += self.constant * state
        return total

    def __matmul__(self, o):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(o, self.__class__):
            if self.form is None or o.form is None:
                raise_error(NotImplementedError, "Multiplication of symbolic Hamiltonians "
                                                 "without symbolic form is not implemented.")
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

    def circuit(self, dt, accelerators=None, memory_device="/CPU:0"):
        """Circuit that implements a Trotter step of this Hamiltonian for a given time step ``dt``."""
        if self.trotter_circuit is None:
            from qibo.core.terms import TermGroup
            groups = TermGroup.from_terms(self.terms)
            self.trotter_circuit = TrotterCircuit(groups, dt, self.nqubits,
                                                  accelerators, memory_device)
        self.trotter_circuit.set(dt)
        return self.trotter_circuit.circuit
