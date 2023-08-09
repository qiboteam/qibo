import itertools
import sympy
from qibo import K, gates
from qibo.config import log, raise_error, EINSUM_CHARS
from qibo.abstractions import hamiltonians, states


class Hamiltonian(hamiltonians.MatrixHamiltonian):
    """Hamiltonian based on a dense or sparse matrix representation.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape ``(2 ** nqubits, 2 ** nqubits)``.
            Sparse matrices based on ``scipy.sparse`` for numpy/qibojit backends
            or on ``tf.sparse`` for the tensorflow backend are also
            supported.
    """

    def __init__(self, nqubits, matrix):
        if not (isinstance(matrix, K.tensor_types) or K.issparse(matrix)):
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
        log.warning("`Hamiltonian.from_symbolic` and the use of symbol maps is "
                    "deprecated. Please use `SymbolicHamiltonian` and Qibo symbols "
                    "to construct Hamiltonians using symbols.")
        return SymbolicHamiltonian(symbolic_hamiltonian, symbol_map)

    def eigenvalues(self, k=6):
        if self._eigenvalues is None:
            self._eigenvalues = self.K.eigvalsh(self.matrix, k)
        return self._eigenvalues

    def eigenvectors(self, k=6):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.K.eigh(self.matrix, k)
        return self._eigenvectors

    def exp(self, a):
        if self._exp.get("a") != a:
            self._exp["a"] = a
            if self._eigenvectors is None or self.K.issparse(self.matrix):
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
        if isinstance(o, K.tensor_types):
            o = complex(o)
        elif not isinstance(o, K.numeric_types):
            raise_error(NotImplementedError, "Hamiltonian multiplication to {} "
                                             "not implemented.".format(type(o)))
        new_matrix = self.matrix * o
        r = self.__class__(self.nqubits, new_matrix)
        if self._eigenvalues is not None:
            if K.qnp.cast(o).real >= 0:
                r._eigenvalues = o * self._eigenvalues
            elif not K.issparse(self.matrix):
                r._eigenvalues = o * self._eigenvalues[::-1]
        if self._eigenvectors is not None:
            if K.qnp.cast(o).real > 0:
                r._eigenvectors = self._eigenvectors
            elif o == 0:
                r._eigenvectors = self.eye(int(self._eigenvectors.shape[0]))
        return r

    def __matmul__(self, o):
        if isinstance(o, self.__class__):
            new_matrix = self.K.dot(self.matrix, o.matrix)
            return self.__class__(self.nqubits, new_matrix)

        if isinstance(o, states.AbstractState):
            o = o.tensor
        if isinstance(o, K.tensor_types):
            rank = len(tuple(o.shape))
            if rank == 1: # vector
                return self.K.dot(self.matrix, o[:, self.K.newaxis])[:, 0]
            elif rank == 2: # matrix
                return self.K.dot(self.matrix, o)
            else:
                raise_error(ValueError, "Cannot multiply Hamiltonian with "
                                        "rank-{} tensor.".format(rank))

        raise_error(NotImplementedError, "Hamiltonian matmul to {} not "
                                         "implemented.".format(type(o)))


class TrotterCircuit:
    """Object that caches the Trotterized evolution circuit.

    This object holds a reference to the circuit models and updates its
    parameters if a different time step ``dt`` is given without recreating
    every gate from scratch.

    Args:
        groups (list): List of :class:`qibo.core.terms.TermGroup` objects that
            correspond to the Trotter groups of terms in the time evolution
            exponential operator.
        dt (float): Time step for the Trotterization.
        nqubits (int): Number of qubits in the system that evolves.
        accelerators (dict): Dictionary with accelerators for distributed
            circuits.
    """

    def __init__(self, groups, dt, nqubits, accelerators):
        from qibo.models import Circuit
        self.gates = {}
        self.dt = dt
        self.circuit = Circuit(nqubits, accelerators=accelerators)
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

    Calculations using symbolic Hamiltonians are either done directly using
    the given ``sympy`` expression as it is (``form``) or by parsing the
    corresponding ``terms`` (which are :class:`qibo.core.terms.SymbolicTerm`
    objects). The latter approach is more computationally costly as it uses
    a ``sympy.expand`` call on the given form before parsing the terms.
    For this reason the ``terms`` are calculated only when needed, for example
    during Trotterization.
    The dense matrix of the symbolic Hamiltonian can be calculated directly
    from ``form`` without requiring ``terms`` calculation (see
    :meth:`qibo.core.hamiltonians.SymbolicHamiltonian.calculate_dense` for details).

    Args:
        form (sympy.Expr): Hamiltonian form as a ``sympy.Expr``. Ideally the
            Hamiltonian should be written using Qibo symbols.
            See :ref:`How to define custom Hamiltonians using symbols? <symbolicham-example>`
            example for more details.
        symbol_map (dict): Dictionary that maps each ``sympy.Symbol`` to a tuple
            of (target qubit, matrix representation). This feature is kept for
            compatibility with older versions where Qibo symbols were not available
            and may be deprecated in the future.
            It is not required if the Hamiltonian is constructed using Qibo symbols.
            The symbol_map can also be used to pass non-quantum operator arguments
            to the symbolic Hamiltonian, such as the parameters in the
            :meth:`qibo.hamiltonians.MaxCut` Hamiltonian.
        ground_state (Callable): Function with no arguments that returns the
            ground state of this Hamiltonian. This is useful in cases where
            the ground state is trivial and is used for initialization,
            for example the easy Hamiltonian in adiabatic evolution,
            however we would like to avoid constructing and diagonalizing the
            full Hamiltonian matrix only to find the ground state.
    """

    def __init__(self, form=None, symbol_map={}, ground_state=None):
        super().__init__(ground_state)
        self._form = None
        self._terms = None
        self.constant = 0 # used only when we perform calculations using ``_terms``
        self._dense = None
        self.symbol_map = symbol_map
        # if a symbol in the given form is not a Qibo symbol it must be
        # included in the ``symbol_map``
        self.trotter_circuit = None
        from qibo.symbols import Symbol
        self._qiboSymbol = Symbol # also used in ``self._get_symbol_matrix``
        if form is not None:
            self.form = form

    @property
    def form(self):
        return self._form

    @form.setter
    def form(self, form):
        # Check that given form is a ``sympy`` expression
        if not isinstance(form, sympy.Expr):
            raise_error(TypeError, "Symbolic Hamiltonian should be a ``sympy`` "
                                   "expression but is {}.".format(type(form)))
        # Calculate number of qubits in the system described by the given
        # Hamiltonian formula
        nqubits = 0
        for symbol in form.free_symbols:
            if isinstance(symbol, self._qiboSymbol):
                q = symbol.target_qubit
            elif isinstance(symbol, sympy.Expr):
                if symbol not in self.symbol_map:
                    raise_error(ValueError, "Symbol {} is not in symbol "
                                            "map.".format(symbol))
                q, matrix = self.symbol_map.get(symbol)
                if not isinstance(matrix, K.tensor_types):
                    # ignore symbols that do not correspond to quantum operators
                    # for example parameters in the MaxCut Hamiltonian
                    q = 0
            if q > nqubits:
                nqubits = q

        self._form = form
        self.nqubits = nqubits + 1

    @property
    def terms(self):
        """List of :class:`qibo.core.terms.HamiltonianTerm` objects of which the Hamiltonian is a sum of."""
        if self._terms is None:
            # Calculate terms based on ``self.form``
            from qibo.core.terms import SymbolicTerm
            form = sympy.expand(self.form)
            terms = []
            for f, c in form.as_coefficients_dict().items():
                term = SymbolicTerm(c, f, self.symbol_map)
                if term.target_qubits:
                    terms.append(term)
                else:
                    self.constant += term.coefficient
            assert self.nqubits == max(q for term in terms for q in term.target_qubits) + 1
            self._terms = terms
        return self._terms

    @terms.setter
    def terms(self, terms):
        self._terms = terms
        self.nqubits = max(q for term in self._terms for q in term.target_qubits) + 1

    def _get_symbol_matrix(self, term):
        """Calculates numerical matrix corresponding to symbolic expression.

        This is partly equivalent to sympy's ``.subs``, which does not work
        in our case as it does not allow us to substitute ``sympy.Symbol``
        with numpy arrays and there are different complication when switching
        to ``sympy.MatrixSymbol``. Here we calculate the full numerical matrix
        given the symbolic expression using recursion.
        Helper method for ``_calculate_dense_from_form``.

        Args:
            term (sympy.Expr): Symbolic expression containing local operators.

        Returns:
            Numerical matrix corresponding to the given expression as a numpy
            array of size ``(2 ** self.nqubits, 2 ** self.nqubits).
        """
        if isinstance(term, sympy.Add):
            # symbolic op for addition
            result = sum(self._get_symbol_matrix(subterm)
                         for subterm in term.as_ordered_terms())

        elif isinstance(term, sympy.Mul):
            # symbolic op for multiplication
            # note that we need to use matrix multiplication even though
            # we use scalar symbols for convenience
            factors = term.as_ordered_factors()
            result = self._get_symbol_matrix(factors[0])
            for subterm in factors[1:]:
                result = result @ self._get_symbol_matrix(subterm)

        elif isinstance(term, sympy.Pow):
            # symbolic op for power
            base, exponent = term.as_base_exp()
            matrix = self._get_symbol_matrix(base)
            # multiply ``base`` matrix ``exponent`` times to itself
            result = matrix
            for _ in range(exponent - 1):
                result = result @ matrix

        elif isinstance(term, sympy.Symbol):
            # if the term is a ``Symbol`` then it corresponds to a quantum
            # operator for which we can construct the full matrix directly
            if isinstance(term, self._qiboSymbol):
                # if we have a Qibo symbol the matrix construction is
                # implemented in :meth:`qibo.core.terms.SymbolicTerm.full_matrix`.
                result = term.full_matrix(self.nqubits)
            else:
                q, matrix = self.symbol_map.get(term)
                if not isinstance(matrix, K.tensor_types):
                    # symbols that do not correspond to quantum operators
                    # for example parameters in the MaxCut Hamiltonian
                    result = complex(matrix) * K.qnp.eye(2 ** self.nqubits)
                else:
                    # if we do not have a Qibo symbol we construct one and use
                    # :meth:`qibo.core.terms.SymbolicTerm.full_matrix`.
                    result = self._qiboSymbol(q, matrix).full_matrix(self.nqubits)

        elif term.is_number:
            # if the term is number we should return in the form of identity
            # matrix because in expressions like `1 + Z`, `1` is not correspond
            # to the float 1 but the identity operator (matrix)
            result = complex(term) * K.qnp.eye(2 ** self.nqubits)

        else:
            raise_error(TypeError, "Cannot calculate matrix for symbolic term "
                                   "of type {}.".format(type(term)))

        return result

    def _calculate_dense_from_form(self):
        """Calculates equivalent :class:`qibo.core.hamiltonians.Hamiltonian` using symbolic form.

        Useful when the term representation is not available.
        """
        matrix = self._get_symbol_matrix(self.form)
        return Hamiltonian(self.nqubits, matrix)

    def _calculate_dense_from_terms(self):
        """Calculates equivalent :class:`qibo.core.hamiltonians.Hamiltonian` using the term representation."""
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

    def calculate_dense(self):
        if self._terms is None:
            # calculate dense matrix directly using the form to avoid the
            # costly ``sympy.expand`` call
            return self._calculate_dense_from_form()
        return self._calculate_dense_from_terms()

    def expectation(self, state, normalize=False):
        return Hamiltonian.expectation(self, state, normalize)

    def __add__(self, o):
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise_error(RuntimeError, "Only hamiltonians with the same "
                                          "number of qubits can be added.")
            new_ham = self.__class__(symbol_map=dict(self.symbol_map))
            if self._form is not None and o._form is not None:
                new_ham.form = self.form + o.form
                new_ham.symbol_map.update(o.symbol_map)
            if self._terms is not None and o._terms is not None:
                new_ham.terms = self.terms + o.terms
                new_ham.constant = self.constant + o.constant
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense + o.dense

        elif isinstance(o, K.numeric_types):
            new_ham = self.__class__(symbol_map=dict(self.symbol_map))
            if self._form is not None:
                new_ham.form = self.form + o
            if self._terms is not None:
                new_ham.terms = self.terms
                new_ham.constant = self.constant + o
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
            new_ham = self.__class__(symbol_map=dict(self.symbol_map))
            if self._form is not None and o._form is not None:
                new_ham.form = self.form - o.form
                new_ham.symbol_map.update(o.symbol_map)
            if self._terms is not None and o._terms is not None:
                new_ham.terms = self.terms + [-1 * x for x in o.terms]
                new_ham.constant = self.constant - o.constant
            if self._dense is not None and o._dense is not None:
                new_ham.dense = self.dense - o.dense

        elif isinstance(o, K.numeric_types):
            new_ham = self.__class__(symbol_map=dict(self.symbol_map))
            if self._form is not None:
                new_ham.form = self.form - o
            if self._terms is not None:
                new_ham.terms = self.terms
                new_ham.constant = self.constant - o
            if self._dense is not None:
                new_ham.dense = self.dense - o

        else:
            raise_error(NotImplementedError, "Hamiltonian subtraction to {} "
                                             "not implemented.".format(type(o)))
        return new_ham

    def __rsub__(self, o):
        if isinstance(o, K.numeric_types):
            new_ham = self.__class__(symbol_map=dict(self.symbol_map))
            if self._form is not None:
                new_ham.form = o - self.form
            if self._terms is not None:
                new_ham.terms = [-1 * x for x in self.terms]
                new_ham.constant = o - self.constant
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
        new_ham = self.__class__(symbol_map=dict(self.symbol_map))
        if self._form is not None:
            new_ham.form = o * self.form
        if self._terms is not None:
            new_ham.terms = [o * x for x in self.terms]
            new_ham.constant = self.constant * o
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
            if self._form is None or o._form is None:
                raise_error(NotImplementedError, "Multiplication of symbolic Hamiltonians "
                                                 "without symbolic form is not implemented.")
            new_form = self.form * o.form
            new_symbol_map = dict(self.symbol_map)
            new_symbol_map.update(o.symbol_map)
            new_ham = self.__class__(new_form, symbol_map=new_symbol_map)
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

    def circuit(self, dt, accelerators=None):
        """Circuit that implements a Trotter step of this Hamiltonian for a given time step ``dt``."""
        if self.trotter_circuit is None:
            from qibo.core.terms import TermGroup
            groups = TermGroup.from_terms(self.terms)
            self.trotter_circuit = TrotterCircuit(groups, dt, self.nqubits,
                                                  accelerators)
        self.trotter_circuit.set(dt)
        return self.trotter_circuit.circuit


class TrotterHamiltonian:
    """"""

    def __init__(self, *parts, ground_state=None):
        raise_error(NotImplementedError,
                    "`TrotterHamiltonian` is substituted by `SymbolicHamiltonian` "
                    "and is no longer supported. Please check the documentation "
                    "of `SymbolicHamiltonian` for more details.")

    @classmethod
    def from_symbolic(cls, symbolic_hamiltonian, symbol_map, ground_state=None):
        return cls()
