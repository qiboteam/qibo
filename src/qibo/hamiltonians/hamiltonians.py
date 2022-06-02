from qibo.config import raise_error, log
from qibo.hamiltonians.abstract import AbstractHamiltonian

class Hamiltonian(AbstractHamiltonian):
    """Hamiltonian based on a dense or sparse matrix representation.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape ``(2 ** nqubits, 2 ** nqubits)``.
            Sparse matrices based on ``scipy.sparse`` for numpy/qibojit backends
            or on ``tf.sparse`` for the tensorflow backend are also
            supported.
    """
    def __init__(self, nqubits, matrix=None, **kwargs):
        super().__init__(**kwargs)
        if not (isinstance(matrix, self.backend.tensor_types) or self.backend.issparse(matrix)):
            raise_error(TypeError, "Matrix of invalid type {} given during "
                                   "Hamiltonian initialization"
                                   "".format(type(matrix)))

        matrix = self.backend.cast(matrix)

        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    @property
    def matrix(self):
        """Returns the full matrix representation.

        Can be a dense ``(2 ** nqubits, 2 ** nqubits)`` array or a sparse
        matrix, depending on how the Hamiltonian was created.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, m):
        shape = tuple(m.shape)
        if shape != 2 * (2 ** self.nqubits,):
            raise_error(ValueError, "The Hamiltonian is defined for {} qubits "
                                    "while the given matrix has shape {}."
                                    "".format(self.nqubits, shape))
        self._matrix = m

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
            self._eigenvalues = self.backend.calculate_eigenvalues(self.matrix, k)
        return self._eigenvalues

    def eigenvectors(self, k=6):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.backend.calculate_eigenvectors(self.matrix, k)
        return self._eigenvectors

    def exp(self, a):
        if self._exp.get("a") != a:
            self._exp["a"] = a
            self._exp["result"] = self.backend.calculate_exp(a, self._eigenvectors,
                                                             self._eigenvalues, self.matrix)
        return self._exp.get("result")

    def expectation(self, state, normalize=False):
        if isinstance(state, self.backend.tensor_types):
            shape = tuple(state.shape)
            if len(shape) == 1: # state vector
                return self.backend.calculate_expectation_state(self.matrix, state, normalize)
            elif len(shape) == 2: # density matrix
                return self.backend.calculate_expectation_density_matrix(self.matrix, state, normalize)
            else:
                raise_error(ValueError, "Cannot calculate Hamiltonian "
                                        "expectation value for state of shape "
                                        "{}.".format(shape))
        else:
            raise_error(TypeError, "Cannot calculate Hamiltonian expectation "
                                   "value for state of type {}."
                                   "".format(type(state)))


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


class SymbolicHamiltonian(AbstractHamiltonian):
    #TODO: update docstring
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
        super().__init__()
        self._dense = None
        self._ground_state = ground_state

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
    def dense(self):
        """Creates the equivalent :class:`qibo.abstractions.hamiltonians.MatrixHamiltonian`."""
        if self._dense is None:
            log.warning("Calculating the dense form of a symbolic Hamiltonian. "
                        "This operation is memory inefficient.")
            self.dense = self.calculate_dense()
        return self._dense

    @dense.setter
    def dense(self, hamiltonian):
        assert isinstance(hamiltonian, Hamiltonian)
        self._dense = hamiltonian
        self._eigenvalues = hamiltonian._eigenvalues
        self._eigenvectors = hamiltonian._eigenvectors
        self._exp = hamiltonian._exp

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

    @property
    def matrix(self):
        """Returns the full ``(2 ** nqubits, 2 ** nqubits)`` matrix representation."""
        return self.dense.matrix

    def eigenvalues(self, k=6):
        return self.dense.eigenvalues(k)

    def eigenvectors(self, k=6):
        return self.dense.eigenvectors(k)

    def ground_state(self):
        if self._ground_state is None:
            log.warning("Ground state for this Hamiltonian was not given.")
            return self.eigenvectors()[:, 0]
        return self._ground_state()

    def exp(self, a):
        return self.dense.exp(a)