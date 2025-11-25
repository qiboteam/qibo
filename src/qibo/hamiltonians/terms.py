from functools import cached_property, reduce
from typing import Optional

import sympy
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.symbols import I, X, Y, Z

# For mapping between qibo symbols and sympy Pauli objects
PAULI_MAPPING = {X: Pauli(1), Y: Pauli(2), Z: Pauli(3)}
INVERSE_PAULI_MAPPING = {_v: _k for _k, _v in PAULI_MAPPING.items()}


class HamiltonianTerm:
    """Term of a :class:`qibo.hamiltonians.hamiltonians.SymbolicHamiltonian`.

    Symbolic Hamiltonians are represented by a list of
    :class:`qibo.hamiltonians.terms.HamiltonianTerm` objects storred in the
    ``SymbolicHamiltonian.terms`` attribute. The mathematical expression of
    the Hamiltonian is the sum of these terms.

    Args:
        matrix (np.ndarray): Full matrix corresponding to the term representation
            in the computational basis. Has size (2^n, 2^n) where n is the
            number of target qubits of this term.
        q (list): List of target qubit ids.
    """

    def __init__(self, matrix, *q, backend: Optional[Backend] = None):
        self.backend = _check_backend(backend)
        for qi in q:
            if qi < 0:
                raise_error(
                    ValueError,
                    f"Invalid qubit id {qi} < 0 was given in Hamiltonian term.",
                )
        if not isinstance(matrix, self.backend.tensor_types):
            raise_error(TypeError, f"Invalid type {type(matrix)} of symbol matrix.")
        dim = int(matrix.shape[0])
        if 2 ** len(q) != dim:
            raise_error(
                ValueError,
                f"Matrix dimension {dim} given in Hamiltonian "
                + "term is not compatible with the number "
                + f"of target qubits {len(q)}.",
            )
        self.target_qubits = tuple(q)
        self._gate = None
        self.hamiltonian = None
        self._matrix = self.backend.cast(matrix)

    @property
    def matrix(self):
        """Matrix representation of the term."""
        return self._matrix

    @property
    def gate(self):
        """:class:`qibo.gates.gates.Unitary` gate that implements the action of the term on states."""
        if self._gate is None:
            self._gate = gates.Unitary(self.matrix, *self.target_qubits)
        return self._gate

    def exp(self, x):
        """Matrix exponentiation of the term."""
        return self.backend.calculate_matrix_exp(self.matrix, phase=-1j * x)

    def expgate(self, x):
        """:class:`qibo.gates.gates.Unitary` gate implementing the action of exp(term) on states."""
        return gates.Unitary(self.exp(x), *self.target_qubits)

    def merge(self, term):
        """Creates a new term by merging the given term to the current one.

        The resulting term corresponds to the sum of the two original terms.
        The target qubits of the given term should be a subset of the target
        qubits of the current term.
        """
        if not set(term.target_qubits).issubset(set(self.target_qubits)):
            raise_error(
                ValueError,
                "Cannot merge HamiltonianTerm acting on "
                + f"qubits {term.target_qubits} to term on qubits {self.target_qubits}.",
            )
        matrix = self.backend.np.kron(
            term.matrix, self.backend.matrices.I(2 ** (len(self) - len(term)))
        )
        matrix = self.backend.np.reshape(matrix, 2 * len(self) * (2,))
        order = []
        i = len(term)
        for qubit in self.target_qubits:
            if qubit in term.target_qubits:
                order.append(term.target_qubits.index(qubit))
            else:
                order.append(i)
                i += 1
        order.extend([x + len(order) for x in order])
        matrix = self.backend.np.transpose(matrix, order)
        matrix = self.backend.np.reshape(matrix, 2 * (2 ** len(self),))
        return HamiltonianTerm(
            self.matrix + matrix, *self.target_qubits, backend=self.backend
        )

    def __len__(self):
        return len(self.target_qubits)

    def __mul__(self, x):
        return HamiltonianTerm(
            x * self.matrix, *self.target_qubits, backend=self.backend
        )

    def __rmul__(self, x):
        return self.__mul__(x)

    def __call__(self, backend, state, nqubits, gate=None, density_matrix=False):
        """Applies the term on a given state vector or density matrix."""
        # TODO: improve this and understand why it works
        if isinstance(gate, bool) or gate is None:
            gate = self.gate
        if density_matrix:
            return backend.apply_gate_half_density_matrix(gate, state, nqubits)
        return backend.apply_gate(gate, state, nqubits)  # pylint: disable=E1102


class SymbolicTerm(HamiltonianTerm):
    """:class:`qibo.hamiltonians.terms.HamiltonianTerm` constructed using ``sympy`` expression.

    Example:
        .. testcode::

            from qibo.symbols import X, Y
            from qibo.hamiltonians.terms import SymbolicTerm
            sham = X(0) * X(1) + 2 * Y(0) * Y(1)
            termsdict = sham.as_coefficients_dict()
            sterms = [SymbolicTerm(c, f) for f, c in termsdict.items()]

    Args:
        coefficient (complex): Complex number coefficient of the underlying
            term in the Hamiltonian.
        factors (sympy.Expr): Sympy expression for the underlying term.
    """

    def __init__(self, coefficient, factors=1, backend: Optional[Backend] = None):
        self._gate = None
        self.hamiltonian = None
        self.backend = _check_backend(backend)
        self.coefficient = complex(coefficient)

        # List of :class:`qibo.symbols.Symbol` that represent the term factors
        self.factors = []
        # Dictionary that maps target qubit ids to a list of matrices that act on each qubit
        self.matrix_map = {}
        if factors != 1:
            for factor in factors.as_ordered_factors():
                factor, power = self._compute_power(factor)
                if isinstance(factor, sympy.Symbol):
                    # Skip any qibo.symbols.I factors
                    if isinstance(factor, I):
                        continue
                    # forces the backend of the factor
                    # this way it is not necessary to explicitely define the
                    # backend of a symbol, i.e. Z(q, backend=backend)
                    factor.backend = self.backend
                    if isinstance(factor.matrix, self.backend.tensor_types):
                        q = factor.target_qubit
                        if q in self.matrix_map:
                            # Check for possible simplifications only if current factor is X/Y/Z
                            if isinstance(factor, (X, Y, Z)):
                                self._simplify_q_factors(factor, q)
                            # Case whereby factor isn't X/Y/Z: Multiply matrix multiple times based on its power,
                            # see the ``SymbolicTerm.matrix`` property for the full matrix calculation
                            else:
                                self.factors.extend(power * [factor])
                                self.matrix_map[q].extend(power * [factor.matrix])
                        else:
                            self.factors.extend(power * [factor])
                            self.matrix_map[q] = power * [factor.matrix]

                    else:
                        self.coefficient *= factor.matrix
                elif factor == sympy.I:
                    self.coefficient *= 1j
                elif factor.is_number:
                    self.coefficient *= complex(factor)
                else:  # pragma: no cover
                    raise_error(TypeError, f"Cannot parse factor {factor}.")

        self.target_qubits = tuple(sorted(self.matrix_map.keys()))

    def __repr__(self):
        return f"{self.coefficient}*" + "*".join(str(factor) for factor in self.factors)

    @staticmethod
    def _compute_power(factor):
        """Break down a factor into a base term and its power, e.g. `X**n` returns `X` and `n`. In case the base term is a Pauli also simplify the power calculation, i.e. return the identity for even `n` and the Pauli itself for odd `n`."""
        if isinstance(factor, sympy.Pow):
            factor, power = factor.args
            assert isinstance(power, sympy.Integer)
            assert isinstance(factor, sympy.Symbol)
            # if the symbol is a Pauli (i.e. a qibo symbol) and `power` is even
            # the power is the identity, thus the factor vanishes. Otherwise,
            # for an odd exponent, it remains unchanged (i.e. `power`=1)
            if factor.__class__ in (I, X, Y, Z):
                if not int(power) % 2:
                    factor = sympy.N(1)
                else:
                    power = 1
            else:
                power = int(power)
        else:
            power = 1
        return factor, power

    def _simplify_q_factors(self, factor, q):
        """Simplify the factors acting on qubit ``q`` using Pauli algebra after factor is added to ``q``."""
        q_factors = [_factor for _factor in self.factors if _factor.target_qubit == q]
        while q_factors:
            q_factor = q_factors[-1]
            # Only simplify if the last term in self.matrix_map[q] is a Pauli matrix
            if q_factor.__class__ in PAULI_MAPPING:
                pauli_product = evaluate_pauli_product(
                    PAULI_MAPPING[q_factor.__class__] * PAULI_MAPPING[factor.__class__]
                )
                # Move -1 or sympy.I from the product into the coefficient
                if -1 in pauli_product.atoms():
                    pauli_product *= -1
                    self.coefficient *= -1
                if sympy.I in pauli_product.atoms():
                    pauli_product *= -sympy.I
                    self.coefficient *= 1j
                # If pauli_product is not identity, then update the original factor/matrix
                factor_index = (
                    len(self.factors) - self.factors[::-1].index(q_factor) - 1
                )
                if pauli_product in INVERSE_PAULI_MAPPING:
                    factor = INVERSE_PAULI_MAPPING[pauli_product](q)
                    factor.backend = self.backend
                    self.factors[factor_index] = factor
                    self.matrix_map[q][-1] = factor.matrix
                # Otherwise, just remove the factor/matrix
                else:
                    self.factors.pop(factor_index)
                    self.matrix_map[q].pop(-1)
                    if not self.matrix_map[q]:
                        del self.matrix_map[q]
                q_factors.pop(-1)
            # X/Y/Z factor wasn't the last term previously => Cannot simplify, just extend
            else:
                self.factors.extend([factor])
                self.matrix_map[q].extend([factor.matrix])
                break

    @cached_property
    def matrix(self):
        """Calculates the full matrix corresponding to this term.

        Returns:
            Matrix as a ``np.ndarray`` of shape ``(2 ** ntargets, 2 ** ntargets)``
            where ``ntargets`` is the number of qubits included in the factors
            of this term.
        """
        matrices = list(self.qubit_to_matrix_map.values())
        return complex(self.coefficient) * reduce(self.backend.np.kron, matrices)

    @cached_property
    def qubit_to_matrix_map(self) -> dict:
        """Dictionary mapping each qubit to the corresponding global matrix
        that acts on it, i.e. the product of the matrices of all the factors
        acting on it.
        """
        return {
            q: reduce(self.backend.np.matmul, self.matrix_map.get(q))
            for q in self.target_qubits
        }

    def copy(self):
        """Creates a shallow copy of the term with the same attributes."""
        new = self.__class__(self.coefficient)
        new.factors = self.factors
        new.matrix_map = self.matrix_map
        new.target_qubits = self.target_qubits
        new.backend = self.backend
        return new

    def __mul__(self, x):
        """Multiplication of scalar to the Hamiltonian term."""
        new = self.copy()
        new.coefficient *= x
        return new

    def __call__(self, backend, state, nqubits, density_matrix=False):
        for factor in self.factors:
            state = super().__call__(
                backend, state, nqubits, factor.gate, density_matrix
            )
        return self.coefficient * state

    def commute(self, term) -> bool:
        """Check whether this term commutes with another term."""
        for q in set(self.target_qubits).intersection(set(term.target_qubits)):
            m1 = self.qubit_to_matrix_map.get(q)
            m2 = term.qubit_to_matrix_map.get(q)
            if not self.backend.np.all(m1 @ m2 == m2 @ m1):
                return False
        return True


class TermGroup(list):
    """Collection of multiple :class:`qibo.hamiltonians.terms.HamiltonianTerm` objects.

    Allows merging multiple terms to a single one for faster exponentiation
    during Trotterized evolution.

    Args:
        term (:class:`qibo.hamiltonians.terms.HamiltonianTerm`): Parent term of the group.
            All terms appended later should target a subset of the parents'
            target qubits.
    """

    def __init__(self, term):
        super().__init__([term])
        self.target_qubits = set(term.target_qubits)
        self._term = None

    def append(self, term):
        """Appends a new :class:`qibo.hamiltonians.terms.HamiltonianTerm` to the collection."""
        super().append(term)
        self.target_qubits |= set(term.target_qubits)
        self._term = None

    def can_append(self, term):
        """Checks if a term can be appended to the group based on its target qubits."""
        return set(term.target_qubits).issubset(self.target_qubits)

    @classmethod
    def from_terms(cls, terms):
        """Divides a list of terms to multiple :class:`qibo.hamiltonians.terms.TermGroup`s.

        Terms that target the same qubits are grouped to the same group.

        Args:
            terms (list): List of :class:`qibo.hamiltonians.terms.HamiltonianTerm` objects.

        Returns:
            List of :class:`qibo.hamiltonians.terms.TermGroup` objects that contain
            all the given terms.
        """
        # split given terms according to their order
        # order = len(term.target_qubits)
        orders = {}
        for term in terms:
            if len(term) in orders:
                orders[len(term)].append(term)
            else:
                orders[len(term)] = [term]

        groups = []
        # start creating groups with the higher order terms as parents and then
        # append each term of lower order to the first compatible group
        for order in sorted(orders.keys())[::-1]:
            for child in orders[order]:
                flag = True
                for i, group in enumerate(groups):
                    if group.can_append(child):
                        group.append(child)
                        flag = False
                        break
                if flag:
                    groups.append(cls(child))
        return groups

    @property
    def term(self):
        """Returns a single :class:`qibo.hamiltonians.terms.HamiltonianTerm`. after merging all terms in the group."""
        if self._term is None:
            self._term = self.to_term()
        return self._term

    def to_term(self, coefficients={}):
        """Calculates a single :class:`qibo.hamiltonians.terms.HamiltonianTerm` by merging all terms in the group.

        Args:
            coefficients (dict): Optional dictionary that allows passing a different
                coefficient to each term according to its parent Hamiltonian.
                Useful for :class:`qibo.core.adiabatic.AdiabaticHamiltonian` calculations.
        """
        c = coefficients.get(self[0].hamiltonian)
        merged = self[0] * c if c is not None else self[0]
        for term in self[1:]:
            c = coefficients.get(term.hamiltonian)
            merged = merged.merge(term * c if c is not None else term)
        return merged
