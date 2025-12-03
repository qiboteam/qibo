"""Module defining Hamiltonian classes."""

import operator
from functools import cache, cached_property, reduce
from itertools import chain
from logging import warn
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy

from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.hamiltonians.abstract import AbstractHamiltonian
from qibo.hamiltonians.terms import SymbolicTerm
from qibo.symbols import PauliSymbol, Symbol, Z


class Hamiltonian(AbstractHamiltonian):
    """Hamiltonian based on a dense or sparse matrix representation.

    Args:
        nqubits (int): number of quantum bits.
        matrix (ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape :math:`2^{n} \\times 2^{n}`.
            Sparse matrices based on ``scipy.sparse`` for ``numpy`` / ``qibojit`` backends
            (or on ``tf.sparse`` for the ``tensorflow`` backend) are also supported.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """

    def __init__(self, nqubits, matrix, backend=None):
        from qibo.backends import _check_backend

        self._backend = _check_backend(backend)

        if not (
            isinstance(matrix, self.backend.tensor_types)
            or self.backend.is_sparse(matrix)
        ):
            raise_error(
                TypeError,
                f"Matrix of invalid type {type(matrix)} given during Hamiltonian initialization",
            )
        matrix = self.backend.cast(matrix)

        super().__init__()
        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    @property
    def matrix(self):
        """Returns the full matrix representation.

        For :math:`n` qubits, can be a dense :math:`2^{n} \\times 2^{n}` array or a sparse
        matrix, depending on how the Hamiltonian was created.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, mat):
        shape = tuple(mat.shape)
        if shape != 2 * (2**self.nqubits,):
            raise_error(
                ValueError,
                f"The Hamiltonian is defined for {self.nqubits} qubits "
                + f"while the given matrix has shape {shape}.",
            )
        self._matrix = mat

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, new_backend: Backend):
        self._backend = new_backend
        self._matrix = new_backend.cast(self._matrix, new_backend.dtype)

    def eigenvalues(self, k=6):
        if self._eigenvalues is None:
            self._eigenvalues = self.backend.calculate_eigenvalues(self.matrix, k)
        return self._eigenvalues

    def eigenvectors(self, k=6):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self.backend.calculate_eigenvectors(
                self.matrix, k
            )
        return self._eigenvectors

    def exp(self, a):
        from qibo.quantum_info.linalg_operations import (  # pylint: disable=C0415
            matrix_exponentiation,
        )

        if self._exp.get("a") != a:
            self._exp["a"] = a
            self._exp["result"] = matrix_exponentiation(
                self.matrix,
                -1j * a,
                self._eigenvectors,
                self._eigenvalues,
                self.backend,
            )
        return self._exp.get("result")

    def expectation(self, circuit, nshots=None, qubit_map=None):
        """Computes the expectation value for a given circuit. This works only for diagonal
        observables if ``nshots != None``.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to calculate the expectation value from.
                If the circuit has already been executed, this will just make use of the cached
                result, otherwise it will execute the circuit.
            nshots (int, optional): number of shots to calculate the expectation value, if ``None``
                it will try to compute the exact expectation value (if possible). Defaults to ``None``.

        Returns:
            float: The expectation value.
        """
        if not circuit.__class__.__name__ == "Circuit":  # pragma: no cover
            warn(
                "Calculation of expectation values starting from the state is deprecated, "
                + "use the ``expectation_from_state`` method if you really need it, "
                + "or simply pass the circuit you want to calculate the expectation value from."
            )
            return self.expectation_from_state(circuit)

        if nshots is None:
            return self.backend.expectation_observable_dense(circuit, self.matrix)

        from qibo import gates  # pylint: disable=import-outside-toplevel

        circuit = circuit.copy(True)
        circuit.add(gates.M(*range(self.nqubits)))
        return self.backend.expectation_diagonal_observable_dense_from_samples(
            circuit, self.matrix, self.nqubits, nshots, qubit_map
        )

    def expectation_from_samples(
        self,
        frequencies: Dict[str | int, int],
        qubit_map: Optional[Tuple[int, ...]] = None,
    ):
        """Compute the expectation value starting from some samples, works only for diagonal
        observables.

        Args:
            frequencies (Dict[str | int, int]): the dictionary of samples.
            qubit_map (Tuple[int, ...], optional): optional qubit reordering.

        Returns:
            float: The expectation value.
        """
        from qibo import Circuit  # pylint: disable=import-outside-toplevel

        circuit = Circuit(1)

        class TMP:
            def frequencies(self):
                return frequencies

        circuit._final_state = TMP()
        return self.backend.expectation_diagonal_observable_dense_from_samples(
            circuit, self.matrix, self.nqubits, nshots=1, qubit_map=qubit_map
        )

    def eye(self, dim: Optional[int] = None):
        """Generate Identity matrix with dimension ``dim``"""
        if dim is None:
            dim = int(self.matrix.shape[0])
        return self.backend.cast(self.backend.matrices.I(dim), dtype=self.matrix.dtype)

    def energy_fluctuation(self, circuit):
        """
        Evaluate energy fluctuation:

        .. math::
            \\Xi_{k}(\\mu) = \\sqrt{\\bra{\\mu} \\, H^{2} \\, \\ket{\\mu}
                - \\bra{\\mu} \\, H \\, \\ket{\\mu}^2} \\, .

        for a given state :math:`\\ket{\\mu}`.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to compute the energy fluctuation from.

        Returns:
            float: Energy fluctuation value.
        """
        energy = self.expectation(circuit)
        h = self.matrix
        h2 = Hamiltonian(nqubits=self.nqubits, matrix=h @ h, backend=self.backend)
        average_h2 = h2.expectation(circuit)
        return self.backend.np.sqrt(self.backend.np.abs(average_h2 - energy**2))

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.nqubits != other.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be added.",
                )
            new_matrix = self.matrix + other.matrix
        elif isinstance(other, self.backend.numeric_types) or isinstance(
            other, self.backend.tensor_types
        ):
            new_matrix = self.matrix + other * self.eye()
        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian addition to {type(other)} not implemented.",
            )
        return self.__class__(
            self.nqubits, new_matrix, backend=self.backend  # pylint: disable=E0606
        )

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            if self.nqubits != other.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be subtracted.",
                )
            new_matrix = self.matrix - other.matrix
        elif isinstance(other, self.backend.numeric_types):
            new_matrix = self.matrix - other * self.eye()
        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian subtraction to {type(other)} not implemented.",
            )
        return self.__class__(
            self.nqubits, new_matrix, backend=self.backend  # pylint: disable=E0606
        )

    def __rsub__(self, other):
        if isinstance(other, self.__class__):  # pragma: no cover
            # impractical case because it will be handled by `__sub__`
            if self.nqubits != other.nqubits:
                raise_error(
                    RuntimeError,
                    "Only hamiltonians with the same number of qubits can be added.",
                )
            new_matrix = other.matrix - self.matrix
        elif isinstance(other, self.backend.numeric_types):
            new_matrix = other * self.eye() - self.matrix
        else:
            raise_error(
                NotImplementedError,
                f"Hamiltonian subtraction to {type(other)} not implemented.",
            )
        return self.__class__(
            self.nqubits, new_matrix, backend=self.backend  # pylint: disable=E0606
        )

    def __mul__(self, other):
        if isinstance(other, self.backend.tensor_types):
            other = complex(other)
        elif not isinstance(other, self.backend.numeric_types):
            raise_error(
                NotImplementedError,
                f"Hamiltonian multiplication to {type(other)} not implemented.",
            )
        new_matrix = self.matrix * other
        r = self.__class__(self.nqubits, new_matrix, backend=self.backend)
        other = self.backend.cast(other)
        if self._eigenvalues is not None:
            if self.backend.np.real(other) >= 0:  # TODO: check for side effects K.qnp
                r._eigenvalues = other * self._eigenvalues
            elif not self.backend.is_sparse(self.matrix):
                axis = (0,) if (self.backend.platform == "pytorch") else 0
                r._eigenvalues = other * self.backend.np.flip(self._eigenvalues, axis)
        if self._eigenvectors is not None:
            if self.backend.np.real(other) > 0:  # TODO: see above
                r._eigenvectors = self._eigenvectors
            elif other == 0:
                r._eigenvectors = self.eye(int(self._eigenvectors.shape[0]))
        return r

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            matrix = self.matrix @ other.matrix
            return self.__class__(self.nqubits, matrix, backend=self.backend)

        if isinstance(other, self.backend.tensor_types):
            return self.matrix @ other

        raise_error(
            NotImplementedError,
            f"Hamiltonian matmul to {type(other)} not implemented.",
        )


def _calculate_nqubits_from_form(form):
    """Calculate number of qubits in the system described by the given
    Hamiltonian formula
    """
    nqubits = 0
    for symbol in form.free_symbols:
        if isinstance(symbol, Symbol):
            q = symbol.target_qubit
        else:
            raise_error(
                RuntimeError,
                f"Symbol {symbol} is not a ``qibo.symbols.Symbol``, you can define a custom symbol for {symbol} by subclassing ``qibo.symbols.Symbol``.",
            )
        if q > nqubits:  # pylint: disable=E0606
            nqubits = q
    return nqubits + 1


class SymbolicHamiltonian(AbstractHamiltonian):
    """Hamiltonian based on a symbolic representation.

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
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """

    def __init__(
        self,
        form: sympy.Expr,
        nqubits: Optional[int] = None,
        backend: Optional[Backend] = None,
    ):
        super().__init__()
        if not isinstance(form, sympy.Expr):
            raise_error(
                TypeError,
                f"The ``form`` of a ``SymbolicHamiltonian`` has to be a ``sympy.Expr``, but a ``{type(form)}`` was passed.",
            )
        self._form = form
        self.constant = 0  # used only when we perform calculations using ``_terms``

        self._backend = _check_backend(backend)

        self.nqubits = (
            _calculate_nqubits_from_form(form) if nqubits is None else nqubits
        )
        self._matrix = None

    def __repr__(self):
        return str(self.form)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, new_backend: Backend):
        self._backend = new_backend
        if self._matrix is not None:
            self._matrix = new_backend.cast(self._matrix, new_backend.dtype)

    @property
    def dense(self) -> "MatrixHamiltonian":  # type: ignore
        """Creates the equivalent Hamiltonian matrix."""
        return self.calculate_dense()

    @property
    def form(self):
        return self._form

    @form.setter
    def form(self, form):
        # Check that given form is a ``sympy`` expression
        if not isinstance(form, sympy.Expr):
            raise_error(
                TypeError,
                f"Symbolic Hamiltonian should be a ``sympy`` expression but is {type(form)}.",
            )
        self._form = form
        self.nqubits = _calculate_nqubits_from_form(form)

    @cached_property
    def terms(self):
        """List of terms of which the Hamiltonian is a sum of.

        Terms will be objects of type :class:`qibo.core.terms.HamiltonianTerm`.
        """
        # Calculate terms based on ``self.form``

        self.constant = 0.0

        form = sympy.expand(self.form)
        terms = []
        for factors, coeff in form.as_coefficients_dict().items():
            term = SymbolicTerm(coeff, factors, backend=self.backend)
            if term.target_qubits:
                # Check for any terms with the same factors and add their coefficients together
                found = False
                for i, _term in enumerate(terms):
                    if set(_term.factors) == set(term.factors):
                        found = True
                        terms[i].coefficient += term.coefficient
                        break
                if not found:
                    terms.append(term)
            else:
                self.constant += term.coefficient
        return terms

    @cached_property
    def simple_terms(self) -> Tuple[List[float], List[str], List[Tuple[int, ...]]]:
        """A simpler (more framework agnostic) representation of the of terms
        composing the Hamiltonian, defined as: their scalar coefficients,
        the strings of the names of their observables and the qubits they act on

        Returns:
            (Tuple[List[float], List[str], List[Tuple[int, ...]]])
        """
        term_coefficients, terms, term_qubits = [], [], []
        for term in self.terms:
            term_coefficients.append(term.coefficient)
            terms.append("".join(factor.__class__.__name__ for factor in term.factors))
            term_qubits.append(tuple(factor.target_qubit for factor in term.factors))
        return term_coefficients, terms, term_qubits

    @cached_property
    def diagonal_terms(self) -> list[list[SymbolicTerm]]:
        """List of terms that can be diagonalized simultaneously, i.e. that
        commute with each other. In detail each element of the list is a sublist
        of commuting ``SymbolicTerm``s.
        """
        diagonal_terms = []
        terms = self.terms
        # loop until there are no more terms
        while len(terms) > 0:
            # take the first (remaining) term
            t0 = terms[0]
            diagonal_term = [t0]
            removable_indices = {
                0,
            }
            # look for all the following terms that commute with the
            # first one and among themselves
            for i, t1 in enumerate(terms[1:], 1):
                # commutes with all the other terms -> append it
                if all(term.commute(t1) for term in diagonal_term):
                    diagonal_term.append(t1)
                    removable_indices.add(i)
            # append the new sublist of commuting terms found
            diagonal_terms.append(diagonal_term)
            # remove them from the original terms
            terms = [term for i, term in enumerate(terms) if i not in removable_indices]
        return diagonal_terms

    @cached_property
    def diagonal_simple_terms(
        self,
    ) -> Tuple[List[List[float]], List[List[str]], List[List[Tuple[int, ...]]]]:
        """A simpler (more framework agnostic) representation of the simultaneously
        diagonalizable terms of the hamiltonian, defined as: their scalar coefficients,
        the strings of the names of their observables and the qubits they act on.

        Returns:
            Tuple[List[List[float]], List[List[str]], List[List[Tuple[int, ...]]]]
        """
        terms_qubits = []
        terms_coefficients = []
        terms_observables = []
        for terms in self.diagonal_terms:
            tmp_qubits, tmp_coeffs, tmp_obs = [], [], []
            for term in terms:
                tmp_qubits.append(term.target_qubits)
                tmp_coeffs.append(term.coefficient.real)
                tmp_obs.append("".join(factor.name[0] for factor in term.factors))
            terms_qubits.append(tmp_qubits)
            terms_coefficients.append(tmp_coeffs)
            terms_observables.append(tmp_obs)
        return terms_coefficients, terms_observables, terms_qubits

    @property
    def matrix(self):
        """Returns the full matrix representation.

        Consisting of :math:`2^{n} \\times 2^{n}`` elements.
        """
        if self._matrix is None:
            self._matrix = self._get_symbol_matrix(self.form)
        return self._matrix

    def eigenvalues(self, k=6):
        return self.dense.eigenvalues(k)

    def eigenvectors(self, k=6):
        return self.dense.eigenvectors(k)

    def ground_state(self):
        return self.eigenvectors()[:, 0]

    def exp(self, a):
        return self.dense.exp(a)

    @cache
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
            ndarray: matrix corresponding to the given expression as an array
            of shape ``(2 ** self.nqubits, 2 ** self.nqubits)``.
        """
        if isinstance(term, sympy.Add):
            # symbolic op for addition
            result = sum(
                self._get_symbol_matrix(subterm) for subterm in term.as_ordered_terms()
            )

        elif isinstance(term, sympy.Mul):
            # symbolic op for multiplication
            # note that we need to use matrix multiplication even though
            # we use scalar symbols for convenience
            factors = term.as_ordered_factors()
            result = reduce(
                self.backend.np.matmul,
                (self._get_symbol_matrix(subterm) for subterm in factors),
            )

        elif isinstance(term, sympy.Pow):
            # symbolic op for power
            base, exponent = term.as_base_exp()
            matrix = self._get_symbol_matrix(base)
            matrix_power = (
                np.linalg.matrix_power
                if self.backend.name == "tensorflow"
                else self.backend.np.linalg.matrix_power
            )
            result = matrix_power(matrix, int(exponent))

        elif isinstance(term, Symbol):
            # if we have a Qibo symbol the matrix construction is
            # implemented in :meth:`qibo.core.terms.SymbolicTerm.full_matrix`.
            # I have to force the symbol's backend
            term.backend = self.backend
            result = term.full_matrix(self.nqubits)

        elif term.is_number:
            # if the term is number we should return in the form of identity
            # matrix because in expressions like `1 + Z`, `1` is not correspond
            # to the float 1 but the identity operator (matrix)
            result = complex(term) * self.backend.matrices.I(2**self.nqubits)

        else:
            raise_error(
                TypeError,
                f"Cannot calculate matrix for symbolic term of type {type(term)}.",
            )

        return result  # pylint: disable=E0606

    def _calculate_dense_from_form(self) -> Hamiltonian:
        """Calculates equivalent Hamiltonian using symbolic form.

        Useful when the term representation is not available.
        """
        return Hamiltonian(self.nqubits, self.matrix, backend=self.backend)

    def calculate_dense(self) -> Hamiltonian:
        log.warning(
            "Calculating the dense form of a symbolic Hamiltonian. "
            "This operation is memory inefficient."
        )
        # calculate dense matrix directly using the form to avoid the
        # costly ``sympy.expand`` call
        return self._calculate_dense_from_form()

    def expectation(self, circuit: "Circuit", nshots: Optional[int] = None) -> float:  # type: ignore
        """Computes the expectation value for a given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to calculate the expectation value from.
                If the circuit has already been executed, this will just make use of the cached
                result, otherwise it will execute the circuit.
            nshots (int, optional): number of shots to calculate the expectation value, if ``None``
                it will try to compute the exact expectation value (if possible). Defaults to ``None``.

        Returns:
            float: The expectation value.
        """
        if not circuit.__class__.__name__ == "Circuit":  # pragma: no cover
            warn(
                "Calculation of expectation values starting from the state is deprecated, "
                + "use the ``expectation_from_state`` method if you really need it, "
                + "or simply pass the circuit you want to calculate the expectation value from."
            )
            return self.expectation_from_state(circuit)
        if nshots is None:
            if not all(
                isinstance(factor, PauliSymbol)
                for term in self.terms
                for factor in term.factors
            ):
                return self.dense.expectation(circuit)
            terms_coefficients, terms, term_qubits = self.simple_terms
            return self.constant.real + self.backend.expectation_observable_symbolic(
                circuit, terms, term_qubits, terms_coefficients, self.nqubits
            )

        terms_coefficients, terms_observables, terms_qubits = self.diagonal_simple_terms
        return self.backend.expectation_observable_symbolic_from_samples(
            circuit,
            terms_coefficients,
            terms_observables,
            terms_qubits,
            self.nqubits,
            self.constant.real,
            nshots,
        )

    def expectation_from_samples(
        self,
        frequencies: Dict[str | int, int],
        qubit_map: Optional[Tuple[int, ...]] = None,
    ):
        """Compute the expectation value starting from some samples, works only for diagonal
        observables.

        Args:
            frequencies (Dict[str | int, int]): the dictionary of samples.
            qubit_map (Tuple[int, ...], optional): optional qubit reordering.

        Returns:
            float: The expectation value.
        """

        from qibo import Circuit

        circuit = Circuit(1)

        class TMP:
            def frequencies(self):
                return frequencies

        circuit._final_state = TMP()
        qubits, coefficients = [], []
        for term in self.terms:
            qubits.append(
                [
                    factor.target_qubit
                    for factor in term.factors
                    if factor.__class__.__name__ != "I"
                ]
            )
            coefficients.append(term.coefficient.real)
        return self.backend.expectation_diagonal_observable_symbolic_from_samples(
            circuit, self.nqubits, qubits, coefficients, nshots=1, qubit_map=qubit_map
        )

    def _compose(self, other, operator):
        form = self._form

        if isinstance(other, self.__class__):
            if self.nqubits != other.nqubits:
                raise_error(
                    RuntimeError,
                    "Only Hamiltonians with the same number of qubits can be composed.",
                )

            if other._form is not None:
                form = operator(form, other._form) if form is not None else other._form

        elif isinstance(other, (self.backend.numeric_types, self.backend.tensor_types)):
            form = (
                operator(form, complex(other)) if form is not None else complex(other)
            )
        else:
            raise_error(
                NotImplementedError,
                f"SymbolicHamiltonian composition to {type(other)} not implemented.",
            )

        return self.__class__(form=form, nqubits=self.nqubits, backend=self.backend)

    def __add__(self, other):
        return self._compose(other, operator.add)

    def __sub__(self, other):
        return self._compose(other, operator.sub)

    def __rsub__(self, other):
        return self._compose(other, lambda x, y: y - x)

    def __mul__(self, other):
        return self._compose(other, lambda x, y: y * x)

    def apply_gates(self, state, density_matrix=False):
        """Applies gates corresponding to the Hamiltonian terms.

        Gates are applied to the given state.

        Helper method for :meth:`qibo.hamiltonians.SymbolicHamiltonian.__matmul__`.
        """
        total = 0
        for term in self.terms:
            total += term(
                self.backend,
                self.backend.cast(state, copy=True),
                self.nqubits,
                density_matrix=density_matrix,
            )
        if self.constant:  # pragma: no cover
            total += self.constant * state
        return total

    def __matmul__(self, other):
        """Matrix multiplication with other Hamiltonians or state vectors."""
        if isinstance(other, self.__class__):
            return other * self

        if isinstance(other, self.backend.tensor_types):
            rank = len(tuple(other.shape))
            if rank not in (1, 2):
                raise_error(
                    NotImplementedError,
                    f"Cannot multiply Hamiltonian with rank-{rank} tensor.",
                )
            state_qubits = int(np.log2(int(other.shape[0])))
            if state_qubits != self.nqubits:
                raise_error(
                    ValueError,
                    f"Cannot multiply Hamiltonian on {self.nqubits} qubits to "
                    + f"state of {state_qubits} qubits.",
                )
            if rank == 1:  # state vector
                return self.apply_gates(other)

            return self.apply_gates(other, density_matrix=True)

        raise_error(
            NotImplementedError,
            f"Hamiltonian matmul to {type(other)} not implemented.",
        )

    def circuit(self, dt, accelerators=None):
        """Circuit that implements a Trotter step of this Hamiltonian.

        Args:
            dt (float): Time step used for Trotterization.
            accelerators (dict, optional): Dictionary with accelerators for distributed circuits.
                Defaults to ``None``.
        """
        from qibo import Circuit  # pylint: disable=import-outside-toplevel
        from qibo.hamiltonians.terms import (  # pylint: disable=import-outside-toplevel
            TermGroup,
        )

        groups = TermGroup.from_terms(self.terms)
        circuit = Circuit(self.nqubits, accelerators=accelerators)
        circuit.add(
            group.term.expgate(dt / 2.0) for group in chain(groups, groups[::-1])
        )

        return circuit
