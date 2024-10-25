from copy import copy
from enum import Enum, auto
from typing import Optional

import numpy as np
import optuna

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.utils import *
from qibo.models.dbi.utils_scheduling import (
    grid_search_step,
    hyperopt_step,
    polynomial_step,
    simulated_annealing_step,
)
from qibo.quantum_info.linalg_operations import commutator, matrix_exponentiation


class DoubleBracketGeneratorType(Enum):
    """Define DBF evolution."""

    canonical = auto()
    """Use canonical commutator."""
    single_commutator = auto()
    """Use single commutator."""
    group_commutator = auto()
    """Use group commutator approximation"""
    group_commutator_third_order = auto()
    """Implements Eq. (8) of Ref. [1], i.e.

    .. math::
        e^{\\frac{\\sqrt{5}-1}{2}isH} \\,
            e^{\\frac{\\sqrt{5}-1}{2}isD} \\,
            e^{-isH} \\,
            e^{isD} \\,
            e^{\\frac{3-\\sqrt{5}}{2}isH} \\,
            e^{isD} \\approx e^{-s^2[H,D]} + O(s^4) \\, .

    :math:`s` must be taken as :math:`\\sqrt{s}` to approximate the flow using the commutator.
    """


class DoubleBracketCostFunction(str, Enum):
    """Define the DBI cost function."""

    off_diagonal_norm = "off_diagonal_norm"
    """Use off-diagonal norm as cost function."""
    least_squares = "least_squares"
    """Use least squares as cost function."""
    energy_fluctuation = "energy_fluctuation"
    """Use energy fluctuation as cost function."""


class DoubleBracketScheduling(Enum):
    """Define the DBI scheduling strategies."""

    hyperopt = hyperopt_step
    """Use optuna package to hyperoptimize the DBI step."""
    grid_search = grid_search_step
    """Use greedy grid search."""
    polynomial_approximation = polynomial_step
    """Use polynomial expansion (analytical) of the loss function."""
    simulated_annealing = simulated_annealing_step
    """Use simulated annealing algorithm"""


class DoubleBracketIteration:
    """
    Class implementing the Double Bracket iteration algorithm. For more details, see Ref. [1].

    Args:
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): starting Hamiltonian.
        mode (:class:`qibo.models.dbi.double_bracket.DoubleBracketGeneratorType`):
            type of generator of the evolution.
        scheduling (:class:`qibo.models.dbi.double_bracket.DoubleBracketScheduling`):
            type of scheduling strategy.
        cost (:class:`qibo.models.dbi.double_bracket.DoubleBracketCostFunction`):
            type of cost function.
        ref_state (ndarray): reference state for computing the energy fluctuation.

    Example:
        .. testcode::

            from qibo.models.dbi.double_bracket import DoubleBracketIteration, DoubleBracketGeneratorType
            from qibo.quantum_info import random_hermitian
            from qibo.hamiltonians import Hamiltonian

            nqubits = 4
            h0 = random_hermitian(2**nqubits, seed=2)
            dbf = DoubleBracketIteration(Hamiltonian(nqubits=nqubits, matrix=h0))

            # diagonalized matrix
            dbf.h

    References:
        1. M. Gluza, *Double-bracket quantum algorithms for diagonalization*.
        `arXiv:2206.11772 [quant-ph] <https://arxiv.org/abs/2206.11772>`_.
    """

    def __init__(
        self,
        hamiltonian,
        mode=DoubleBracketGeneratorType.canonical,
        scheduling=DoubleBracketScheduling.grid_search,
        cost=DoubleBracketCostFunction.off_diagonal_norm,
        ref_state=None,
    ):
        self.h = hamiltonian
        self.h0 = copy(self.h)
        self.mode = mode
        self.scheduling = scheduling
        self.cost = cost
        self.ref_state = ref_state

    def __call__(self, step: float, mode=None, d=None):
        """We use the following convention:

        .. math::
            H^{'} = U^{\\dagger} \\, H \\, U \\, ,

        where :math:`U=e^{-s\\,W}`, and :math:`W =[D, H]` (or depending on ``mode`` an
        approximation, see `eval_dbr_unitary`). If :math:`s > 0`, then,
        for :math:`D = \\Delta(H)`, the GWW DBR will give a :math:`\\sigma`-decrease.

        References:
            1. M. Gluza, *Double-bracket quantum algorithms for diagonalization*.
            `arXiv:2206.11772 [quant-ph] <https://arxiv.org/abs/2206.11772>`_."""

        operator = self.eval_dbr_unitary(step, mode, d)
        operator_dagger = self.backend.cast(
            np.array(np.matrix(self.backend.to_numpy(operator)).getH())
        )
        self.h.matrix = operator_dagger @ self.h.matrix @ operator
        return operator

    def eval_dbr_unitary(
        self,
        step: float,
        mode=None,
        d=None,
    ):
        """In :meth:`qibo.models.dbi.double_bracket.DoubleBracketIteration.__call__`,
        we are working in the following convention:

        .. math::
            H^{'} = U^{\\dagger} \\, H \\, U \\, ,

        where :math:`U = e^{-s\\,W}`, and  :math:`W = [D, H]`
        (or an approximation of that by a group commutator).
        That is convenient because if we switch from the DBI in the Heisenberg picture for the
        Hamiltonian, we get that the transformation of the state is
        :math:`|\\psi'\\rangle = U \\, |\\psi\\rangle`, so that

        .. math::
            \\langle H\\rangle_{\\psi'} = \\langle H' \\rangle_\\psi \\, ,

        i.e. when writing the unitary acting on the state dagger notation is avoided).
        The group commutator must approximate :math:`U = e^{-s\\, [D,H]}`.
        This is achieved by setting :math:`r = \\sqrt{s}` so that

        .. math::
            V = e^{-i\\,r\\,H} \\, e^{i\\,r\\,D} \\, e^{i\\,r\\,H} \\, e^{-i\\,r\\,D}

        because

        .. math::
            e^{-i\\,r\\,H} \\, D \\, e^{i\\,r\\,H} = D + i\\,r\\,[D, H] +O(r^2)

        so

        .. math::
            V \\approx \\exp\\left(i\\,r\\,D + i^2 \\, r^2 \\, [D, H] + O(r^2) -i\\,r\\,D\\right) \\approx U \\, .

        See the Appendix in Ref. [1] for the complete derivation.

        References:
            1. M. Gluza, *Double-bracket quantum algorithms for diagonalization*.
            `arXiv:2206.11772 [quant-ph] <https://arxiv.org/abs/2206.11772>`_.
        """
        if mode is None:
            mode = self.mode

        if mode is DoubleBracketGeneratorType.canonical:
            operator = matrix_exponentiation(
                -1.0j * step,
                self.commutator(self.diagonal_h_matrix, self.h.matrix),
                backend=self.backend,
            )
        elif mode is DoubleBracketGeneratorType.single_commutator:
            if d is None:
                d = self.diagonal_h_matrix
            operator = matrix_exponentiation(
                -1.0j * step,
                self.commutator(self.backend.cast(d), self.h.matrix),
                backend=self.backend,
            )
        elif mode is DoubleBracketGeneratorType.group_commutator:
            if d is None:
                d = self.diagonal_h_matrix
            operator = (
                self.h.exp(step)
                @ matrix_exponentiation(-step, d, backend=self.backend)
                @ self.h.exp(-step)
                @ matrix_exponentiation(step, d, backend=self.backend)
            )
        elif mode is DoubleBracketGeneratorType.group_commutator_third_order:
            if d is None:
                d = self.diagonal_h_matrix
            operator = (
                self.h.exp(-step * (np.sqrt(5) - 1) / 2)
                @ matrix_exponentiation(
                    -step * (np.sqrt(5) - 1) / 2, d, backend=self.backend
                )
                @ self.h.exp(step)
                @ matrix_exponentiation(
                    step * (np.sqrt(5) + 1) / 2, d, backend=self.backend
                )
                @ self.h.exp(-step * (3 - np.sqrt(5)) / 2)
                @ matrix_exponentiation(-step, d, backend=self.backend)
            )
            operator = (
                matrix_exponentiation(step, d, backend=self.backend)
                @ self.h.exp(step * (3 - np.sqrt(5)) / 2)
                @ matrix_exponentiation(
                    -step * (np.sqrt(5) + 1) / 2, d, backend=self.backend
                )
                @ self.h.exp(-step)
                @ matrix_exponentiation(
                    step * (np.sqrt(5) - 1) / 2, d, backend=self.backend
                )
                @ self.h.exp(step * (np.sqrt(5) - 1) / 2)
            )
        else:
            raise_error(
                NotImplementedError, f"The mode {mode} is not supported"
            )  # pragma: no cover

        return operator

    @staticmethod
    def commutator(a, b):
        """Compute commutator between two arrays."""
        return commutator(a, b)

    @property
    def diagonal_h_matrix(self):
        """Diagonal H matrix."""
        return self.backend.cast(np.diag(np.diag(self.backend.to_numpy(self.h.matrix))))

    @property
    def off_diag_h(self):
        """Off-diagonal H matrix."""
        return self.h.matrix - self.diagonal_h_matrix

    @property
    def off_diagonal_norm(self):
        """Hilbert Schmidt norm of off-diagonal part of H matrix, namely :math:`\\text{Tr}(\\sqrt{A^{\\dagger} A})`."""
        off_diag_h_dag = self.backend.cast(
            np.matrix(self.backend.to_numpy(self.off_diag_h)).getH()
        )
        return np.sqrt(
            np.real(np.trace(self.backend.to_numpy(off_diag_h_dag @ self.off_diag_h)))
        )

    @property
    def backend(self):
        """Get Hamiltonian's backend."""
        return self.h0.backend

    @property
    def nqubits(self):
        """Number of qubits."""
        return self.h.nqubits

    def least_squares(self, d: np.array):
        """Least squares cost function."""
        d = self.backend.to_numpy(d)
        return np.real(
            0.5 * np.linalg.norm(d) ** 2
            - np.trace(self.backend.to_numpy(self.h.matrix) @ d)
        )

    def choose_step(
        self,
        d: Optional[np.array] = None,
        scheduling: Optional[DoubleBracketScheduling] = None,
        **kwargs,
    ):
        """Calculate the optimal step using respective the `scheduling` methods."""
        if scheduling is None:
            scheduling = self.scheduling
        step = scheduling(self, d=d, **kwargs)
        # TODO: write test for this case
        if (
            step is None
            and scheduling is DoubleBracketScheduling.polynomial_approximation
        ):  # pragma: no cover
            kwargs["n"] = kwargs.get("n", 3)
            kwargs["n"] += 1
            # if n==n_max, return None
            step = scheduling(self, d=d, **kwargs)
            # if for a given polynomial order n, no solution is found, we increase the order of the polynomial by 1
        return step

    def loss(self, step: float, d: np.array = None, look_ahead: int = 1):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step (float): iteration step.
            d (np.array): diagonal operator, use canonical by default.
            look_ahead (int): number of iteration steps to compute the loss function;
        """
        # copy initial hamiltonian
        h_copy = copy(self.h)

        for _ in range(look_ahead):
            self.__call__(mode=self.mode, step=step, d=d)

        # loss values depending on the cost function
        if self.cost is DoubleBracketCostFunction.off_diagonal_norm:
            loss = self.off_diagonal_norm
        elif self.cost is DoubleBracketCostFunction.least_squares:
            loss = self.least_squares(d)
        elif self.cost == DoubleBracketCostFunction.energy_fluctuation:
            loss = self.energy_fluctuation(self.ref_state)

        # set back the initial configuration
        self.h = h_copy

        return loss

    def energy_fluctuation(self, state):
        """
        Evaluate energy fluctuation.

        .. math::
            \\Xi(\\mu) = \\sqrt{\\langle\\mu|\\hat{H}^2|\\mu\\rangle - \\langle\\mu|\\hat{H}|\\mu\\rangle^2} \\,

        for a given state :math:`|\\mu\\rangle`.

        Args:
            state (np.ndarray): quantum state to be used to compute the energy fluctuation with H.
        """
        return self.h.energy_fluctuation(state)

    def sigma(self, h: np.array):
        """Returns the off-diagonal restriction of matrix `h`."""
        return self.backend.cast(h) - self.backend.cast(
            np.diag(np.diag(self.backend.to_numpy(h)))
        )

    def generate_gamma_list(self, n: int, d: np.array):
        r"""Computes the n-nested Gamma functions, where $\Gamma_k=[W,...,[W,[W,H]]...]$, where we take k nested commutators with $W = [D, H]$"""
        W = self.commutator(self.backend.cast(d), self.sigma(self.h.matrix))
        gamma_list = [self.h.matrix]
        for _ in range(n - 1):
            gamma_list.append(self.commutator(W, gamma_list[-1]))
        return gamma_list

    def cost_expansion(self, d, n):
        d = self.backend.cast(d)

        if self.cost is DoubleBracketCostFunction.off_diagonal_norm:
            coef = off_diagonal_norm_polynomial_expansion_coef(self, d, n)
        elif self.cost is DoubleBracketCostFunction.least_squares:
            coef = least_squares_polynomial_expansion_coef(
                self, d, n, backend=self.backend
            )
        elif self.cost is DoubleBracketCostFunction.energy_fluctuation:
            coef = energy_fluctuation_polynomial_expansion_coef(
                self, d, n, self.ref_state
            )
        else:  # pragma: no cover
            raise ValueError(f"Cost function {self.cost} not recognized.")
        return coef
