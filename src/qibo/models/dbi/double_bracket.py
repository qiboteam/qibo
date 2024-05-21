from copy import deepcopy
from enum import Enum, auto
from typing import Optional

import numpy as np

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.utils_analytical import (
    energy_fluctuation_polynomial_expansion_coef,
    least_squares_polynomial_expansion_coef,
    off_diagonal_norm_polynomial_expansion_coef,
)
from qibo.models.dbi.utils_scheduling import (
    grid_search_step,
    hyperopt_step,
    polynomial_step,
)


class DoubleBracketGeneratorType(Enum):
    """Define DBF evolution."""

    canonical = auto()
    """Use canonical commutator."""
    single_commutator = auto()
    """Use single commutator."""
    group_commutator = auto()
    """Use group commutator approximation"""
    # TODO: add double commutator (does it converge?)


class DoubleBracketScheduling(Enum):
    """Define the DBI scheduling strategies."""

    hyperopt = hyperopt_step
    """Use hyperopt package."""
    grid_search = grid_search_step
    """Use greedy grid search."""
    polynomial_approximation = polynomial_step
    """Use polynomial expansion (analytical) of the loss function."""


class DoubleBracketCostFunction(Enum):
    """Define the DBI cost function."""

    off_diagonal_norm = auto()
    """Use off-diagonal norm as cost function."""
    least_squares = auto()
    """Use least squares as cost function."""
    energy_fluctuation = auto()
    """Use energy fluctuation as cost function."""


class DoubleBracketIteration:
    """
    Class implementing the Double Bracket iteration algorithm.
    For more details, see https://arxiv.org/pdf/2206.11772.pdf

    Args:
        hamiltonian (Hamiltonian): Starting Hamiltonian;
        mode (DoubleBracketGeneratorType): type of generator of the evolution.

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
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.canonical,
        scheduling: DoubleBracketScheduling = DoubleBracketScheduling.grid_search,
        cost: DoubleBracketCostFunction = DoubleBracketCostFunction.off_diagonal_norm,
        ref_state: int = 0,
    ):
        self.h = hamiltonian
        self.h0 = deepcopy(self.h)
        self.mode = mode
        self.scheduling = scheduling
        self.cost = cost
        self.ref_state = ref_state

    def __call__(
        self, step: float, mode: DoubleBracketGeneratorType = None, d: np.array = None
    ):
        r"""We use convention that $H' = U^\dagger H U$ where $U=e^{-sW}$ with $W=[D,H]$ (or depending on `mode` an approximation, see `eval_dbr_unitary`). If $s>0$ then for $D = \Delta(H)$ the GWW DBR will give a $\sigma$-decrease, see https://arxiv.org/abs/2206.11772."""

        operator = self.eval_dbr_unitary(step, mode, d)
        operator_dagger = self.backend.cast(
            np.matrix(self.backend.to_numpy(operator)).getH()
        )
        self.h.matrix = operator_dagger @ self.h.matrix @ operator
        return operator

    def eval_dbr_unitary(
        self, step: float, mode: DoubleBracketGeneratorType = None, d: np.array = None
    ):
        """In call we will are working in the convention that $H' = U^\\dagger H U$ where $U=e^{-sW}$ with $W=[D,H]$ or an approximation of that by a group commutator. That is handy because if we switch from the DBI in the Heisenberg picture for the Hamiltonian, we get that the transformation of the state is $|\\psi'\rangle = U |\\psi\rangle$ so that $\\langle H\rangle_{\\psi'} = \\langle H' \rangle_\\psi$ (i.e. when writing the unitary acting on the state dagger notation is avoided).

        The group commutator must approximate $U=e^{-s[D,H]}$. This is achieved by setting $r = \\sqrt{s}$ so that
        $$V = e^{-irH}e^{irD}e^{irH}e^{-irD}$$
        because
        $$e^{-irH}De^{irH} = D+ir[D,H]+O(r^2)$$
        so
        $$V\approx e^{irD +i^2 r^2[D,H] + O(r^2) -irD} \approx U\\ .$$
        See the app in https://arxiv.org/abs/2206.11772 for a derivation.
        """
        if mode is None:
            mode = self.mode

        if mode is DoubleBracketGeneratorType.canonical:
            operator = self.backend.calculate_matrix_exp(
                -1.0j * step,
                self.commutator(self.diagonal_h_matrix, self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.single_commutator:
            if d is None:
                d = self.diagonal_h_matrix
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(self.backend.cast(d), self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.group_commutator:
            if d is None:
                d = self.diagonal_h_matrix

            sqrt_step = np.sqrt(step)
            operator = (
                self.h.exp(-np.sqrt(step))
                @ self.backend.calculate_matrix_exp(np.sqrt(step), d)
                @ self.h.exp(np.sqrt(step))
                @ self.backend.calculate_matrix_exp(-np.sqrt(step), d)
            )
        else:
            raise_error(ValueError, f"Mode { mode } not recognized.")
        return operator

    @staticmethod
    def commutator(a, b):
        """Compute commutator between two arrays."""
        return a @ b - b @ a

    @property
    def diagonal_h_matrix(self):
        """Diagonal H matrix."""
        return self.backend.cast(np.diag(np.diag(self.backend.to_numpy(self.h.matrix))))

    @property
    def off_diag_h(self):
        return self.h.matrix - self.diagonal_h_matrix

    @property
    def off_diagonal_norm(self):
        r"""Hilbert Schmidt norm of off-diagonal part of H matrix, namely :math:`\\text{Tr}(\\sqrt{A^{\\dagger} A})`."""
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

    def least_squares(self, d: np.array):
        """Least squares cost function."""
        h_np = self.backend.cast(self.h.matrix)

        return np.real(0.5 * np.linalg.norm(d) ** 2 - np.trace(h_np @ d))

    def choose_step(
        self,
        d: Optional[np.array] = None,
        scheduling: Optional[DoubleBracketScheduling] = None,
        **kwargs,
    ):
        """
        Calculate the optimal step using respective `scheduling` methods.
        """
        if scheduling is None:
            scheduling = self.scheduling
        step = scheduling(self, d=d, **kwargs)
        if (
            step is None
            and scheduling == DoubleBracketScheduling.polynomial_approximation
        ):
            kwargs["n"] = kwargs.get("n", 3)
            kwargs["n"] += 1
            # if n==n_max, return None
            step = scheduling(self, d=d, **kwargs)
        return step

    def loss(self, step: float, d: np.array = None, look_ahead: int = 1):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step: iteration step.
            d: diagonal operator, use canonical by default.
            look_ahead: number of iteration steps to compute the loss function;
        """
        # copy initial hamiltonian
        h_copy = deepcopy(self.h)

        for _ in range(look_ahead):
            self.__call__(mode=self.mode, step=step, d=d)

        # loss values depending on the cost function
        if self.cost == DoubleBracketCostFunction.off_diagonal_norm:
            loss = self.off_diagonal_norm
        elif self.cost == DoubleBracketCostFunction.least_squares:
            loss = self.least_squares(d)
        elif self.cost == DoubleBracketCostFunction.energy_fluctuation:
            loss = self.energy_fluctuation(self.ref_state)

        # set back the initial configuration
        self.h = h_copy

        return loss

    def energy_fluctuation(self, state):
        """
        Evaluate energy fluctuation

        .. math::
            \\Xi(\\mu) = \\sqrt{\\langle\\mu|\\hat{H}^2|\\mu\\rangle - \\langle\\mu|\\hat{H}|\\mu\\rangle^2} \\,

        for a given state :math:`|\\mu\\rangle`.

        Args:
            state (np.ndarray): quantum state to be used to compute the energy fluctuation with H.
        """
        h_np = self.backend.cast(np.diag(np.diag(self.backend.to_numpy(self.h.matrix))))
        h2 = h_np @ h_np
        state_cast = self.backend.cast(state)
        state_conj = self.backend.cast(state.conj())
        a = state_conj @ h2 @ state_cast
        b = state_conj @ h_np @ state_cast
        return (np.sqrt(np.real(a - b**2))).item()
        r  # return np.real(self.h.energy_fluctuation(state))

    def sigma(self, h: np.array):
        return self.backend.cast(h) - self.backend.cast(
            np.diag(np.diag(self.backend.to_numpy(h)))
        )

    def generate_Gamma_list(self, n: int, d: np.array):
        r"""Computes the n-nested Gamma functions, where $\Gamma_k=[W,...,[W,[W,H]]...]$, where we take k nested commutators with $W = [D, H]$"""
        W = self.commutator(self.backend.cast(d), self.sigma(self.h.matrix))
        Gamma_list = [self.h.matrix]
        for _ in range(n - 1):
            Gamma_list.append(self.commutator(W, Gamma_list[-1]))
        return Gamma_list

    def cost_expansion(self, d, n):
        if self.cost is DoubleBracketCostFunction.off_diagonal_norm:
            coef = off_diagonal_norm_polynomial_expansion_coef(self, d, n)
        elif self.cost is DoubleBracketCostFunction.least_squares:
            coef = least_squares_polynomial_expansion_coef(self, d, n)
        elif self.cost is DoubleBracketCostFunction.energy_fluctuation:
            coef = energy_fluctuation_polynomial_expansion_coef(
                self, d, n, self.ref_state
            )
        else:
            raise ValueError(f"Cost function {self.cost} not recognized.")
        return coef
