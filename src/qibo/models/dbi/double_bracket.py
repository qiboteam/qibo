from copy import deepcopy
from enum import Enum, auto
from typing import Optional

import hyperopt
import numpy as np

from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.utils_scheduling import (
    grid_search_step,
    hyperopt_step,
    polynomial_step,
    simulated_annealing_step,
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
    simulated_annealing = simulated_annealing_step
    """Use simulated annealing algorithm"""


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
        state: int = 0,
    ):
        self.h = hamiltonian
        self.h0 = deepcopy(self.h)
        self.mode = mode
        self.scheduling = scheduling
        self.cost = cost
        self.state = state

    def __call__(
        self, step: float, mode: DoubleBracketGeneratorType = None, d: np.array = None
    ):
        if mode is None:
            mode = self.mode

        if mode is DoubleBracketGeneratorType.canonical:
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(self.diagonal_h_matrix, self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.single_commutator:
            if d is None:
                d = self.diagonal_h_matrix
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(d, self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.group_commutator:
            if d is None:
                d = self.diagonal_h_matrix
            operator = (
                self.h.exp(-step)
                @ self.backend.calculate_matrix_exp(-step, d)
                @ self.h.exp(step)
                @ self.backend.calculate_matrix_exp(step, d)
            )
        operator_dagger = self.backend.cast(
            np.matrix(self.backend.to_numpy(operator)).getH()
        )

        self.h.matrix = operator @ self.h.matrix @ operator_dagger

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

    def least_squares(self, D: np.array):
        """Least squares cost function."""
        H = self.h.matrix
        return -np.real(
            np.trace(H @ D) - 0.5 * (np.linalg.norm(H) ** 2 + np.linalg.norm(D) ** 2)
        )

    def choose_step(
        self,
        d: Optional[np.array] = None,
        scheduling: Optional[DoubleBracketScheduling] = None,
        **kwargs,
    ):
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
            loss = self.energy_fluctuation(self.state)

        # set back the initial configuration
        self.h = h_copy

        return loss

    def energy_fluctuation(self, state=None):
        """
        Evaluate energy fluctuation

        .. math::
            \\Xi(\\mu) = \\sqrt{\\langle\\mu|\\hat{H}^2|\\mu\\rangle - \\langle\\mu|\\hat{H}|\\mu\\rangle^2} \\,

        for a given state :math:`|\\mu\\rangle`.

        Args:
            state (np.ndarray): quantum state to be used to compute the energy fluctuation with H.
        """
        if state is None:
            state = self.state
        state_vector = np.zeros(len(self.h.matrix))
        state_vector[state] = 1.0
        return np.real(self.h.energy_fluctuation(state_vector))

    def sigma(self, h: np.array):
        return h - self.backend.cast(np.diag(np.diag(self.backend.to_numpy(h))))

    def generate_Gamma_list(self, n: int, d: np.array):
        r"""Computes the n-nested Gamma functions, where $\Gamma_k=[W,...,[W,[W,H]]...]$, where we take k nested commutators with $W = [D, H]$"""
        W = self.commutator(d, self.sigma(self.h.matrix))
        Gamma_list = [self.h.matrix]
        for _ in range(n - 1):
            Gamma_list.append(self.commutator(W, Gamma_list[-1]))
        return Gamma_list
