import math
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Optional

import hyperopt
import numpy as np

from qibo.hamiltonians import Hamiltonian


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

    hyperopt = auto()
    """Use hyperopt package."""
    grid_search = auto()
    """Use greedy grid search."""
    polynomial_approximation = auto()
    """Use polynomial expansion (analytical) of the loss function."""


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
    ):
        self.h = hamiltonian
        self.h0 = deepcopy(self.h)
        self.mode = mode
        self.scheduling = scheduling

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

    def grid_search_step(
        self,
        step_min: float = 1e-5,
        step_max: float = 1,
        num_evals: int = 100,
        space: Optional[np.array] = None,
        d: Optional[np.array] = None,
    ):
        """
        Greedy optimization of the iteration step.

        Args:
            step_min: lower bound of the search grid;
            step_max: upper bound of the search grid;
            mnum_evals: number of iterations between step_min and step_max;
            d: diagonal operator for generating double-bracket iterations.

        Returns:
            (float): optimized best iteration step (minimizing off-diagonal norm).
        """
        if space is None:
            space = np.linspace(step_min, step_max, num_evals)

        if d is None:
            d = self.diagonal_h_matrix

        loss_list = [self.loss(step, d=d) for step in space]
        idx_max_loss = loss_list.index(min(loss_list))
        return space[idx_max_loss]

    def hyperopt_step(
        self,
        step_min: float = 1e-5,
        step_max: float = 1,
        max_evals: int = 1000,
        space: callable = None,
        optimizer: callable = None,
        look_ahead: int = 1,
        verbose: bool = False,
        d: Optional[np.array] = None,
    ):
        """
        Optimize iteration step using hyperopt.

        Args:
            step_min: lower bound of the search grid;
            step_max: upper bound of the search grid;
            max_evals: maximum number of iterations done by the hyperoptimizer;
            space: see hyperopt.hp possibilities;
            optimizer: see hyperopt algorithms;
            look_ahead: number of iteration steps to compute the loss function;
            verbose: level of verbosity;
            d: diagonal operator for generating double-bracket iterations.

        Returns:
            (float): optimized best iteration step (minimizing off-diagonal norm).
        """
        if space is None:
            space = hyperopt.hp.uniform
        if optimizer is None:
            optimizer = hyperopt.tpe
        if d is None:
            d = self.diagonal_h_matrix

        space = space("step", step_min, step_max)
        best = hyperopt.fmin(
            fn=partial(self.loss, d=d, look_ahead=look_ahead),
            space=space,
            algo=optimizer.suggest,
            max_evals=max_evals,
            verbose=verbose,
        )
        return best["step"]

    def polynomial_step(
        self,
        n: int = 4,
        n_max: int = 5,
        d: np.array = None,
        backup_scheduling: DoubleBracketScheduling = None,
    ):
        r"""
        Optimizes iteration step by solving the n_th order polynomial expansion of the loss function.
        e.g. $n=2$: $2\Trace(\sigma(\Gamma_1 + s\Gamma_2 + s^2/2\Gamma_3)\sigma(\Gamma_0 + s\Gamma_1 + s^2/2\Gamma_2))
        Args:
            n (int, optional): the order to which the loss function is expanded. Defaults to 4.
            n_max (int, optional): maximum order allowed for recurring calls of `polynomial_step`. Defaults to 5.
            d (np.array, optional): diagonal operator, default as $\delta(H)$.
            backup_scheduling (`DoubleBracketScheduling`): the scheduling method to use in case no real positive roots are found.
        """

        if d is None:
            d = self.diagonal_h_matrix

        if backup_scheduling is None:
            backup_scheduling = DoubleBracketScheduling.grid_search

        # list starting from s^n highest order to s^0
        sigma_gamma_list = np.array(
            [self.sigma(self.Gamma(k, d)) for k in range(n + 2)]
        )
        exp_list = np.array([1 / math.factorial(k) for k in range(n + 1)])
        # coefficients for rotation with [W,H] and H
        c1 = [
            exp_coef * delta_gamma
            for exp_coef, delta_gamma in zip(exp_list, sigma_gamma_list[1:])
        ]
        c2 = [
            exp_coef * delta_gamma
            for exp_coef, delta_gamma in zip(exp_list, sigma_gamma_list[:-1])
        ]
        # product coefficient
        trace_coefficients = [0] * (2 * n + 1)
        for k in range(n + 1):
            for j in range(n + 1):
                power = k + j
                product_matrix = c1[k] @ c2[j]
                trace_coefficients[power] += 2 * np.trace(product_matrix)
        taylor_coefficients = list(reversed(trace_coefficients[: n + 1]))
        roots = np.roots(taylor_coefficients)
        error = 1e-3
        real_positive_roots = [
            np.real(root)
            for root in roots
            if np.imag(root) < error and np.real(root) > 0
        ]
        # solution exists, return minimum s
        if len(real_positive_roots) > 0:
            return min(real_positive_roots), taylor_coefficients
        # solution does not exist, resort to backup scheduling
        elif (
            backup_scheduling == DoubleBracketScheduling.polynomial_approximation
            and n < n_max + 1
        ):
            return self.polynomial_step(
                n=n + 1, d=d, backup_scheduling=backup_scheduling
            )
        else:
            print(
                f"Unable to find roots with current order, resorting to {backup_scheduling}"
            )
            return self.choose_step(d=d, scheduling=backup_scheduling), list(
                reversed(trace_coefficients[: n + 1])
            )

    def choose_step(
        self,
        d: Optional[np.array] = None,
        scheduling: Optional[DoubleBracketScheduling] = None,
        **kwargs,
    ):
        if scheduling is None:
            scheduling = self.scheduling
        if scheduling is DoubleBracketScheduling.grid_search:
            return self.grid_search_step(d=d, **kwargs)
        if scheduling is DoubleBracketScheduling.hyperopt:
            return self.hyperopt_step(d=d, **kwargs)
        if scheduling is DoubleBracketScheduling.polynomial_approximation:
            # omit taylor coefficients
            step, _ = self.polynomial_step(d=d, **kwargs)
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

        # off_diagonal_norm's value after the steps
        loss = self.off_diagonal_norm

        # set back the initial configuration
        self.h = h_copy

        return loss

    def energy_fluctuation(self, state):
        """
        Evaluate energy fluctuation

        .. math::
            \\Xi_{k}(\\mu) = \\sqrt{\\langle\\mu|\\hat{H}^2|\\mu\\rangle - \\langle\\mu|\\hat{H}|\\mu\\rangle^2} \\,

        for a given state :math:`|\\mu\\rangle`.

        Args:
            state (np.ndarray): quantum state to be used to compute the energy fluctuation with H.
        """
        return self.h.energy_fluctuation(state)

    def sigma(self, h: np.array):
        return h - self.backend.cast(np.diag(np.diag(self.backend.to_numpy(h))))

    def Gamma(self, k: int, d: np.array):
        r"""Computes the k_th Gamma function i.e $\Gamma_k=[W,...,[W,[W,H]]...]$, where we take k nested commutators with $W = [D, H]$"""
        if k == 0:
            return self.h.matrix
        else:
            W = self.commutator(d, self.sigma(self.h.matrix))
            result = self.h.matrix
            for _ in range(k):
                result = self.commutator(W, result)
        return result
