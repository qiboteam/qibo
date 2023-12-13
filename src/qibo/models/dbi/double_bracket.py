from copy import deepcopy
from enum import Enum, auto
from functools import partial

import hyperopt
import numpy as np

from qibo.config import raise_error
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


class DoubleBracketIteration:
    """
    Class implementing the Double Bracket iteration algorithm.
    For more details, see https://arxiv.org/pdf/2206.11772.pdf

    Args:
        hamiltonian (Hamiltonian): Starting Hamiltonian;
        mode (DoubleBracketGeneratorType): type of generator of the evolution.

    Example:
        .. testcode::

            import numpy as np
            from qibo.models.dbi.double_bracket import DoubleBracketIteration, DoubleBracketGeneratorType
            from qibo.hamiltonians import Hamiltonian
            from qibo.quantum_info import random_hermitian

            nqubits = 4
            h0 = random_hermitian(2**nqubits)
            dbf = DoubleBracketIteration(Hamiltonian(nqubits=nqubits, matrix=h0))

            # diagonalized matrix
            dbf.h
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.canonical,
    ):
        self.h = hamiltonian
        self.h0 = deepcopy(self.h)
        self.mode = mode

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
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(d, self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.group_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
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
        """Norm of off-diagonal part of H matrix."""
        off_diag_h_dag = self.backend.cast(
            np.matrix(self.backend.to_numpy(self.off_diag_h)).getH()
        )
        return np.real(
            np.trace(self.backend.to_numpy(off_diag_h_dag @ self.off_diag_h))
        )

    @property
    def backend(self):
        """Get Hamiltonian's backend."""
        return self.h0.backend

    def hyperopt_step(
        self,
        step_min: float = 1e-5,
        step_max: float = 1,
        max_evals: int = 1000,
        space: callable = None,
        optimizer: callable = None,
        look_ahead: int = 1,
        verbose: bool = False,
    ):
        """
        Optimize iteration step.

        Args:
            step_min: lower bound of the search grid;
            step_max: upper bound of the search grid;
            max_evals: maximum number of iterations done by the hyperoptimizer;
            space: see hyperopt.hp possibilities;
            optimizer: see hyperopt algorithms;
            look_ahead: number of iteration steps to compute the loss function;
            verbose: level of verbosity.

        Returns:
            (float): optimized best iteration step.
        """
        if space is None:
            space = hyperopt.hp.uniform
        if optimizer is None:
            optimizer = hyperopt.tpe

        space = space("step", step_min, step_max)
        best = hyperopt.fmin(
            fn=partial(self.loss, look_ahead=look_ahead),
            space=space,
            algo=optimizer.suggest,
            max_evals=max_evals,
            verbose=verbose,
        )

        return best["step"]

    def loss(self, step: float, look_ahead: int = 1):
        """
        Compute loss function distance between `look_ahead` steps.

        Args:
            step: iteration step.
            look_ahead: number of iteration steps to compute the loss function;
        """
        # copy initial hamiltonian
        h_copy = deepcopy(self.h)

        for _ in range(look_ahead):
            self.__call__(mode=self.mode, step=step)

        # off_diagonal_norm's value after the steps
        loss = self.off_diagonal_norm

        # set back the initial configuration
        self.h = h_copy

        return loss

    def energy_fluctuation(self, state):
        """Evaluate energy fluctuations"""
        energy = self.h.expectation(state)
        h = self.h.matrix
        h2 = Hamiltonian(nqubits=self.h.nqubits, matrix=h @ h, backend=self.backend)
        average_h2 = self.backend.calculate_expectation_state(h2, state, normalize=True)
        return np.sqrt(average_h2 - energy**2)
