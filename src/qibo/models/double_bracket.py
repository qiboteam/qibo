from enum import Enum, auto
from functools import partial

import hyperopt
import numpy as np

from ..config import raise_error
from ..hamiltonians import Hamiltonian


class FlowGeneratorType(Enum):
    """Define DBF evolution."""

    canonical = auto()
    """Use canonical commutator."""
    single_commutator = auto()
    """Use single commutator."""
    group_commutator = auto()
    """Use group commutator approximation"""
    # TODO: add double commutator (does it converge?)


class DoubleBracketFlow:
    """
    Class implementing the Double Bracket flow algorithm.
    For more details, see https://arxiv.org/pdf/2206.11772.pdf

    Args:
        hamiltonian (Hamiltonian): Starting Hamiltonian

    Example:
        .. code-block:: python

        import numpy as np
        from qibo.models.double_bracket import DoubleBracketFlow, FlowGeneratorType
        from qibo.quantum_info import random_hermitian

        nqubits = 4
        h0 = random_hermitian(2**nqubits)
        dbf = DoubleBracketFlow(Hamiltonian(nqubits=nqubits, matrix=h0))

        # diagonalized matrix
        dbf.h
    """

    def __init__(self, hamiltonian: Hamiltonian):
        # TODO: consider passing Mode here
        self.h = self.h0 = hamiltonian

    def __call__(self, step: float, mode: FlowGeneratorType, d: np.array = None):
        if mode is FlowGeneratorType.canonical:
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(self.diagonal_h_matrix, self.h.matrix),
            )
        elif mode is FlowGeneratorType.single_commutator:
            if d is None:
                raise_error(ValueError, f"Cannot use group_commutator with matrix {d}")
            operator = self.backend.calculate_matrix_exp(
                1.0j * step,
                self.commutator(d, self.h.matrix),
            )
        elif mode is FlowGeneratorType.group_commutator:
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
    def off_diagonal_norm(self):
        """Norm of off-diagonal part of H matrix."""
        off_diag_h = self.h.matrix - self.diagonal_h_matrix
        off_diag_h_dag = self.backend.cast(
            np.matrix(self.backend.to_numpy(off_diag_h)).getH()
        )
        return np.real(np.trace(self.backend.to_numpy(off_diag_h_dag @ off_diag_h)))

    @property
    def backend(self):
        """Get Hamiltonian's backend."""
        return self.h0.backend

    def hyperopt_step(
        self,
        step_min: float = 0.0001,
        step_max: float = 0.5,
        max_evals: int = 1000,
        space: callable = hyperopt.hp.uniform,
        optimizer: callable = hyperopt.tpe,
        look_ahead: int = 1,
        verbose: bool = False,
    ):
        """
        Optimize flow step.

        Args:
            step_min: lower bound of the search grid;
            step_max: upper bound of the search grid;
            max_evals: maximum number of iterations done by the hyperoptimizer;
            space: see hyperopt.hp possibilities;
            optimizer: see hyperopt algorithms;
            look_ahead: number of flow steps to compute the loss function;
            verbose: level of verbosity.

        Returns:
            (float): optimized best flow step.
        """

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
            step: flow step.
            look_ahead: number of flow steps to compute the loss function;
        """
        # copy initial hamiltonian
        h_copy = self.h

        # off_diagonal_norm's value before the steps
        old_loss = self.off_diagonal_norm

        for _ in range(look_ahead):
            self.__call__(mode=FlowGeneratorType.canonical, step=step)

        # off_diagonal_norm's value after the steps
        new_loss = self.off_diagonal_norm

        # set back the initial configuration
        self.h = h_copy

        return new_loss - old_loss
