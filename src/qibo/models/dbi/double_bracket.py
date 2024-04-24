from copy import deepcopy
from enum import Enum, auto
from functools import partial

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
    ):
        self.h = hamiltonian
        self.h0 = deepcopy(self.h)
        self.mode = mode

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
                -1.0j * step,
                self.commutator(d, self.h.matrix),
            )
        elif mode is DoubleBracketGeneratorType.group_commutator:
            if d is None:
                d = self.diagonal_h_matrix

            sqrt_step = np.sqrt(step)
            operator = (
                self.h.exp(sqrt_step)
                @ self.backend.calculate_matrix_exp(-sqrt_step, d)
                @ self.h.exp(-sqrt_step)
                @ self.backend.calculate_matrix_exp(sqrt_step, d)
            )
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

    def hyperopt_step(
        self,
        step_min: float = 1e-5,
        step_max: float = 1,
        max_evals: int = 1000,
        space: callable = None,
        optimizer: callable = None,
        look_ahead: int = 1,
        verbose: bool = False,
        d: np.array = None,
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
            verbose: level of verbosity;
            d: diagonal operator for generating double-bracket iterations.

        Returns:
            (float): optimized best iteration step.
        """
        if space is None:
            space = hyperopt.hp.uniform
        if optimizer is None:
            optimizer = hyperopt.tpe

        space = space("step", step_min, step_max)
        best = hyperopt.fmin(
            fn=partial(self.loss, d=d, look_ahead=look_ahead),
            space=space,
            algo=optimizer.suggest,
            max_evals=max_evals,
            verbose=verbose,
        )
        return best["step"]

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
