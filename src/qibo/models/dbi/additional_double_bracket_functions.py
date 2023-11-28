from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hyperopt import hp, tpe

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)


class DoubleBracketIterationStrategies(DoubleBracketIteration):
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        NSTEPS: int = 5,
        please_be_verbose=True,
        please_use_hyperopt=True,
        mode: DoubleBracketGeneratorType = DoubleBracketGeneratorType.canonical,
    ):
        super().__init__(hamiltonian, mode)
        self.NSTEPS = NSTEPS
        self.please_be_verbose = please_be_verbose
        self.pleas_use_hyperopt = please_use_hyperopt

    @staticmethod
    def visualize_matrix(matrix, title=""):
        """Visualize hamiltonian in a heatmap form."""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(title)
        try:
            im = ax.imshow(np.absolute(matrix), cmap="inferno")
        except TypeError:
            im = ax.imshow(np.absolute(matrix.get()), cmap="inferno")
        fig.colorbar(im, ax=ax)

    @staticmethod
    def visualize_drift(h0, h):
        """Visualize drift (absolute difference) of the evolved hamiltonian w.r.t. h0."""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(r"Drift: $|\hat{H}_0 - \hat{H}_{\ell}|$")
        try:
            im = ax.imshow(np.absolute(h0 - h), cmap="inferno")
        except TypeError:
            im = ax.imshow(np.absolute((h0 - h).get()), cmap="inferno")

        fig.colorbar(im, ax=ax)

    @staticmethod
    def plot_histories(histories, labels):
        """Plot off-diagonal norm histories over a sequential evolution."""
        colors = sns.color_palette("inferno", n_colors=len(histories)).as_hex()
        plt.figure(figsize=(5, 5 * 6 / 8))
        for i, (h, l) in enumerate(zip(histories, labels)):
            plt.plot(h, lw=2, color=colors[i], label=l, marker=".")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel(r"$\| \sigma(\hat{H}) \|^2$")
        plt.title("Loss function histories")
        plt.grid(True)
        plt.show()

    def flow_step(
        self,
        step: float,
        mode: DoubleBracketGeneratorType = None,
        d: np.array = None,
        update_h=False,
    ):
        """ "Computes the flowed hamiltonian after one double bracket iteration (and updates)"""
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
        if update_h is True:
            self.h.matrix = operator @ self.h.matrix @ operator_dagger
        return operator @ self.h.matrix @ operator_dagger

    def flow_forwards_invariant(self, H=None, step=0.1):
        """Execute multiple Double Bracket iterations with the same flow generator"""
        if H is None:
            H = deepcopy(self.h)
        for s in range(self.NSTEPS):
            if self.pleas_use_hyperopt is True:
                step = self.hyperopt_step(
                    step_min=1e-5,
                    step_max=1,
                    space=hp.uniform,
                    optimizer=tpe,
                    max_evals=100,
                    verbose=True,
                )
            self.flow_step(step, update_h=True)
