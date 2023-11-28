
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from double_bracket import DoubleBracketGeneratorType, DoubleBracketIteration
from hyperopt import hp, tpe

from qibo.config import raise_error
from qibo.hamiltonians import Hamiltonian


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

    def flow_forward(step):
        super().__call__(step)
