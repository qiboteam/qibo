from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from qibo.backends import _check_backend
from qibo.config import raise_error


def _validate_colors(colors):
    if colors is None:
        return "#ff7f0e", "#1f77b4"

    if len(colors) != 2:
        raise_error(
            ValueError, f"Colors must be a list of length 2, got {len(colors)} instead"
        )

    pos_color = "#ff7f0e" if colors[0] is None else colors[0]
    neg_color = "#1f77b4" if colors[1] is None else colors[1]

    return pos_color, neg_color


def _get_labels(nqubits):
    indices = np.arange(2**nqubits)
    bits = np.array(np.unravel_index(indices, [2] * nqubits)).T
    labels = ["".join(str(b) for b in row) for row in bits]
    return labels


# Based on Qiskit density state plot
# https://github.com/Qiskit/qiskit/blob/stable/2.0/qiskit/visualization/state_visualization.py#L372-L622
def plot_density_hist(
    circuit,
    title: str = "",
    alpha: float = 0.5,
    colors: Optional[List[str]] = None,
    fig_width: int = 16,
    fig_height: int = 8,
    backend=None,
):
    """Plot the real and imaginary parts of the density matrix.

    Given a :class:`qibo.models.circuit.Circuit`, plots the real and imaginary parts
    of the final density matrix as separate 3D cityscape plots, side by side, and
    with a gray ``z=0`` plane for the imaginary part.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit to visualize.
        title (str, optional): Title of the plot. Defaults to ``""``.
        alpha (float, optional): Transparency level for the bars in the plot.
            Defaults to :math:`0.5`.
        colors (list, optional): A list of two colors for the positive and negative
            parts of the density matrix. If ``None``, default colors will be used.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.
        fig_width (int, optional): Width of the figure in inches. Defaults to ``16``.
        fig_height (int, optional): Height of the figure in inches. Defaults to ``8``.

    Returns:
        tuple: Respectively, the figure, and axes for the real and the imaginary parts.
    """

    # if exec_circ has measurements then, error measure gates are present
    if len(circuit.measurements) > 0:
        raise_error(
            ValueError,
            "Circuit must not contain measurement gates for density matrix visualization",
        )

    backend = _check_backend(backend)
    # Execute the circuit to get the state
    exec_circ = backend.execute_circuit(circuit)

    state = exec_circ.state()

    # Create a density matrix from state vector
    if not circuit.density_matrix:
        state = backend.cast(state)
        state = backend.np.outer(state, backend.np.conj(state))

    nqubits = circuit.nqubits
    row_names = _get_labels(nqubits)
    column_names = _get_labels(nqubits)

    matrix_real = state.real
    matrix_imag = state.imag
    del state

    label_y, label_x = matrix_real.shape[:2]
    xpos = np.arange(0, label_x, 1)
    ypos = np.arange(0, label_y, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(label_x * label_y)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dzr = backend.np.reshape(matrix_real, (-1))
    dzi = backend.np.reshape(matrix_imag, (-1))

    max_dzr = np.max(dzr)
    max_dzi = np.max(dzi)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="w")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)

    max_plot_size = min(fig_width / 2.25, fig_height)
    max_font_size = int(2 * max_plot_size)
    max_zoom = 10 / (10 + np.sqrt(max_plot_size))

    pos_color, neg_color = _validate_colors(colors)
    cmap_pos = LinearSegmentedColormap.from_list("cmap_pos", 4 * [pos_color])
    cmap_neg = LinearSegmentedColormap.from_list("cmap_neg", 4 * [neg_color])

    for ax, dz, zlabel in (
        (ax1, dzr, "Real"),
        (ax2, dzi, "Imaginary"),
    ):

        max_dz = np.max(dz)
        min_dz = np.min(dz)

        # Normalize the heights for colormap
        norm_pos = plt.Normalize(vmin=0, vmax=max_dz)
        norm_neg = plt.Normalize(vmin=min_dz, vmax=0)

        # Create a color array based on the heights
        colors_mapping = np.array(
            [
                (
                    cmap_pos(norm_pos(height))
                    if height >= 0
                    else cmap_neg(norm_neg(height))
                )
                for height in dz
            ]
        )

        for mask, zorder in ((dz < 0, 0.625), (dz >= 0, 0.875)):
            if np.any(mask):
                ax.bar3d(
                    xpos[mask],
                    ypos[mask],
                    zpos[mask],
                    dx[mask],
                    dy[mask],
                    dz[mask],
                    alpha=alpha,
                    zorder=zorder,
                    color=colors_mapping[mask],
                    shade=True,
                )

        if min_dz < 0 < max_dz:
            xlim, ylim = [0, label_x], [0, label_y]
            verts = [list(zip(xlim + xlim[::-1], np.repeat(ylim, 2), [0] * 4))]
            plane = Poly3DCollection(verts, alpha=0.25, facecolor="k", linewidths=1)
            plane.set_zorder(0.75)
            ax.add_collection3d(plane)

        ax.set_title(zlabel, fontsize=max_font_size)

        ax.set_xticks(np.arange(0.5, label_x + 0.5, 1))
        ax.set_yticks(np.arange(0.5, label_y + 0.5, 1))

        if max_dz != min_dz:
            ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-9, max_dzi))
        else:
            if min_dz == 0:
                ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-9, max_dzi))
            else:
                ax.axes.set_zlim3d(auto=True)
        ax.get_autoscalez_on()

        ax.xaxis.set_ticklabels(
            row_names, fontsize=max_font_size, rotation=45, ha="right", va="top"
        )
        ax.yaxis.set_ticklabels(
            column_names, fontsize=max_font_size, rotation=-22.5, ha="left", va="center"
        )

        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(max_font_size)
            tick.label1.set_horizontalalignment("left")
            tick.label1.set_verticalalignment("bottom")

        ax.set_xlabel("Ket", labelpad=20)
        ax.xaxis.label.set_rotation(30)
        ax.set_ylabel("Bra", labelpad=15)
        ax.yaxis.label.set_rotation(30)
        ax.set_box_aspect(aspect=(4, 4, 4), zoom=max_zoom)
        ax.set_xmargin(0)
        ax.set_ymargin(0)

    if title != "":
        fig.suptitle(title, fontsize=max_font_size)
    fig.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)

    return fig, ax1, ax2
