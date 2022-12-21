import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from qibo import *


def plot_density_hist(circuit):
    """Function that plots density matrix as a 3d histogram

    Args:
        circuit (:class:`qibo.models.Circuit`): a circuit with either a state vector or density
    """

    state = circuit.execute().state()
    # Create a density matrix from state vector
    if not circuit.density_matrix:
        ket = np.asarray(state)
        state = np.outer(ket, ket.conj())

    # Plot real values of density matrix
    matrix = state.real

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xpos = [range(matrix.shape[0])]
    ypos = [range(matrix.shape[1])]
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten("F")
    ypos = ypos.flatten("F")
    zpos = np.zeros_like(xpos)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = matrix.flatten()

    # Ticks length
    number_of_states_x = len(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(state.shape[0])]
    )
    number_of_states_y = len(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(2**circuit.nqubits)]
    )
    ax.set_xticks(0.5 + np.arange(number_of_states_x))
    ax.set_yticks(0.5 + np.arange(number_of_states_y))

    # Ticklabels
    ax.set_xticklabels(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(state.shape[0])]
    )
    ax.set_yticklabels(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(2**circuit.nqubits)]
    )
    ax.set_title("Real")

    # Color
    cmap = cm.get_cmap("jet")
    # Get range of colorbars so we can normalize
    max_height = np.max(dz)
    min_height = np.min(dz)
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort="average")

    # Plot imaginary values of a density matrix
    matrix1 = state.imag

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")

    xpos1 = [range(matrix1.shape[0])]
    ypos1 = [range(matrix1.shape[1])]
    xpos1, ypos1 = np.meshgrid(xpos1, ypos1)
    xpos1 = xpos1.flatten("F")
    ypos1 = ypos1.flatten("F")
    zpos1 = np.zeros_like(xpos1)

    dx1 = 0.5 * np.ones_like(zpos1)
    dy1 = dx1.copy()
    dz1 = matrix1.flatten()

    number_of_states_x = len(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(state.shape[0])]
    )
    number_of_states_y = len(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(2**circuit.nqubits)]
    )

    ax1.set_xticks(0.5 + np.arange(number_of_states_x))
    ax1.set_yticks(0.5 + np.arange(number_of_states_y))

    ax1.set_xticklabels(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(state.shape[0])]
    )
    ax1.set_yticklabels(
        [bin(i)[2:].zfill(circuit.nqubits) for i in range(2**circuit.nqubits)]
    )
    ax1.set_title("Imaginary")

    ax1.bar3d(xpos1, ypos1, zpos1, dx1, dy1, dz1, color=rgba, zsort="average")

    plt.show()
