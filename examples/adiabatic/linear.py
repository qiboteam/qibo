"""Adiabatic evolution for the Ising Hamiltonian using linear scaling."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from qibo import callbacks, hamiltonians, models

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--hfield", default=1, type=float)
parser.add_argument("--T", default=1, type=float)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)
parser.add_argument("--save", action="store_true")


def main(nqubits, hfield, T, dt, solver, save):
    """Performs adiabatic evolution with critical TFIM as the "hard" Hamiltonian.

    Plots how the <H1> energy and the overlap with the actual ground state
    changes during the evolution.
    Linear scheduling is used.

    Args:
        nqubits (int): Number of qubits in the system.
        hfield (float): Transverse field Ising model h-field h value.
        T (float): Total time of the adiabatic evolution.
        dt (float): Time step used for integration.
        solver (str): Solver used for integration.
        save (bool): Whether to save the plots.
    """
    h0 = hamiltonians.X(nqubits)
    h1 = hamiltonians.TFIM(nqubits, h=hfield)
    bac = h1.backend

    # Calculate target values (H1 ground state)
    target_state = h1.ground_state()
    target_energy = bac.to_numpy(h1.eigenvalues()[0]).real

    # Check ground state
    state_energy = bac.to_numpy(h1.expectation(target_state)).real
    np.testing.assert_allclose(state_energy.real, target_energy)

    energy = callbacks.Energy(h1)
    overlap = callbacks.Overlap(target_state)
    evolution = models.AdiabaticEvolution(
        h0, h1, lambda t: t, dt=dt, solver=solver, callbacks=[energy, overlap]
    )
    final_psi = evolution(final_time=T)

    # Plots
    tt = np.linspace(0, T, int(T / dt) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(tt, energy[:], linewidth=2.0, label="Evolved state")
    plt.axhline(y=target_energy, color="red", linewidth=2.0, label="Ground state")
    plt.xlabel("$t$")
    plt.ylabel("$H_1$")
    plt.legend()

    plt.subplot(122)
    plt.plot(tt, overlap[:], linewidth=2.0)
    plt.xlabel("$t$")
    plt.ylabel("Overlap")

    if save:
        out_fol = Path("images")
        out_fol.mkdir(exist_ok=True)
        plt.savefig(out_fol / f"dynamics_n{nqubits}T{T}.png", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
