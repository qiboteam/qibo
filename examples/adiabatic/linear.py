"""Adiabatic evolution for the Ising Hamiltonian using linear scaling."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from qibo import callbacks, hamiltonians, models


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--hfield", default=4, type=float)
parser.add_argument("--T", default=1, type=float)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)


def main(nqubits, hfield, T, dt, solver):
    """Performs adiabatic evolution with critical TFIM as the "hard" Hamiltonian.

    Plots how the <H1> energy and the overlap with the actual ground state
    changes during the evolution.
    Linear scheduling is used.

    Args:
        nqubits: Number of qubits in the system.
        T: Total time of the adiabatic evolution.
        dt: Time step used for integration.
        solver: Solver used for integration.
    """
    h0 = hamiltonians.X(nqubits)
    h1 = hamiltonians.TFIM(nqubits, h=hfield)

    # Calculate target values (H1 ground state)
    target_state = h1.eigenvectors()[:, 0]
    target_energy = h1.eigenvalues()[0].numpy().real

    # Check ground state
    state_energy = (target_state.numpy().conj() *
                    (h1 @ target_state).numpy()).sum()
    np.testing.assert_allclose(state_energy.real, target_energy)

    energy = callbacks.Energy(h1)
    overlap = callbacks.Overlap(target_state)
    evolution = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt,
                                          solver=solver,
                                          callbacks=[energy, overlap])
    final_psi = evolution(T=T)

    tt = np.linspace(0, T, int(T / dt) + 1)

    plt.subplot(121)
    plt.plot(tt, energy[:], linewidth=2.0)
    plt.axhline(y=target_energy, color="red", linewidth=2.0)
    plt.xlabel("$t$")
    plt.ylabel("$H_1$")

    plt.subplot(122)
    plt.plot(tt, overlap[:], linewidth=2.0)
    plt.xlabel("$t$")
    plt.ylabel("Overlap")
    plt.show()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
