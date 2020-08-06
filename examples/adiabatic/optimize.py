"""Adiabatic evolution for the Ising Hamiltonian using linear scaling."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from qibo import callbacks, hamiltonians, models


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--T", default=1, type=float)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)
parser.add_argument("--method", default="BFGS", type=str)


def main(nqubits, T, dt, solver, method):
    """Performs adiabatic evolution with Ising as the "hard" Hamiltonian.

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
    h1 = hamiltonians.TFIM(nqubits)

    # Calculate target values (H1 ground state)
    target_state = h1.eigenvectors()[:, 0]
    target_energy = h1.eigenvalues()[0].numpy().real

    # Check ground state
    state_energy = (target_state.numpy().conj() *
                    (h1 @ target_state).numpy()).sum()
    np.testing.assert_allclose(state_energy.real, target_energy)

    #energy = callbacks.Energy(h1)
    #overlap = callbacks.Overlap(target_state)
    #s = lambda t, p: p[0] * t ** 3 + p[1] * t ** 2 + p[2] * t + (1 - p.sum()) * np.sqrt(t)
    s = lambda t: t
    evolution = models.AdiabaticEvolution(h0, h1, s, dt=dt, solver=solver)

    print("Target energy:", target_energy)
    energy, parameters = evolution.minimize([T], method=method)

    print("Best energy:", energy)
    print("p =", parameters)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
