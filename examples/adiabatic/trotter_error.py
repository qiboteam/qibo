"""Error of evolution using Trotter decomposition."""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from qibo import callbacks, hamiltonians, models

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--hfield", default=1, type=float)
parser.add_argument("--T", default=1, type=float)
parser.add_argument("--save", action="store_true")


def main(nqubits, hfield, T, save):
    """Compares exact with Trotterized state evolution.

    Plots the overlap between the exact final state and the state from the
    Trotterized evolution as a function of the number of time steps used for
    the discretization of time.
    The transverse field Ising model (TFIM) is used as a toy model for this
    experiment.

    Args:
        nqubits (int): Number of qubits in the system.
        hfield (float): Transverse field Ising model h-field h value.
        T (float): Total time of the adiabatic evolution.
        save (bool): Whether to save the plots.
    """
    dense_h = hamiltonians.TFIM(nqubits, h=hfield)
    trotter_h = hamiltonians.TFIM(nqubits, h=hfield, dense=False)
    initial_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)

    nsteps_list = np.arange(50, 550, 50)
    overlaps = []
    for nsteps in nsteps_list:
        exact_ev = models.StateEvolution(dense_h, dt=T / nsteps)
        trotter_ev = models.StateEvolution(trotter_h, dt=T / nsteps)
        exact_state = exact_ev(final_time=T, initial_state=np.copy(initial_state))
        trotter_state = trotter_ev(final_time=T, initial_state=np.copy(initial_state))
        ovlp = callbacks.Overlap(exact_state).apply(dense_h.backend, trotter_state)
        overlaps.append(dense_h.backend.to_numpy(ovlp))

    dt_list = T / nsteps_list
    overlaps = 1 - np.array(overlaps)

    exponent = int(linregress(np.log(dt_list), np.log(overlaps))[0])
    err = [
        overlaps[0] * (dt_list / dt_list[0]) ** (exponent - 1),
        overlaps[0] * (dt_list / dt_list[0]) ** exponent,
        overlaps[0] * (dt_list / dt_list[0]) ** (exponent + 1),
    ]
    alphas = [1.0, 0.7, 0.4]
    labels = [
        f"$\\delta t ^{exponent - 1}$",
        f"$\\delta t ^{exponent}$",
        f"$\\delta t ^{exponent - 1}$",
    ]

    plt.figure(figsize=(7, 4))
    plt.semilogy(
        nsteps_list, overlaps, marker="o", markersize=8, linewidth=2.0, label="Error"
    )
    for e, a, l in zip(err, alphas, labels):
        plt.semilogy(
            nsteps_list, e, color="red", alpha=a, linestyle="--", linewidth=2.0, label=l
        )
    plt.xlabel("Number of steps")
    plt.ylabel("$1 -$ Overlap")
    plt.legend()

    if save:
        plt.savefig(f"images/trotter_error_n{nqubits}T{T}.png", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
