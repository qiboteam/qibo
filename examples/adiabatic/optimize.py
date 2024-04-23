"""Adiabatic evolution scheduling optimization for the Ising Hamiltonian."""

import argparse
from pathlib import Path

import numpy as np

from qibo import callbacks, hamiltonians, models

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--hfield", default=1, type=float)
parser.add_argument("--params", default="1", type=str)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)
parser.add_argument("--method", default="Powell", type=str)
parser.add_argument("--maxiter", default=None, type=int)
parser.add_argument("--save", default=None, type=str)


def spolynomial(t, params):
    """General polynomial scheduling satisfying s(0)=0 and s(1)=1"""
    f = sum(p * t ** (i + 2) for i, p in enumerate(params))
    f += (1 - np.sum(params)) * t
    return f


def main(nqubits, hfield, params, dt, solver, method, maxiter, save):
    """Optimizes the scheduling of the adiabatic evolution.

    The ansatz for s(t) is a polynomial whose order is defined by the length of
    ``params`` given.

    Args:
        nqubits (int): Number of qubits in the system.
        hfield (float): Transverse field Ising model h-field h value.
        params (str): Initial guess for the free parameters.
        dt (float): Time step used for integration.
        solver (str): Solver used for integration.
        method (str): Which scipy optimizer to use.
        maxiter (int): Maximum iterations for scipy optimizer.
        save (str): Name to use for saving optimization history.
            If ``None`` history will not be saved.
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

    evolution = models.AdiabaticEvolution(h0, h1, spolynomial, dt=dt, solver=solver)
    options = {"maxiter": maxiter, "disp": True}
    energy, parameters, _ = evolution.minimize(
        params, method=method, options=options, messages=True
    )

    print("\nBest energy found:", energy)
    print("Final parameters:", parameters)

    final_state = evolution(parameters[-1])
    overlap = bac.to_numpy(callbacks.Overlap(target_state).apply(bac, final_state)).real
    print("Target energy:", target_energy)
    print("Overlap:", overlap)

    if save:
        out_fol = Path("optparams")
        out_fol.mkdir(exist_ok=True)
        evolution.opt_history["loss"].append(target_energy)
        np.save(out_fol / f"{save}_n{nqubits}_loss.npy", evolution.opt_history["loss"])
        np.save(
            out_fol / f"{save}_n{nqubits}_params.npy", evolution.opt_history["params"]
        )


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["params"] = [float(x) for x in args["params"].split(",")]
    main(**args)
