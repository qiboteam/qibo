#!/usr/bin/env python
import argparse

import functions
import numpy as np

from qibo import callbacks, hamiltonians, models


def main(nqubits, instance, T, dt, solver, plot, dense, params, method, maxiter):
    """Adiabatic evoluition to find the solution of an exact cover instance.

    Args:
        nqubits (int): number of qubits for the file that contains the
            information of an Exact Cover instance.
        instance (int): intance used for the desired number of qubits.
        T (float): maximum schedule time. The larger T, better final results.
        dt (float): time interval for the evolution.
        solver (str): solver used for the adiabatic evolution.
        plot (bool): decides if plots of the energy and gap will be returned.
        dense (bool): decides if the full Hamiltonian matrix will be used.
        params (list): list of polynomial coefficients for scheduling function.
            Default is linear scheduling.
        method (str): Method to use for scheduling optimization (optional).
        maxiter (bool): Maximum iterations for scheduling optimization (optional).

    Returns:
        Result of the most probable outcome after the adiabatic evolution.
        Plots of the ground and excited state energies and the underlying gap
        during the adiabatic evolution. The plots are created only if the
        ``--plot`` option is enabled.
    """
    # Read 3SAT clauses from file
    control, solution, clauses = functions.read_file(nqubits, instance)
    nqubits = int(control[0])
    # Define "easy" and "problem" Hamiltonians
    times = functions.times(nqubits, clauses)
    sh0 = functions.h_initial(nqubits, times)
    sh1 = functions.h_problem(nqubits, clauses)
    H0 = hamiltonians.SymbolicHamiltonian(sh0)
    H1 = hamiltonians.SymbolicHamiltonian(sh1)
    if dense:
        print("Using the full Hamiltonian evolution\n")
        H0, H1 = H0.dense, H1.dense
    else:
        print("Using Trotter decomposition for the Hamiltonian\n")

    print("-" * 20 + "\n")
    if plot and nqubits >= 14:
        print(
            f"Currently not possible to calculate gap energy for {nqubits} qubits."
            + "\n Proceeding to adiabatic evolution without plotting data.\n"
        )
        plot = False
    if plot and method is not None:
        print("Not possible to calculate gap energy during optimization.")
        plot = False

    # Define scheduling according to given params
    if params is None:
        # default is linear scheduling
        s = lambda t: t
    else:
        if method is None:
            s = lambda t: functions.spolynomial(t, params)
        else:
            s = functions.spolynomial

    # Define evolution model and (optionally) callbacks
    if plot:
        ground = callbacks.Gap(0)
        excited = callbacks.Gap(1)
        gap = callbacks.Gap()
        evolve = models.AdiabaticEvolution(
            H0, H1, s, dt, solver=solver, callbacks=[gap, ground, excited]
        )
    else:
        evolve = models.AdiabaticEvolution(H0, H1, s, dt, solver=solver)

    if method is not None:
        print(f"Optimizing scheduling using {method}.\n")
        if params is None:
            params = [T]
        else:
            params.append(T)
        if method == "sgd":
            options = {"nepochs": maxiter}
        else:
            options = {"maxiter": maxiter, "disp": True}
        energy, params, _ = evolve.minimize(params, method=method, options=options)
        T = params[-1]

    # Perform evolution
    initial_state = np.ones(2**nqubits) / np.sqrt(2**nqubits)
    final_state = evolve(final_time=T, initial_state=initial_state)
    output_dec = (np.abs(final_state) ** 2).argmax()
    max_output = "{0:0{bits}b}".format(output_dec, bits=nqubits)
    max_prob = (np.abs(final_state) ** 2).max()
    print(f"Exact cover instance with {nqubits} qubits.\n")
    if solution:
        print(f"Known solution: {''.join(solution)}\n")
    print("-" * 20 + "\n")
    print(
        f"Adiabatic evolution with total time {T}, evolution step {dt} and "
        f"solver {solver}.\n"
    )
    print(f"Most common solution after adiabatic evolution: {max_output}.\n")
    print(f"Found with probability: {max_prob}.\n")
    if plot:
        print("-" * 20 + "\n")
        functions.plot(nqubits, ground[:], excited[:], gap[:], dt, T)
        print("Plots finished.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=8, type=int)
    parser.add_argument("--instance", default=1, type=int)
    parser.add_argument("--T", default=10, type=float)
    parser.add_argument("--dt", default=1e-2, type=float)
    parser.add_argument("--solver", default="exp", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--params", default=None, type=str)
    parser.add_argument("--method", default=None, type=str)
    parser.add_argument("--maxiter", default=None, type=int)
    args = vars(parser.parse_args())
    if args["params"] is not None:
        args["params"] = [float(x) for x in args["params"].split(",")]
    main(**args)
