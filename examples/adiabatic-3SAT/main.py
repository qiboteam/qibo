#!/usr/bin/env python
import numpy as np
from qibo import hamiltonians, models, callbacks
import functions
import argparse


def main(nqubits, instance, T, dt, solver, plot, trotter, params,
         method, maxiter):
    """Adiabatic evoluition to find the solution of an exact cover instance.

    Args:
        nqubits (int): number of qubits for the file that contains the information of an Exact Cover instance.
        instance (int): intance used for the desired number of qubits.
        T (float): maximum schedule time. The larger T, better final results.
        dt (float): time interval for the evolution.
        solver (str): solver used for the adiabatic evolution.
        plot (bool): decides if plots of the energy and gap will be returned.
        trotter (bool): decides if a Trotter Hamiltonian will be used.
        params (list):
        method (str):
        maxiter (bool):

    Returns:
        result of the most probable outcome after the adiabatic evolution. And plots of the energy and gap energy
        during the adiabatic evolution.
    """
    # Read 3SAT clauses from file
    control, solution, clauses = functions.read_file(nqubits, instance)
    nqubits = int(control[0])
    # Define "easy" and "problem" Hamiltonians
    if trotter:
        print('Using Trotter decomposition for the Hamiltonian\n')
        parts0, parts1 = functions.trotter_dict(clauses)
        gs = lambda: functions.ground_state(nqubits)
        H0 = hamiltonians.TrotterHamiltonian(*parts0, ground_state=gs)
        H1 = hamiltonians.TrotterHamiltonian(*parts1) + len(clauses)
    else:
        print('Using the full Hamiltonian evolution\n')
        t = functions.times(nqubits, clauses)
        H0 = hamiltonians.Hamiltonian(nqubits, functions.h0(nqubits, t))
        H1 = hamiltonians.Hamiltonian(nqubits, functions.h_p(nqubits, clauses))

    print('-'*20+'\n')
    if plot and nqubits >= 14:
        print('Currently not possible to calculate gap energy for {} qubits.'
              '\n Proceeding to adiabatic evolution without plotting data.\n'
              ''.format(nqubits))
        plot = False
    if plot and method is not None:
        print('Not possible to calculate gap energy during optimization.')
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
        evolve = models.AdiabaticEvolution(H0, H1, s, dt, solver=solver,
                                           callbacks=[gap, ground, excited])
    else:
        evolve = models.AdiabaticEvolution(H0, H1, s, dt, solver=solver)

    if method is not None:
        print(f'Optimizing scheduling using {method}.\n')
        params.append(T)
        options = {"maxiter": maxiter, "disp": True}
        energy, params = evolve.minimize(params, method=method,
                                         options=options)
        T = params[-1]

    # Perform evolution
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    final_state = evolve(final_time=T, initial_state=initial_state)
    output_dec = (np.abs(final_state.numpy())**2).argmax()
    max_output = "{0:0{bits}b}".format(output_dec, bits = nqubits)
    max_prob = (np.abs(final_state.numpy())**2).max()
    print("Exact cover instance with {} qubits.\n".format(nqubits))
    if solution:
        print('Known solution: {}\n'.format(''.join(solution)))
    print('-'*20+'\n')
    print(f'Adiabatic evolution with total time {T}, evolution step {dt} and '
          f'solver {solver}.\n')
    print(f'Most common solution after adiabatic evolution: {max_output}.\n')
    print(f'Found with probability: {max_prob}.\n')
    if plot:
        print('-'*20+'\n')
        functions.plot(nqubits, ground[:], excited[:], gap[:], dt, T)
        print('Plots finished.\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=8, type=int)
    parser.add_argument("--instance", default=1, type=int)
    parser.add_argument("--T", default=10, type=float)
    parser.add_argument("--dt", default=1e-2, type=float)
    parser.add_argument("--solver", default="exp", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--trotter", action="store_true")
    parser.add_argument("--params", default=None, type=str)
    parser.add_argument("--method", default=None, type=str)
    parser.add_argument("--maxiter", default=None, type=int)
    args = vars(parser.parse_args())
    if args["params"] is not None:
        args["params"] = [float(x) for x in args["params"].split(",")]
    main(**args)
