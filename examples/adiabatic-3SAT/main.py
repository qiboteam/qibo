#!/usr/bin/env python
import numpy as np
from qibo import hamiltonians, models, callbacks
import functions
import argparse


def main(file_name, T, dt, solver):
    """Adiabatic evoluition to find the solution of an exact cover instance
    Args:
        file_name (str): name of the file that contains the information of an Exact Cover instance.
        dt (float): time interval for the evolution.
        solver (str): solver used for the adiabatic evolution.

    Returns:
        result of the most probable outcome after the adiabatic evolution. And plots of the energy and gap energy
        during the adiabatic evolution.
    """
    control, solution, clauses = functions.read_file(file_name)
    nqubits = int(control[0])
    t = functions.times(nqubits, clauses)
    H0 = hamiltonians.Hamiltonian(nqubits, functions.h0(nqubits, t))
    H1 = hamiltonians.Hamiltonian(nqubits, functions.h_p(nqubits, clauses))
    initial_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)
    s = lambda t: t
    ground = callbacks.Gap(0)
    excited = callbacks.Gap(1)
    gap = callbacks.Gap()
    evolve = models.AdiabaticEvolution(H0, H1, s, dt, solver=solver, callbacks=[gap, ground, excited])
    final_state = evolve(final_time=T, initial_state=initial_state)
    max_output = "{0:0{bits}b}".format((np.abs(final_state.numpy())**2).argmax(), bits = nqubits)
    max_prob = (np.abs(final_state.numpy())**2).max()
    print("Exact cover instance with {} qubits.\n".format(nqubits))
    if solution:
        print('Known solution: {}\n'.format(''.join(solution)))
    print('-'*20+'\n')
    print('Adiabatic evolution with total time {T}, evolution step {dt} and solver {solver}.\n'.format(T=T, dt=dt, solver=solver))
    print('Most common solution after adiabatic evolution: {}.\n'.format(max_output))
    print('Found with probability: {}.\n'.format(max_prob))
    print('-'*20+'\n')
    functions.plot(ground[:], excited[:], gap[:], dt, T)
    print('Plots finished.\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=8, type=int)
    parser.add_argument("--T", default=10, type=float)
    parser.add_argument("--dt", default=1e-2, type=float)
    parser.add_argument("--solver", default="exp", type=str)
    args = vars(parser.parse_args())
    args["file_name"] = 'n{}.txt'.format(args.pop('nqubits'))
    main(**args)
