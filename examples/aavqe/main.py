#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qibo import gates, models, hamiltonians
import argparse


def AAVQE(nqubits, layers, maxsteps, T_max, initial_parameters, easy_hamiltonian, problem_hamiltonian):
    """Implements the Adiabatically Assisted Variational Quantum Eigensolver (AAVQE).

    Args:
        nqubits (int): number of quantum bits.
        layers (int): number of ansatz layers.
        maxsteps (int): number of maximum iterations on each adiabatic step.
        T_max (int): number of maximum adiabatic steps.
        initial_parameters (array or list): values of the initial parameters.
        easy_hamiltonian (qibo.hamiltonians.Hamiltonian): initial Hamiltonian object.
        problem_hamiltonian (qibo.hamiltonians.Hamiltonian): problem Hamiltonian object.

    Returns:
        Groundstate energy of the problem Hamiltonian and best set of parameters.
    """
    # Create variational circuit
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        circuit.add(gates.VariationalLayer(range(nqubits), pairs,
                                           gates.RY, gates.CZ,
                                           np.zeros(nqubits), np.zeros(nqubits)))
        circuit.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add((gates.RY(i, theta=0) for i in range(nqubits)))

    for t in range(T_max+1):
        s = t/T_max
        print('s =',s)
        hamiltonian =  (1-s)*easy_hamiltonian + s*problem_hamiltonian
        vqe = models.VQE(circuit, hamiltonian)
        energy, params, _ = vqe.minimize(initial_parameters, method='Nelder-Mead',
                                         options={'maxfev': maxsteps},
                                         compile=False)
        initial_parameters = params
    return energy, params


def main(nqubits, layers, maxsteps, T_max):
    nparams = 2 * nqubits * layers + nqubits
    initial_parameters = np.random.uniform(0, 0.01, nparams)

    #Define the easy Hamiltonian and the problem Hamiltonian.
    easy_hamiltonian = hamiltonians.Z(nqubits)
    problem_hamiltonian = -1 * hamiltonians.TFIM(nqubits, h=1.0)

    #Run the AAVQE
    best, params = AAVQE(nqubits, layers, maxsteps, T_max, initial_parameters,
                         easy_hamiltonian, problem_hamiltonian)
    print('Final parameters: ', params)
    print('Final energy: ', best)

    #We compute the difference from the exact value to check performance
    eigenvalue = problem_hamiltonian.eigenvalues()
    print('Difference from exact value: ',best - eigenvalue[0].numpy().real)
    print('Log difference: ',-np.log10(best - eigenvalue[0].numpy().real))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=6, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--maxsteps", default=5000, type=int)
    parser.add_argument("--T_max", default=5, type=int)
    args = parser.parse_args()
    main(**vars(args))
