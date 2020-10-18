#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qibo.models import Circuit
from qibo import hamiltonians, gates, models
from scipy.optimize import minimize
import argparse


def main(nqubits, layers, compress, lambdas):
    
    def encoder_hamiltonian_simple(nqubits, ncompress):
        """Creates the encoding Hamiltonian.
        Args:
            nqubits (int): total number of qubits.
            ncompress (int): number of discarded/trash qubits.

        Returns:
            Encoding Hamiltonian.
        """
        m0 = hamiltonians.Z(ncompress, numpy=True).matrix
        m1 = np.eye(2 ** (nqubits - ncompress), dtype=m0.dtype)
        ham = hamiltonians.Hamiltonian(nqubits, np.kron(m1, m0))
        return 0.5 * (ham + ncompress)

    def cost_function(params, count):
        """Evaluates the cost function to be minimized.

        Args:
            params (array or list): values of the parameters.

        Returns:
            Value of the cost function.
        """
        circuit = models.Circuit(nqubits)
        for l in range(layers):
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            for q in range(0, nqubits-1, 2):
                circuit.add(gates.CZ(q, q+1))
            for q in range(nqubits):
                circuit.add(gates.RY(q, theta=0))
            for q in range(1, nqubits-2, 2):
                circuit.add(gates.CZ(q, q+1))
            circuit.add(gates.CZ(0, nqubits-1))
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=0))

        cost = 0
        circuit.set_parameters(params) # this will change all thetas to the appropriate values
        for i in range(len(ising_groundstates)):
            final_state = circuit(np.copy(ising_groundstates[i]))
            cost += encoder.expectation(final_state).numpy().real

        if count[0] % 50 == 0:
            print(count[0], cost/len(ising_groundstates))
        count[0] += 1

        return cost/len(ising_groundstates)

    nparams = 2 * nqubits * layers + nqubits
    initial_params = np.random.uniform(0, 2*np.pi, nparams)
    encoder = encoder_hamiltonian_simple(nqubits, compress)

    ising_groundstates = []
    for lamb in lambdas:
        ising_ham = -1 * hamiltonians.TFIM(nqubits, h=lamb)
        ising_groundstates.append(ising_ham.eigenvectors()[0])

    count = [0]
    result = minimize(lambda p: cost_function(p, count), initial_params,
                      method='L-BFGS-B', options={'maxiter': 2.0e3, 'maxfun': 2.0e3})

    print('Final parameters: ', result.x)
    print('Final cost function: ', result.fun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=4, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--compress", default=2, type=int)
    parser.add_argument("--lambdas", default=[0.9, 0.95, 1.0, 1.05, 1.10], type=list)
    args = parser.parse_args()
    main(**vars(args))
