#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:39:11 2020

@author: carlos
"""

import numpy as np
from qibo.models import Circuit
from qibo import gates
from qibo.hamiltonians import Hamiltonian
from qibo import matrices
from scipy.optimize import minimize
from qibo import K
import argparse


def multikron(matrices):
    h = 1
    for m in matrices:
        h = np.kron(h, m)
    return h


def ising(nqubits, lamb=1.0):
    """Implements the Ising model. The function uses the Identity and the
    Z-Pauli and X-Pauli matrices, and builds the final Hamiltonian:

    .. math::
        H = \sum_{i=0}^{nqubits} sz_i sz_{i+1} + 
                                    \\lamb \cdot \sum_{i=0}^{nqubits} sx_i.

    Args:
        nqubits (int): number of quantum bits.
        lamb (float): coefficient for the X component (default 1.0).
        
    Returns:
        ``Hamiltonian`` object for the Ising model.
    """
    ising = Hamiltonian(nqubits)
    eye = matrices.I
    sz = matrices.Z
    sx = matrices.X
    ising.hamiltonian = sum(multikron((sz if i in {j % nqubits, (j+1) % nqubits} else eye 
                                   for j in range(nqubits))) for i in range(nqubits))
    ising.hamiltonian += lamb * sum(multikron((sx if i == j % nqubits else eye 
                                           for j in range(nqubits))) for i in range(nqubits))
    return ising


def encoder_hamiltonian(nqubits, ncompress):
    """Implements the Encoding hamiltonian. The function uses the Identity and
    Z-Pauli matrices:

    .. math::
        H = 1/2 * \sum_{i=0}^{ncompress} 1 - sz_i

    Args:
        nqubits (int): number of quantum bits.
        ncompress (int): number of compressed/discarded qubits.
        
    Returns:
        ``Hamiltonian`` object for the Encoding hamiltonian.
    """
    encoder = Hamiltonian(nqubits)
    encoder.hamiltonian = 0
    eye = matrices.I
    sz = matrices.Z
    for i in range(ncompress):
        h = 1
        for j in range(nqubits):
            if i == j:
                h = np.kron(sz, h)
            else:
                h = np.kron(eye, h)
        encoder.hamiltonian -= h
    
    h = 1
    for i in range(nqubits):
        h = np.kron(eye, h)        
    encoder.hamiltonian += ncompress*h
    encoder.hamiltonian /= 2
    return encoder


def main(nqubits, layers, compress, lambdas):
    
    def cost_function(params):
        """Evaluates the cost function to be minimized.
        
        Args:
            params (array or list): values of the parameters.
            
        Returns:
            Value of the cost function.
        """        
        def ansatz(theta):
            """Implements the variational quantum circuit.
        
            Args:
                theta (array or list): values of the initial parameters.
                
            Returns:
                Circuit that implements the variational ansatz.
            """
            c = Circuit(nqubits)
            index = 0
            for l in range(layers):
                for q in range(nqubits):
                    c.add(gates.RY(q, theta[index]))
                    index+=1
                for q in range(0, nqubits-1, 2):
                    c.add(gates.CZ(q, q+1))
                for q in range(nqubits):
                    c.add(gates.RY(q, theta[index]))
                    index+=1
                for q in range(1, nqubits-2, 2):
                    c.add(gates.CZ(q, q+1))
                c.add(gates.CZ(0, nqubits-1))
            for q in range(nqubits):
                c.add(gates.RY(q, theta[index]))
                index+=1
            return c
        
        cost=0
        for i in range(len(ising_groundstates)):
            final_state = ansatz(params).execute(np.copy(ising_groundstates[i]))
            cost += encoder.expectation(final_state).numpy().real
    
        global ii
        if ii%50 == 0:
            print(ii, cost/len(ising_groundstates))
        ii = ii+1
    
        return cost/len(ising_groundstates)
    
    ii=0
    nparams = 2 * nqubits * layers + nqubits
    initial_params = np.random.uniform(0, 2*np.pi, nparams)
    encoder = encoder_hamiltonian(nqubits,compress)
    
    ising_groundstates = []
    for lamb in lambdas:
        ising_ham = ising(nqubits, lamb=lamb)
        ising_groundstates.append(ising_ham.eigenvectors()[0])        

    result = minimize(cost_function, initial_params, method='L-BFGS-B', options = {'maxiter' : 0.5e3, 'maxfun': 0.5e3})
    print('Final parameters: ',result.x)
    print('Final cost function: ',result.fun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=4, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--compress", default=2, type=int)
    parser.add_argument("--lambdas", default=[0.9, 0.95, 1.0, 1.05, 1.10], type=list)
    args = parser.parse_args()
    main(**vars(args))