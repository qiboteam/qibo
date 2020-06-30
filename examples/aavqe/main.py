#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qibo.models import Circuit, VQE
from qibo import gates, models
from qibo.hamiltonians import XXZ, Hamiltonian
from qibo import matrices
import argparse


def multikron(matrices):
    h = 1
    for m in matrices:
        h = np.kron(h, m)
    return h


def sz_hamiltonian(nqubits):
    """Implements a simple sz Hamiltonian for which groundstate is the
    |0> state. The mode uses the Identity and the Z-Pauli matrix, and 
    builds the Hamiltonian:

    .. math::
        H = - \sum_{i=0}^{nqubits} sz_i.

    Args:
        nqubits (int): number of quantum bits.
    """
    sz_sum = Hamiltonian(nqubits)  
    sz_sum.hamiltonian = 0
    eye = matrices.I
    sz = matrices.Z
    sz_sum.hamiltonian = sum(multikron((sz if i == j % nqubits else eye for j in
                                        range(nqubits))) for i in range(nqubits))
    sz_sum.hamiltonian = -1*sz_sum.hamiltonian
    return sz_sum


def ising(nqubits, lamb=1.0):
    """Implements the Ising model. The mode uses the Identity and the
    Z-Pauli and X-Pauli matrices, and builds the final Hamiltonian:

    .. math::
        H = \sum_{i=0}^{nqubits} sz_i sz_{i+1} + 
                                    \\lamb \cdot \sum_{i=0}^{nqubits} sx_i.

    Args:
        nqubits (int): number of quantum bits.
        lamb (float): coefficient for the X component (default 1.0).
    """
    Ising_hamiltonian = Hamiltonian(nqubits)  
    Ising_hamiltonian.hamiltonian = 0
    eye = matrices.I
    sz = matrices.Z
    sx = matrices.X
    for i in range(nqubits):
        h = 1
        for j in range(nqubits):
            if i == j % nqubits or i == (j+1) % nqubits:
                h = np.kron(sz, h)
            else:
                h = np.kron(eye, h)
        Ising_hamiltonian.hamiltonian += h
    for i in range(nqubits):
        h = 1
        for j in range(nqubits):
            if i == j % nqubits:
                h = np.kron(sx, h)
            else:
                h = np.kron(eye, h)
        Ising_hamiltonian.hamiltonian += lamb*h
    return Ising_hamiltonian


def ansatz(theta):
    """Implements the variational quantum circuit.

    Args:
        theta (array): values of the initial parameters.
    """
    nqubits = args.nqubits
    layers = args.layers
    theta_iter = iter(theta)
    pairs1 = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    pairs2 = list((i, i + 1) for i in range(1, nqubits - 2, 2))
    pairs2.append((0, nqubits - 1))
    c = models.Circuit(nqubits)
    for l in range(layers):
        # parameters for one-qubit gates before CZ layer
        theta_map1 = {i: next(theta_iter) for i in range(nqubits)}
        # parameters for one-qubit gates after CZ layer
        theta_map2 = {i: next(theta_iter) for i in range(nqubits)}
        c.add(gates.VariationalLayer(pairs1, gates.RY, gates.CZ, theta_map1, theta_map2))
        # this ``VariationalLayer`` includes two layers of RY gates with a
        # layer of CZ in the middle.
        # We have to add an additional CZ layer manually:
        c.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
        c.add(gates.CZ(0, nqubits - 1))
    c.add((gates.RY(i, next(theta_iter)) for i in range(nqubits)))
    return c


def AAVQE(nqubits, layers, maxsteps, T_max, initial_parameters, easy_hamiltonian, problem_hamiltonian):
    """Implements the Adiabatically Assisted Variational Quantum Eigensolver
    (AAVQE), and returns the best set of parameters and the groundstate 
    energy of the problem Hamiltonian.

    Args:
        nqubits (int): number of quantum bits.
        layers (float): number of ansatz layers.
        maxsteps (int): number of maximum steps on each adiabatic path.
        T_max (int): number of maximum adiabatic paths.
        initial_parameters (array): values of the initial parameters.
        easy_hamiltonian (qibo.hamiltonians): initial hamiltonian object.
        problem_hamiltonian (qibo.hamiltonians): problem hamiltonian object.
    """
    for t in range(T_max+1):
        s = t/T_max
        print('s =',s)
        hamiltonian =  (1-s)*easy_hamiltonian + s*problem_hamiltonian
        v = VQE(ansatz, hamiltonian)
        best, params = v.minimize(initial_parameters, method='Nelder-Mead', options={'maxfev': maxsteps}, compile=False)
        initial_parameters = params
    return best, params
    

def main(nqubits, layers, maxsteps, T_max):
    initial_parameters = np.random.uniform(0, 0.01,
                                        2*nqubits*layers + nqubits)

    #Define the easy Hamiltonian and the problem Hamiltonian.
    easy_hamiltonian = sz_hamiltonian(nqubits=nqubits)
    problem_hamiltonian = ising(nqubits=nqubits)
    
    #Run the AAVQE
    best, params = AAVQE(nqubits, layers, maxsteps, T_max, initial_parameters, easy_hamiltonian, problem_hamiltonian)
    print('Final parameters: ', params)
    print('Final energy: ', best)
    
    #We compute the difference from the exact value to check performance
    eigenvalue = problem_hamiltonian.eigenvalues()
    print('Difference from exact value: ',best - eigenvalue[0].numpy().real)
    print('Log difference: ',-1*np.log10(best - eigenvalue[0].numpy().real))
 
    
parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=6, type=int)
parser.add_argument("--layers", default=2, type=int)
parser.add_argument("--maxsteps", default=5000, type=int)
parser.add_argument("--T_max", default=5, type=int)
args = parser.parse_args()
main(args.nqubits, args.layers, args.maxsteps, args.T_max)