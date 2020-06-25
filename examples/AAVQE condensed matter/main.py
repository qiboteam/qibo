#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from qibo.models import Circuit, VQE
from qibo import gates
from qibo.hamiltonians import XXZ, Hamiltonian
from qibo.config import matrices


def sz_hamiltonian(nqubits):
    """This function implements a simple sz Hamiltonian, which groundstate
    is the |0> state. The mode uses the Identity and the Z-Pauli matrix, and 
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
    for i in range(nqubits):
        h = 1
        for j in range(nqubits):
            if i == j % nqubits:
                h = np.kron(sz, h)
            else:
                h = np.kron(eye, h)
        sz_sum.hamiltonian += h
    sz_sum.hamiltonian = -1*sz_sum.hamiltonian
    return sz_sum


def ising(nqubits, lamb=1.0):
    """This function implements the Ising model. The mode uses the Identity
    and the Z-Pauli and X-Pauli matrices, and builds the final Hamiltonian:

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
    """This function implements the variational quantum circuit.

    Args:
        theta (array): values of the initial parameters.
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


def AAVQE(nqubits, layers, maxsteps, T_max, initial_parameters, easy_hamiltonian, problem_hamiltonian):
    """This function implements the Adiabatically Assisted Variational Quantum
    Eigensolver (AAVQE), and returns the best set of parameters and the
    groundstate energy of the problem Hamiltonian.

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
    

#Define the arguments needed for the AAVQE
nqubits = 6
layers  = 2
maxsteps = 5000
T_max = 5
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