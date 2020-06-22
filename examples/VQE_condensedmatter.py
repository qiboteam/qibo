#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from qibo.models import Circuit, VQE
from qibo import gates
from qibo.hamiltonians import XXZ, Hamiltonian
from qibo.config import matrices

def sz_hamiltonian(nqubits):
    sz_prod = Hamiltonian(nqubits)  
    sz_prod.hamiltonian = 1
    sz = matrices._npZ()
    for i in range(nqubits):
        sz_prod.hamiltonian = np.kron(sz, sz_prod.hamiltonian)
    sz_prod.hamiltonian = -1*sz_prod.hamiltonian
    return sz_prod

def Ising(nqubits, lamb=1.0):
    Ising_hamiltonian = Hamiltonian(nqubits)  
    Ising_hamiltonian.hamiltonian = 0
    eye = matrices._npI()
    sz = matrices._npZ()
    sx = matrices._npX()
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


nqubits = 6
layers  = 2

easy_hamiltonian = sz_hamiltonian(nqubits=nqubits)
#problem_hamiltonian = XXZ(nqubits=nqubits)
problem_hamiltonian = Ising(nqubits=nqubits)

maxsteps = 5000
T_max = 5
initial_parameters = np.random.uniform(0, 0.01,
                                        2*nqubits*layers + nqubits)

for t in range(T_max+1):
    s = t/T_max
    print('s =',s)
    hamiltonian =  (1-s)*easy_hamiltonian + s*problem_hamiltonian
    v = VQE(ansatz, hamiltonian)
    best, params = v.minimize(initial_parameters, method='Nelder-Mead', options={'maxfev': maxsteps}, compile=False)
    initial_parameters = params
    
print('Final parameters: ', params)
print('Final energy: ', best)


