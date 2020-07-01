#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from qsvd import QSVD

# We set the number of qubits and size of the subsystem

Nqubits = 6
Subsize = 3

# We initialize the QSVD

Qsvd = QSVD(Nqubits, Subsize)

# We choose an intial random state

import numpy as np

initial_state = np.random.rand(2**Nqubits)
initial_state = initial_state / np.linalg.norm(initial_state)


# We set the number of layers and choose initial random parameters

Nlayers = 4

                        # if Rx,Rz,Rx rotations are employed in the anstaz
initial_parameters = 2*np.pi * np.random.rand(16*Nqubits*Nlayers+3*Nqubits) 

                             # if Ry rotations are employed in the anstaz
#initial_parameters = 2*np.pi * np.random.rand(2*Nqubits*Nlayers+Nqubits) 


# We train the QSVD

cost_function, optimal_angles = Qsvd.minimize(initial_parameters, Nlayers, init_state=initial_state,
                                              nshots=10000, RY=False, method='Powell')


# We use the optimal angles to compute the Schmidt coefficients of the bipartion

Schmidt_coefficients = Qsvd.Schmidt_coeff(optimal_angles, Nlayers, initial_state)
print(Schmidt_coefficients)

# We compute the von Neumann entropy using the Schmidt coefficients

VonNeumann_entropy = -np.sum(Schmidt_coefficients * np.log2(Schmidt_coefficients))
print(VonNeumann_entropy)


