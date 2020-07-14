#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from qsvd import QSVD
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=6, help="Number of qubits", type=int)
parser.add_argument("--subsize", default=3, help="Subsize of the bipartition with qubits 0,1,...,subzize-1", type=int)
parser.add_argument("--nlayers", default=4, help="Number of layers of the variational circuit", type=int)
parser.add_argument("--nshots", default=100000, help="Number of shots used when sampling the circuit", type=int)
parser.add_argument("--RY", action="store_true", help="Use Ry rotations or RxRzRx rotations in the ansatz")
parser.add_argument("--method", default='Powell', help="Classical otimizer employed", type=str)


def main(nqubits, subsize, nlayers, nshots, RY, method):

    # We initialize the QSVD
    Qsvd = QSVD(nqubits, subsize)

    # We choose an initial random state
    initial_state = np.random.rand(2**nqubits)/2 + 1j*np.random.rand(2**nqubits)/2
    initial_state = initial_state / np.linalg.norm(initial_state) 
    print('Initial random state: ', initial_state)
    
    # We choose initial random parameters
    if RY==False: # if Rx,Rz,Rx rotations are employed in the anstaz                
        initial_parameters = 2*np.pi * np.random.rand(6*nqubits*nlayers+3*nqubits) 
        
    elif RY==True: # if Ry rotations are employed in the anstaz                           
        initial_parameters = 2*np.pi * np.random.rand(2*nqubits*nlayers+nqubits) 
        
    # We train the QSVD
    cost_function, optimal_angles = Qsvd.minimize(initial_parameters, nlayers, init_state=initial_state,
                                              nshots=nshots, RY=RY, method=method)

    # We use the optimal angles to compute the Schmidt coefficients of the bipartion
    Schmidt_coefficients = Qsvd.Schmidt_coeff(optimal_angles, nlayers, initial_state)
    print('Schmidt coefficients: ', Schmidt_coefficients)

    # We compute the von Neumann entropy using the Schmidt coefficients
    VonNeumann_entropy = Qsvd.VonNeumann_entropy(optimal_angles, nlayers, initial_state)
    print('Von Neumann entropy: ', VonNeumann_entropy)



if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)