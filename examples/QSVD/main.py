#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from qsvd import QSVD
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", help="Number of qubits", type=int)
parser.add_argument("--subsize", help="Subsize of the bipartition with qubits 0,1,...,subzize-1", type=int)
parser.add_argument("--nlayers", help="Number of layers of the variational circuit", type=int)
parser.add_argument("--nshots", default=10000, help="Number of shots used when sampling the circuit", type=int)
parser.add_argument("--RY", default=False, help="Use Ry rotations or RxRzRx rotations in the ansatz", type=bool)
parser.add_argument("--method", default='Powell', help="Classical otimizer employed", type=str)



def main(nqubits, subsize, nlayers, nshots, RY, method):

    # We initialize the QSVD

    Qsvd = QSVD(nqubits, subsize)

    # We choose an intial random state

    initial_state = np.random.rand(2**nqubits)
    initial_state = initial_state / np.linalg.norm(initial_state)

    # We choose initial random parameters

                        # if Rx,Rz,Rx rotations are employed in the anstaz
    initial_parameters = 2*np.pi * np.random.rand(6*nqubits*nlayers+3*nqubits) 

                             # if Ry rotations are employed in the anstaz
    #initial_parameters = 2*np.pi * np.random.rand(2*Nqubits*Nlayers+Nqubits) 

    # We train the QSVD

    cost_function, optimal_angles = Qsvd.minimize(initial_parameters, nlayers, init_state=initial_state,
                                              nshots=10000, RY=False, method='Powell')

    # We use the optimal angles to compute the Schmidt coefficients of the bipartion

    Schmidt_coefficients = Qsvd.Schmidt_coeff(optimal_angles, nlayers, initial_state)
    print('Schmidt coefficients: ', Schmidt_coefficients)

    # We compute the von Neumann entropy using the Schmidt coefficients

    VonNeumann_entropy = Qsvd.VonNeumann_entropy(optimal_angles, nlayers, initial_state)
    print('Von Neumann entropy: ', VonNeumann_entropy)




if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)