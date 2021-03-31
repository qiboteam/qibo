#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from qclassifier import QuantumClassifer
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--nclasses", default=3, help="Number of classes to be classified", type=int)
parser.add_argument("--nqubits", default=4, help="Number of qubits", type=int)
parser.add_argument("--nlayers", default=5, help="Number of layers of the variational circuit", type=int)
parser.add_argument("--nshots", default=int(1e5), help="Number of shots used when sampling the circuit", type=int)
parser.add_argument("--training", action="store_true", help="Train the quantum classifier or ortherwise use precomputed angles for the circuit")
parser.add_argument("--RxRzRx", action="store_true", help="Use Ry rotations or RxRzRx rotations in the ansatz")
parser.add_argument("--method", default='Powell', help="Classical optimizer employed", type=str)


def main(nclasses, nqubits, nlayers, nshots, training, RxRzRx, method):
      
    # We initialize the quantum classifier
    RY = not RxRzRx
    qc = QuantumClassifer(nclasses, nqubits, nlayers, RY=RY)
    
    # We load the iris data set
    data = open('data/iris.data')
    data = data.readlines()
    data = [i.split(',') for i in data]

    for i in range(len(data)):    
        del data[i][-1]  # We delete text labels from the kets        
        data[i] += [0]*(2**nqubits-4) # We pad with zeros
        data[i] = np.float32(data[i]) # We transform strings to floats   
        # We re-scale each feature of the data
        data [i][0] /= 7.9
        data[i][1] /= 4.4
        data[i][2] /= 6.9
        data[i][3] /= 2.5
        data[i] = data[i]**2
        data[i] /= np.linalg.norm(data[i]) # We normalize the ket
    
    data = np.float32(np.array(data)) 
         
    # We define our training set (both kets and labels)
    data_train = np.concatenate((data[0:35], data[50:85], data[100:135]))
    labels_train = np.array([[1,1]]*35 + [[1,-1]]*35 + [[-1,1]]*35)
    
    # We load pre-trained angles or actually train the Quantum_Classifer
    if not training:
        if RY:
            try:
                optimal_angles = np.load('data/optimal_angles_ry_{}q_{}l.npy'.format(nqubits,nlayers))
            except:
                raise FileNotFoundError('There are no pre-trained angles saved for this choice of nqubits, nlayers and type of ansatz.')
        else:
            try:
                optimal_angles = np.load('data/optimal_angles_rxrzrx_{}q_{}l.npy'.format(nqubits,nlayers))
            except:
                raise FileNotFoundError('There are no pre-trained angles saved for this choice of nqubits, nlayers and type of ansatz.')
    else:
        # We choose initial random parameters (execpt for the biases, that we set to zero)
        measured_qubits = int(np.ceil(np.log2(nclasses))) 
        if RY:  # if Ry rotations are employed in the ansatz
            initial_parameters = 2*np.pi * np.random.rand(2*nqubits*nlayers+nqubits+measured_qubits)
            for bias in range(measured_qubits):       
                initial_parameters[bias] = 0.                
            print('Training classifier...') 
            cost_function, optimal_angles = qc.minimize(initial_parameters, data_train, labels_train,
                                              nshots=nshots, method=method)
            np.save('data/optimal_angles_ry_{}q_{}l.npy'.format(nqubits,nlayers), optimal_angles)
        else:  # if RxRzRx rotations are employed in the ansatz
            initial_parameters = 2*np.pi * np.random.rand(6*nqubits*nlayers+3*nqubits+measured_qubits)           
            for bias in range(measured_qubits):       
                initial_parameters[bias] = 0.                
            print('Training classifier...') 
            cost_function, optimal_angles = qc.minimize(initial_parameters, data_train, labels_train,
                                              nshots=nshots, method=method)
            np.save('data/optimal_angles_rxrzrx_{}q_{}l.npy'.format(nqubits,nlayers), optimal_angles)
        
    # We define our test set (both kets and labels)
    data_test = np.concatenate((data[35:50], data[85:100], data[135:150]))  
    labels_test = [[1,1]]*15 + [[1,-1]]*15 + [[-1,1]]*15
    
    # We run an accuracy check for the training and the test sets
    predictions_train = [qc.Predictions(qc.Classifier_circuit(optimal_angles), optimal_angles, init_state=ket, nshots=nshots) 
                            for ket in data_train]   
    predictions_test = [qc.Predictions(qc.Classifier_circuit(optimal_angles), optimal_angles, init_state=ket, nshots=nshots)
                        for ket in data_test]
    
    print('Train set | # Clases: {} | # Qubits: {} | # Layers: {} | Accuracy: {}'.format(nclasses, nqubits, nlayers, qc.Accuracy(labels_train, predictions_train)))
    print('Test set  | # Clases: {} | # Qubits: {} | # Layers: {} | Accuracy: {}'.format(nclasses, nqubits, nlayers, qc.Accuracy(labels_test, predictions_test)))


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)