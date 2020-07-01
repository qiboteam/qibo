#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:41:46 2020

@author: diego
"""

import numpy as np
import qibo
from math import ceil
from qibo.models import Circuit
from qibo import gates

##########################
# QUANTUM CIRCUIT ANSATZ #
##########################

def ansatz(theta):
    """  
    Args: theta: list or numpy.array with the angles to be used in the circuit
    
    Returns: qibo.tensorflow.circuit.TensorflowCircuit
    
    Caveat: number of qubits (nqubits) and number of layers (nlayers) must be provided as outer values
    """
    
    c = Circuit(nqubits)
    
    index = 0
    for l in range(nlayers):
        
        # Ry rotations
        for q in range(nqubits):
            
            c.add(gates.RY(q, theta[index]))
            index+=1
        
        # CZ gates
        for q in range(0, nqubits-1, 2):
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        # Ry rotations   `
        for q in range(nqubits):
            
            c.add(gates.RY(q, theta[index]))
            index+=1
            
        # CZ gates
        for q in range(1, nqubits-2, 2):
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        c.add(gates.CZPow(0, nqubits-1, np.pi))
        
    # Final Ry rotations    
    for q in range(nqubits):
        
        c.add(gates.RY(q, theta[index]))
        index+=1 

    return c


######################
# QUANTUM CLASSIFIER #
######################

def Q_classifierCircuit(theta):
    """    
    Args: theta: list or numpy.array with the biases and the angles to be used in the circuit
    
    Returns: qibo.tensorflow.circuit.TensorflowCircuit with the variational circuit for angles "theta"
    
    Caveat: number of qubits (nqubits) and number of layers (nlayers) and number of classes (nclasses)
            must be provided as outer values
    """
    
    measured_qubits = ceil(np.log2(nclases))
    
    bias = np.array(theta[0:measured_qubits])
    angles = theta[measured_qubits:] 
    
    circuit = ansatz(angles)        
    circuit.add(gates.M(*range(measured_qubits)))
    
    return circuit
    
    
def Q_classifierPredictions(circuit, theta, init_state, nshots=10000):
    """    
    Args: circuit: qibo.tensorflow.circuit.TensorflowCircuit 
                          with the variational circuit for angles "theta"
    
                theta: list or numpy.array with the biases to be used in the circuit
                    
                init_state: numpy.array with the quantum state to be classified
                
                nshots: int number of runs of the circuit during the sampling process (default=10000)
    
    Returns: numpy.array() with predictions for each qubit, for the initial state
    
    Caveat: number of classes (nclasses) must be provided as outer value
    """
    
    measured_qubits = ceil(np.log2(nclases))
    bias = np.array(theta[0:measured_qubits])
    
    circuit = circuit(init_state, nshots)
        
    result = circuit.frequencies(binary=False)
    prediction = np.zeros(measured_qubits)
    
    for qubit in range(measured_qubits):
        
        for clase in range(nclases):
            
            binary = bin(clase)[2:].zfill(measured_qubits)
            
            prediction[qubit] += result[clase] * (1-2*int(binary[-qubit-1]))
    
    return prediction/nshots + bias



def square_loss(labels, predictions):
    """    
    Args: labels: list or numpy.array with the qubit labels of the quantum states to be classified
                    
          predictions: list or numpy.array with the qubit predictions for the quantum states to be classified
    
    Returns: numpy.float32() with the value of the square-loss function
    
    Caveat: number of qubits (nqubits) and number of clases (nclases) must be provided as outer variable
    """
    
    measured_qubits = ceil(np.log2(nclases))

    loss = 0
    for l, p in zip(labels, predictions):
    
        for qubit in range(measured_qubits):
            
            loss += (l[qubit] - p[qubit]) ** 2
    
    return loss / len(labels)



def Q_classifier(theta, data=None, labels=None, nshots=10000):
    """    
    Args: theta: list or numpy.array with the biases and the angles to be used in the circuit
    
                data: numpy.array data[page][word]  (this is an array of kets)
        
                labels: list or numpy.array with the labels of the quantum states to be classified
                    
                nshots: int number of runs of the circuit during the sampling process (default=10000)
    
    Returns: numpy.float32() with the value of the square-loss function
    
    Caveat: number of qubits (nqubits) and number of layers (nlayers) and number of classes (nclasses)
            must be provided as outer values
    """
    
    qibo.set_backend("matmuleinsum")
    circ = Q_classifierCircuit(theta)
    circ.compile()
    
    measured_qubits = ceil(np.log2(nclases))
    
    Bias = np.array(theta[0:measured_qubits])
    predictions = np.zeros(shape=(len(data),ceil(np.log2(nclases))))
    
    for i, text in enumerate(data):
                
        predictions[i] = Q_classifierPredictions(circ, Bias, text, nshots)
    
    s = square_loss(labels, predictions)
    print(s)
    return s


################################
# QLASSIFIER'S ACCURACY CHECK #
###############################
    
def accuracy(labels, predictions, sign=True, tolerance=1e-2):
    """
    Args: labels: numpy.array with the labels of the quantum states to be classified
    
          predictions: numpy.array with the predictions for the quantum states classified
                
          sign: if True, labels = np.sign(labels) and predictions = np.sign(predictions) (default=True)
                
          tolerance: tolerance level to consider a prediction correct (default=1e-2)
    
    Returns: float with the proportion of states classified successfully
    """
    
    if sign == True:
        
        labels = np.sign(labels)
        predictions = np.sign(predictions)
    
    accur = 0
    for l, p in zip(labels, predictions):
        
        if np.allclose(l, p, rtol=0., atol=tolerance):
            
            accur += 1
            
    accur = accur / len(labels)
    
    return accur