#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:41:46 2020

@author: diego
"""

import numpy as np
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
    
    Caveat: # of qubits (nqubits) and # of layers (nlayers) must be provided as outer values
    """
    
    c = Circuit(nqubits)
    
    index = 0
    for l in range(nlayers):
        
        # Rx,Rz,Rx rotations
        for q in range(nqubits):
            
            c.add(gates.RX(q, theta[index]))
            index+=1
            
            c.add(gates.RZ(q, theta[index]))
            index+=1
            
            c.add(gates.RX(q, theta[index]))
            index+=1
            
        # CZ gates
        for q in range(0, nqubits-1, 2):
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        # Rx,Rz,Rx rotations   `
        for q in range(nqubits):
            
            c.add(gates.RX(q, theta[index]))
            index+=1
            
            c.add(gates.RZ(q, theta[index]))
            index+=1
            
            c.add(gates.RX(q, theta[index]))
            index+=1
            
        # CZ gates
        for q in range(1, nqubits-2, 2):
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        c.add(gates.CZPow(0, nqubits-1, np.pi))
        
    # Final Rx,Rz,Rx rotations    
    for q in range(nqubits):
        
        c.add(gates.RX(q, theta[index]))
        index+=1
            
        c.add(gates.RZ(q, theta[index]))
        index+=1
            
        c.add(gates.RX(q, theta[index]))
        index+=1
        
        #c.add(gates.M(q)) # Measurements


    return c
"""
def ansatz(theta):
        
    #Parameters: theta: list or numpy.array with the angles to be used in the circuit
    
    #Returns: qibo.tensorflow.circuit.TensorflowCircuit
    
    #Caveat: # of qubits (nqubits) and # of layers (nlayers) must be provided as outer values
    
    
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
"""

######################
# QUANTUM CLASSIFIER #
######################

def Q_classifierCircuit(theta):
    """    
    Parameters: theta: list or numpy.array with the biases and the angles to be used in the circuit
    
    Returns: qibo.tensorflow.circuit.TensorflowCircuit with the variational circuit for angles "theta"
    
    Caveat: # of qubits (nqubits) and # of layers (nlayers) and # of classes (nclasses)
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
    Parameters: circuit: qibo.tensorflow.circuit.TensorflowCircuit 
                          with the variational circuit for angles "theta"
    
                theta: list or numpy.array with the biases to be used in the circuit
                    
                init_state: numpy.array with the quantum state to be classified
                
                nshots: int number of runs of the circuit during the sampling process (default=10000)
    
    Returns: numpy.array() with predictions for each qubit, for the initial state
    
    Caveat: # of classes (nclasses) must be provided as outer value
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
    Parameters: labels: list or numpy.array with the qubit labels of the quantum states to be classified
                    
                predictions: list or numpy.array with the qubit predictions for the quantum states to be classified
    
    Returns: numpy.float32() with the value of the square-loss function
    
    Caveat: # of qubits (nqubits) and # of clases (nclases) must be provided as outer variable
    """
    
    measured_qubits = ceil(np.log2(nclases))

    loss = 0
    for l, p in zip(labels, predictions):
    
        for qubit in range(measured_qubits):
            
            loss += (l[qubit] - p[qubit]) ** 2
    
    return loss / len(labels)



def Q_classifier(theta, data=None, labels=None, nshots=10000):
    """    
    Parameters: theta: list or numpy.array with the biases and the angles to be used in the circuit
    
                data: numpy.array data[page][word]  (this is an array of kets)
        
                labels: list or numpy.array with the labels of the quantum states to be classified
                    
                nshots: int number of runs of the circuit during the sampling process (default=10000)
    
    Returns: numpy.float32() with the value of the square-loss function
    
    Caveat: # of qubits (nqubits) and # of layers (nlayers) and # of classes (nclasses)
            must be provided as outer values
    """
    
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
    Parameters: labels: numpy.array with the labels of the quantum states to be classified
    
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



from scipy.optimize import minimize

#Data = np.loadtxt('Data.txt')
#Labels = np.loadtxt('Labels.txt')

nqubits=6
nclases=4
nlayers=2

#dat = Data(100,[1,2,6,7,14,15,16,19])
dat = Data(100,[1,2,6,7],nqubits=nqubits)

data = dat[0]
#data = np.e**data
#for i in range(len(data)):
    #data[i] /= np.linalg.norm(data[i])
    
Labels = dat[1]

#initial_parameters = 2* np.pi* np.random.rand(2*nlayers*nqubits+nqubits+int(np.log2(nclases)))
initial_parameters = 2* np.pi* np.random.rand(6*nlayers*nqubits+3*nqubits+int(np.log2(nclases)))
#initial_parameters = np.zeros(2*nlayers*nqubits+nqubits+int(np.log2(nclases)))

for q in range(ceil(np.log2(nclases))):
    initial_parameters[q]=0
    
#previous_layer = np.load('7Q_2C_2L_100T_Powell.npy')

#for i,angle in enumerate(previous_layer):
    #initial_parameters[i] = angle

result5 = minimize(Q_classifier, initial_parameters, args=(data, Labels), method='Powell')
print(result5)
np.save('xzx_6Q_4C_2L_100T_Powell.npy',result5.x)

"""
nqubits=6
nclases=8
nlayers=4

initial_parameters = 2* np.pi* np.random.rand(2*nlayers*nqubits+nqubits+int(np.log2(nclases)))

for q in range(ceil(np.log2(nclases))):
    initial_parameters[q]=0

result4 = minimize(Q_classifier, initial_parameters, args=(data, Labels), method='Powell')
print(result4)
np.save('8C_4L_100T_Powell.npy',result4.x)
"""

"""
Angles2 = np.loadtxt('2L_diff_ev.txt')
Q_classifier(Angles2, Data, Labels)

nlayers=3
Angles3 = np.loadtxt('3L_diff_ev.txt')
Q_classifier(Angles3, Data, Labels)
"""

nqubits=6
nclases=4
nlayers=2
tes= Test(40,[1,2,6,7],nqubits=nqubits)
#tes = Test(40,[1,2,6,7,14,15,16,19])
optimal_angles = np.load('xzx_6Q_4C_2L_100T_Powell.npy')

Predictions = [Q_classifierPredictions(Q_classifierCircuit(optimal_angles), optimal_angles, init_state=text) for text in data]
Predictions2 = [Q_classifierPredictions(Q_classifierCircuit(optimal_angles), optimal_angles, init_state=text) for text in tes[0]]

print('# Clases: {} | # Layers: {} | Accuracy: {} | Train Set'.format(nclases, nlayers, accuracy(dat[1],Predictions)))
print('# Clases: {} | # Layers: {} | Accuracy: {} | Test Set'.format(nclases, nlayers, accuracy(tes[1],Predictions2)))

"""
lista=[]      
for i in range(1,54):
    
    lista.append(len(data_class_counter(i)))
print(lista)
print(max(lista))
print(min(lista))
import matplotlib.pyplot as plt

plt.plot(range(1,54),lista,'o')
plt.title('Test Set',fontsize=12)
plt.text(41,830,'max=1187\nmin=0',fontsize=10)
plt.grid()
plt.xlabel('Category')
plt.ylabel('Number of texts')
plt.savefig('Test_categories')
"""    