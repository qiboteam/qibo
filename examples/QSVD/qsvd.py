#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:41:47 2020

@author: diego
"""

import numpy as np
from qibo.models import Circuit
from qibo import gates



##########################
# QUANTUM CIRCUIT ANSATZ #
##########################
    
def ansatz_RY(theta, sub_size):
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
        for q in range(0, sub_size-1, 2): # U
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        for q in range(sub_size, nqubits-1, 2): # V
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        # Ry rotations   `
        for q in range(nqubits):
            
            c.add(gates.RY(q, theta[index]))
            index+=1
            
        # CZ gates
        for q in range(1, sub_size-1, 2): # U
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        c.add(gates.CZPow(0, sub_size-1, np.pi))
        
        for q in range(sub_size+1, nqubits-1, 2): # V
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        c.add(gates.CZPow(sub_size, nqubits-1, np.pi))
        
    # Final Ry rotations    
    for q in range(nqubits):
        
        c.add(gates.RY(q, theta[index]))
        #c.add(gates.M(q))
        index+=1    
 
    return c


def ansatz(theta, sub_size):
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
        for q in range(0, sub_size-1, 2): # U
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        for q in range(sub_size, nqubits-1, 2): #V
            
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
        for q in range(1, sub_size-1, 2): # U
            
            c.add(gates.CZPow(q, q+1, np.pi))
            
        c.add(gates.CZPow(0, sub_size-1, np.pi))
        
        for q in range(sub_size+1, nqubits-1, 2): # V
            
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
        
        c.add(gates.M(q)) # Measurements


    return c


#####################################
# QUANTUM SINGULAR VALUE DECOMPOSER #
####################################
    
def Hamming(string1, string2):
    
    l1 = len(string1)
    l2 = len(string2)    
    length = min(l1, l2)
    
    h = 0
    
    for i in range(length):
        
        if string1[i] != string2[i]:
            
            h+=1
            
    h += abs(l1-l2)
    
    return h

    
def QSVD_circuit(theta, sub_size, RY=False):
    
    if not RY:
        
        circuit = ansatz(theta, sub_size)
    
    elif RY:
        
        circuit = ansatz_RY(theta, sub_size)
    
    else:
        raise ValueError('RY must take a Boolean value')
        
    return circuit
        


def QSVD(theta, initial_state=None, subsize=nqubits//2, RY=False, nshots=10000, display=True):
    
    Circuit = QSVD_circuit(theta, subsize, RY=RY)
    Circuit = Circuit(initial_state, nshots)
    
    result = circ.frequencies(binary=True)

    loss = 0
    
    for bit_string in result:
        
        a = bit_string[:subsize]
        b = bit_string[subsize:]
        
        if a != b:
            
            loss += Hamming(a,b) * result[bit_string]
            
    if display:
        print(loss/nshots)
        
    return loss/nshots


def Schmidt_coeff(theta, initial_state, subsize=nqubits//2, nshots=10000, RY=False):
    
    qsvd = QSVD_circuit(theta, subsize, RY=RY)
    qsvd = qsvd(initial_state, nshots)
    
    result = qsvd.frequencies(binary=True)
    
    small = min(subsize, nqubits-subsize)
    
    Schmidt = []
    for i in range(2**small):
        
        bit_string = bin(i)[2:].zfill(small)
               
        Schmidt.append(result[2*bit_string])
      
    Schmidt = np.array(sorted(Schmidt, reverse=True))
    
    return Schmidt / np.linalg.norm(Schmidt)











def QSVD_SWAP(theta, initial_state):
    
    pass


def QSVD_Encoder(theta, initial_state)

    pass
        
        
        
        


