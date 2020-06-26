#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from qibo.models import Circuit
from qibo import gates


##########################
# QUANTUM CIRCUIT ANSATZ #
##########################
    
def ansatz_RY(theta, sub_size):
    """    
    Args: theta: list or numpy.array with the angles to be used in the circuit
    
    Returns: qibo.tensorflow.circuit.TensorflowCircuit with the ansatz to be used in the variational circuit
    
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
            
            c.add(gates.CZ(q, q+1))
            
        for q in range(sub_size, nqubits-1, 2): # V
            
            c.add(gates.CZ(q, q+1))
            
        # Ry rotations   `
        for q in range(nqubits):
            
            c.add(gates.RY(q, theta[index]))
            index+=1
            
        # CZ gates
        for q in range(1, sub_size-1, 2): # U
            
            c.add(gates.CZ(q, q+1))
            
        c.add(gates.CZ(0, sub_size-1))
        
        for q in range(sub_size+1, nqubits-1, 2): # V
            
            c.add(gates.CZ(q, q+1))
            
        c.add(gates.CZ(sub_size, nqubits-1))
        
    # Final Ry rotations    
    for q in range(nqubits):
        
        c.add(gates.RY(q, theta[index]))
        c.add(gates.M(q))
        index+=1    
 
    return c


def ansatz(theta, sub_size):
    """    
    Args: theta: list or numpy.array with the angles to be used in the circuit
    
    Returns: qibo.tensorflow.circuit.TensorflowCircuit with the ansatz to be used in the variational circuit
    
    Caveat: number of qubits (nqubits) and number of layers (nlayers) must be provided as outer values
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
            
            c.add(gates.CZ(q, q+1))
            
        for q in range(sub_size, nqubits-1, 2): #V
            
            c.add(gates.CZ(q, q+1))
            
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
            
            c.add(gates.CZ(q, q+1))
            
        c.add(gates.CZ(0, sub_size-1))
        
        for q in range(sub_size+1, nqubits-1, 2): # V
            
            c.add(gates.CZ(q, q+1))
            
        c.add(gates.CZ(0, nqubits-1))
        
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
    """
    Args: the two strings to be compared
    
    Returns: Hamming distance of the strings
    """
    
    l1 = len(string1)
    l2 = len(string2)    
    
    h = sum(q1 != q2 for q1,q2 in zip(string1,string2))

    h += abs(l1-l2)
    
    return h

    
def QSVD_circuit(theta, sub_size=None, RY=False):
    """
    Args: theta: list or numpy.array with the angles to be used in the circuit
    
          sub_size: size of the subsystem with qubits 0,1,...,sub_size-1
          
          RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
              if False, parameterized Ry gates are used in the circuit (default=False)
        
    Returns: qibo.tensorflow.circuit.TensorflowCircuit with the variational circuit for the QSVD
    
    Caveat: number of qubits (nqubits) and number of layers (nlayers) must be provided as outer values
    """
    
    if not RY:
        
        circuit = ansatz(theta, sub_size)
    
    elif RY:
        
        circuit = ansatz_RY(theta, sub_size)
    
    else:
        raise ValueError('RY must take a Boolean value')
        
    return circuit
        


def QSVD(theta, subsize=None, init_state=None, nshots=10000, RY=False):
    """
    Args: theta: list or numpy.array with the angles to be used in the circuit
    
          subsize: size of the subsystem with qubits 0,1,...,sub_size-1 in the bipartition of the state
          
          init_state: numpy.array with the quantum state to be Schmidt-decomposed
              
          nshots: int number of runs of the circuit during the sampling process (default=10000)
          
          RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
              if False, parameterized Ry gates are used in the circuit (default=False)
    
    Returns: np.float32 with the value of the cost function for the QSVD with angles theta
        
    Caveat: number of qubits (nqubits) and number of layers (nlayers) must be provided as outer values
    """
    
    Circuit = QSVD_circuit(theta, sub_size=subsize, RY=RY)
    Circuit = Circuit(init_state, nshots)
    result = Circuit.frequencies(binary=True)

    loss = 0
    
    for bit_string in result:
        
        a = bit_string[:subsize]
        b = bit_string[subsize:]
        
        if a != b:
            
            loss += Hamming(a,b) * result[bit_string]
    
    loss = loss/nshots
    print(loss)
        
    return loss


def Schmidt_coeff(theta, subsize, init_state, nshots=10000, RY=False):
    """
    Args: theta: list or numpy.array with the angles to be used in the circuit
    
          subsize: size of the subsystem with qubits 0,1,...,sub_size-1
          
          init_state: numpy.array with the quantum state to be Schmidt-decomposed
          
          nshots: int number of runs of the circuit during the sampling process (default=10000)
          
          RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
              if False, parameterized Ry gates are used in the circuit (default=False)
        
    Returns: np.array with the Schmidt coefficients given by the QSVD, in decreasing order
        
    Caveat: number of qubits (nqubits) and number of layers (nlayers) must be provided as outer values
    """
    
    qsvd = QSVD_circuit(theta, subsize, RY=RY)
    qsvd = qsvd(init_state, nshots)
    
    result = qsvd.frequencies(binary=True)
    small = min(subsize, nqubits-subsize)
    
    Schmidt = []
    for i in range(2**small):
        
        bit_string = bin(i)[2:].zfill(small)
        Schmidt.append(result[2*bit_string])
      
    Schmidt = np.array(sorted(Schmidt, reverse=True))
    
    return Schmidt / np.linalg.norm(Schmidt)











def QSVD_SWAP(theta, initial_state):
    """
    Args:
        
    Returns:
        
    Caveat: number of qubits (nqubits) and number of layers (nlayers) must be provided as outer values
    """
    
    pass


def QSVD_Encoder(theta, initial_state):
    """
    Args:
        
    Returns:
        
    Caveat: number of qubits (nqubits) and number of layers (nlayers) must be provided as outer values
    """

    pass
        
