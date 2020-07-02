#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from qibo.models import Circuit
from qibo import gates


class QSVD():
    """
    Class for the Quantum Singular Value Decomposer variational algorithm
    
    Args: nqubits: number of qubits
    
          subsize: size of the subsystem with qubits 0,1,...,sub_size-1
    """
    
    def __init__(self, nqubits, subsize):
        
        self.nqubits = nqubits
        self.subsize = subsize
        self.subsize2 = nqubits - subsize
        
        
##########################
# QUANTUM CIRCUIT ANSATZ #
##########################

    def ansatz_RY(self, theta, nlayers):
        """
        Args: theta: list or numpy.array with the angles to be used in the circuit
        
              nlayers: number of layers of the varitional ansatz
        
        Returns: qibo.tensorflow.circuit.TensorflowCircuit with the ansatz to be used in the variational circuit
        """             
        
        c = Circuit(self.nqubits)    
    
        index = 0
        for l in range(nlayers):
            
            # Ry rotations
            for q in range(self.nqubits):
            
                c.add(gates.RY(q, theta[index]))
                index+=1
        
            # CZ gates
            for q in range(0, self.subsize-1, 2): # U
            
                c.add(gates.CZ(q, q+1))
            
            for q in range(self.subsize, self.nqubits-1, 2): # V
            
                c.add(gates.CZ(q, q+1))
            
            # Ry rotations   `
            for q in range(self.nqubits):
            
                c.add(gates.RY(q, theta[index]))
                index+=1
            
            # CZ gates
            for q in range(1, self.subsize-1, 2): # U
            
                c.add(gates.CZ(q, q+1))
                    
            for q in range(self.subsize+1, self.nqubits-1, 2): # V
            
                c.add(gates.CZ(q, q+1))
                
        # Final Ry rotations    
        for q in range(self.nqubits): 
            
            c.add(gates.RY(q, theta[index]))
            index+=1    
                
        # Measurements
        
        small = min(self.subsize, self.subsize2)            
        for q in range(small):
            
            c.add(gates.M(q))        
            c.add(gates.M(q+self.subsize))
        
        return c


    def ansatz(self, theta, nlayers):
        """        
        Args: theta: list or numpy.array with the angles to be used in the circuit
        
              nlayers: number of layers of the varitional ansatz
        
        Returns: qibo.tensorflow.circuit.TensorflowCircuit with the ansatz to be used in the variational circuit
        """
 
        c = Circuit(self.nqubits)
    
        index = 0
        for l in range(nlayers):
        
            # Rx,Rz,Rx rotations
            for q in range(self.nqubits):
            
                c.add(gates.RX(q, theta[index]))
                index+=1
            
                c.add(gates.RZ(q, theta[index]))
                index+=1
            
                c.add(gates.RX(q, theta[index]))
                index+=1
            
            # CZ gates
            for q in range(0, self.subsize-1, 2): # U
            
                c.add(gates.CZ(q, q+1))
            
            for q in range(self.subsize, self.nqubits-1, 2): #V
            
                c.add(gates.CZ(q, q+1))
            
            # Rx,Rz,Rx rotations   `
            for q in range(self.nqubits):
            
                c.add(gates.RX(q, theta[index]))
                index+=1
                c.add(gates.RZ(q, theta[index]))
                index+=1
            
                c.add(gates.RX(q, theta[index]))
                index+=1
            
            # CZ gates
            for q in range(1, self.subsize-1, 2): # U
            
                c.add(gates.CZ(q, q+1))
                    
            for q in range(self.subsize+1, self.nqubits-1, 2): # V
            
                c.add(gates.CZ(q, q+1))
        
        # Final Rx,Rz,Rx rotations    
        for q in range(self.nqubits): 
        
            c.add(gates.RX(q, theta[index]))
            index+=1
            
            c.add(gates.RZ(q, theta[index]))
            index+=1
            
            c.add(gates.RX(q, theta[index]))
            index+=1
          
        # Measurements    
            
        small = min(self.subsize, self.subsize2)            
        for q in range(small):
            
            c.add(gates.M(q))        
            c.add(gates.M(q+self.subsize))

        return c


#####################################
# QUANTUM SINGULAR VALUE DECOMPOSER #
####################################

    def QSVD_circuit(self, theta, nlayers, RY=False):
        """
        Args: theta: list or numpy.array with the angles to be used in the circuit
        
              nlayers: number of layers of the varitional ansatz
              
              RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                  if False, parameterized Ry gates are used in the circuit (default=False)
        
        Returns: qibo.tensorflow.circuit.TensorflowCircuit with the variational circuit for the QSVD
        """
    
        if not RY:
        
            circuit = self.ansatz(theta, nlayers)
    
        elif RY:
        
            circuit = self.ansatz_RY(theta, nlayers)
    
        else:
            raise ValueError('RY must take a Boolean value')
        
        
        return circuit
        


    def QSVD_cost(self, theta, nlayers, init_state=None, nshots=10000, RY=False):
        """
        Args: theta: list or numpy.array with the angles to be used in the circuit
        
              nlayers: number of layers of the varitional ansatz
              
              init_state: numpy.array with the quantum state to be Schmidt-decomposed
              
              nshots: int number of runs of the circuit during the sampling process (default=10000)
          
              RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                  if False, parameterized Ry gates are used in the circuit (default=False)
    
        Returns: numpy.float32 with the value of the cost function for the QSVD with angles theta
        """
        
        def Hamming(string1, string2):
            """
            Args: two strings to be compared
    
            Returns: Hamming distance of the strings
            """  
            
            return sum(q1 != q2 for q1,q2 in zip(string1,string2))
    
    
        Circuit = self.QSVD_circuit(theta, nlayers, RY=RY)
        Circuit = Circuit(init_state, nshots)
        result = Circuit.frequencies(binary=True)

        loss = 0
            
        for bit_string in result:
            
            a = bit_string[:self.subsize2]
            b = bit_string[self.subsize2:]
            
            loss += Hamming(a,b) * result[bit_string]
    
        loss = loss/nshots
        print(loss)
        
             
        return loss
    
    
    def minimize(self, init_theta, nlayers, init_state=None, nshots=10000,
                  RY=False, method='Powell'):
        """
        Args: theta: list or numpy.array with the angles to be used in the circuit
        
              nlayers: number of layers of the varitional ansatz
              
              init_state: numpy.array with the quantum state to be Schmidt-decomposed
          
              nshots: int number of runs of the circuit during the sampling process (default=10000)
          
              RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                  if False, parameterized Ry gates are used in the circuit (default=False)
                  
             method: 'classical optimizer for the minimization. All methods from scipy.optimize.minmize are suported (default='Powell')
        
        Returns: numpy.float64 with value of the minimum found, numpy.ndarray with the optimal angles        
        """
        
        from scipy.optimize import minimize
        
        result = minimize(self.QSVD_cost, init_theta, args=(nlayers,init_state,nshots,RY), method=method)
        loss = result.fun
        optimal_angles = result.x
        
        
        return loss, optimal_angles
        


    def Schmidt_coeff(self, theta, nlayers, init_state, nshots=10000, RY=False):
        """
        Args: theta: list or numpy.array with the angles to be used in the circuit
        
              nlayers: number of layers of the varitional ansatz
              
              init_state: numpy.array with the quantum state to be Schmidt-decomposed
          
              nshots: int number of runs of the circuit during the sampling process (default=10000)
          
              RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                  if False, parameterized Ry gates are used in the circuit (default=False)
        
        Returns: numpy.array with the Schmidt coefficients given by the QSVD, in decreasing order        
        """

        Qsvd = self.QSVD_circuit(theta, nlayers, RY=RY)
        Qsvd = Qsvd(init_state, nshots)
    
        result = Qsvd.frequencies(binary=True)
        small = min(self.subsize, self.subsize2)
    
        Schmidt = []
        for i in range(2**small):
        
            bit_string = bin(i)[2:].zfill(small)
            Schmidt.append(result[2*bit_string])
        
        Schmidt = np.array(sorted(Schmidt, reverse=True))
        Schmidt = np.sqrt(Schmidt / nshots)
    
        return Schmidt / np.linalg.norm(Schmidt)


    def VonNeumann_entropy(self, theta, nlayers, init_state, tol=1e-14, nshots=10000, RY=False):
        """
        Args: theta: list or numpy.array with the angles to be used in the circuit
        
              nlayers: number of layers of the varitional ansatz
              
              init_state: numpy.array with the quantum state to be Schmidt-decomposed
          
              nshots: int number of runs of the circuit during the sampling process (default=10000)
          
              RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                  if False, parameterized Ry gates are used in the circuit (default=False)
        
        Returns: numpy.float64 with the value of the Von Neumann entropy for the given bipartition     
        """
        
        Schmidt = self.Schmidt_coeff(theta, nlayers, init_state, nshots=10000, RY=False)       
        Schmidt = Schmidt**2
        
        non_zero_coeff = np.array([coeff for coeff in Schmidt if coeff>tol])
    
        return  -np.sum(non_zero_coeff * np.log2(non_zero_coeff))