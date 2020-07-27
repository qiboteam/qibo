#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import qibo
from qibo.models import Circuit
from qibo import gates


class Quantum_Classifer():
    
    def __init__(self, nclasses, nqubits):
        """
        Class for a multi-task variational quantum classifier
    
        Args: 
            nclases: int number of classes to be classified     
            nqubits: int number of qubits employed in the quantum circuit
        """     
        self.nclasses = nclasses        
        self.nqubits = nqubits   
        self.measured_qubits = int(np.ceil(np.log2(self.nclasses)))
        
        if self.nqubits <= 1:
            raise ValueError('nqubits must be larger than 1')
        
    def ansatz(self, theta, nlayers):
        """    
        Args: 
            theta: list or numpy.array with the angles to be used in the circuit     
            nlayers: int number of layers of the varitional circuit ansatz
    
        Returns: 
            qibo.tensorflow.circuit.TensorflowCircuit with the ansatz to be used in the variational circuit
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
            for q in range(0, self.nqubits-1, 2):         
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
            for q in range(1, self.nqubits-1, 2):          
                c.add(gates.CZ(q, q+1))
            
            c.add(gates.CZ(0, self.nqubits-1))
        
        # Final Rx,Rz,Rx rotations    
        for q in range(self.nqubits):       
            c.add(gates.RX(q, theta[index]))
            index+=1           
            c.add(gates.RZ(q, theta[index]))
            index+=1          
            c.add(gates.RX(q, theta[index]))
            index+=1
        
        return c

    def ansatz_RY(self, theta, nlayers):
        """
        Args: 
            theta: list or numpy.array with the angles to be used in the circuit
            nlayers: int number of layers of the varitional circuit ansatz
    
        Returns: 
            qibo.tensorflow.circuit.TensorflowCircuit with the ansatz to be used in the variational circuit
        """ 
        c = Circuit(self.nqubits)
    
        index = 0
        for l in range(nlayers):
        
            # Ry rotations
            for q in range(self.nqubits):           
                c.add(gates.RY(q, theta[index]))
                index+=1
        
            # CZ gates
            for q in range(0, self.nqubits-1, 2):          
                c.add(gates.CZ(q, q+1))
            
            # Ry rotations   `
            for q in range(self.nqubits):
                c.add(gates.RY(q, theta[index]))
                index+=1
            
            # CZ gates
            for q in range(1, self.nqubits-1, 2):         
                c.add(gates.CZ(q, q+1))
            
            c.add(gates.CZ(0, self.nqubits-1))
        
        # Final Ry rotations    
        for q in range(self.nqubits):       
            c.add(gates.RY(q, theta[index]))
            index+=1 

        return c

    def Circuit(self, theta, nlayers, RY=True):
        """    
        Args: 
            theta: list or numpy.array with the biases and the angles to be used in the circuit       
            nlayers: int number of layers of the varitional circuit ansatz         
            RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                if False, parameterized Ry gates are used in the circuit (default=False)
    
        Returns: 
            qibo.tensorflow.circuit.TensorflowCircuit with the variational circuit for angles "theta"
        """
        bias = np.array(theta[0:self.measured_qubits])
        angles = theta[self.measured_qubits:] 
        
        if RY:         
            circuit = self.ansatz_RY(angles, nlayers)        
            circuit.add(gates.M(*range(self.measured_qubits)))            
        elif not RY:           
            circuit = self.ansatz(angles, nlayers)        
            circuit.add(gates.M(*range(self.measured_qubits)))          
        else:       
            raise ValueError('RY must take a Boolean value')
    
        return circuit
    
    
    def Predictions(self, circuit, theta, init_state, nshots=10000):
        """    
        Args: 
            theta: list or numpy.array with the biases to be used in the circuit                         
            init_state: numpy.array with the quantum state to be classified
            nshots: int number of runs of the circuit during the sampling process (default=10000)
    
        Returns: 
            numpy.array() with predictions for each qubit, for the initial state
        """  
        bias = np.array(theta[0:self.measured_qubits]) 
        circuit = circuit(init_state, nshots)      
        result = circuit.frequencies(binary=False)
        prediction = np.zeros(self.measured_qubits)
    
        for qubit in range(self.measured_qubits):      
            for clase in range(self.nclasses):           
                binary = bin(clase)[2:].zfill(self.measured_qubits)
                prediction[qubit] += result[clase] * (1-2*int(binary[-qubit-1]))
    
        return prediction/nshots + bias


    def square_loss(self, labels, predictions):
        """    
        Args: 
            labels: list or numpy.array with the qubit labels of the quantum states to be classified                    
            predictions: list or numpy.array with the qubit predictions for the quantum states to be classified
    
        Returns: 
            numpy.float32 with the value of the square-loss function
        """
        loss = 0
        for l, p in zip(labels, predictions):    
            for qubit in range(self.measured_qubits):
                loss += (l[qubit] - p[qubit]) ** 2
    
        return loss / len(labels)



    def Cost_function(self, theta, nlayers, data=None, labels=None, nshots=10000, RY=True):
        """    
        Args: 
            theta: list or numpy.array with the biases and the angles to be used in the circuit  
            nlayers: int number of layers of the varitional circuit ansatz   
            data: numpy.array data[page][word]  (this is an array of kets)       
            labels: list or numpy.array with the labels of the quantum states to be classified                 
            nshots: int number of runs of the circuit during the sampling process (default=10000)
    
        Returns: 
            numpy.float32 with the value of the square-loss function
        """  
        qibo.set_backend("matmuleinsum")
        circ = self.Circuit(theta, nlayers, RY)
        circ.compile()
        
        Bias = np.array(theta[0:self.measured_qubits])
        predictions = np.zeros(shape=(len(data),self.measured_qubits))
    
        for i, text in enumerate(data):
            predictions[i] = self.Predictions(circ, Bias, text, nshots)
    
        s = self.square_loss(labels, predictions)
        print(s)
        return s
    
    
    def minimize(self, init_theta, nlayers, data=None, labels=None, nshots=10000, 
                 RY=True, method='Powell'):
        """
        Args: 
            theta: list or numpy.array with the angles to be used in the circuit        
            nlayers: int number of layers of the varitional ansatz             
            init_state: numpy.array with the quantum state to be Schmidt-decomposed    
            nshots: int number of runs of the circuit during the sampling process (default=10000)         
            RY: if True, parameterized Rx,Rz,Rx gates are used in the circuit
                if False, parameterized Ry gates are used in the circuit (default=True)               
            method: str 'classical optimizer for the minimization'. All methods from scipy.optimize.minmize are suported (default='Powell')
        
        Returns: 
            numpy.float64 with value of the minimum found, numpy.ndarray with the optimal angles        
        """        
        from scipy.optimize import minimize
        
        result = minimize(self.Cost_function, init_theta, args=(nlayers,data,labels,nshots,RY), method=method)
        loss = result.fun
        optimal_angles = result.x     
        
        return loss, optimal_angles

    
    def Accuracy(self, labels, predictions, sign=True, tolerance=1e-2):
        """
        Args: 
            labels: numpy.array with the labels of the quantum states to be classified
            predictions: numpy.array with the predictions for the quantum states classified         
            sign: if True, labels = np.sign(labels) and predictions = np.sign(predictions) (default=True)            
            tolerance: float tolerance level to consider a prediction correct (default=1e-2)
    
        Returns: 
            float with the proportion of states classified successfully
        """
        if sign == True:     
            labels = [np.sign(label) for label in labels]
            predictions = [np.sign(prediction) for prediction in predictions]
    
        accur = 0
        for l, p in zip(labels, predictions):
            if np.allclose(l, p, rtol=0., atol=tolerance):           
                accur += 1
            
        accur = accur / len(labels)
    
        return accur