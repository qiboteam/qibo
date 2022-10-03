# -*- coding: utf-8 -*-
import numpy as np
import math
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.models.qcnn import QuantumCNN
from qibo import matrices


def test_classifier_circuit2():
  """
  """
  nqubits = 2
  nlayers = int(nqubits / 2)
  init_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)  #   
  
  qcnn = QuantumCNN(nqubits, nlayers, nclasses=2, RY=True)
  num_angles =  21 #qcnn.nparams_layer
  angles = [i * math.pi / num_angles for i in range(num_angles)]
  
  circuit = qcnn.Classifier_circuit(angles)
  #circuit.set_circuit_params(angles) #this line is included in Classififer_circuit()  
  statevector = circuit(init_state).state()  
  real_vector = get_real_vector2()
  
  #to compare statevector and real_vector
  np.testing.assert_allclose(statevector.real, real_vector.real, atol=1e-5)
  np.testing.assert_allclose(statevector.imag, real_vector.imag, atol=1e-5) 
  
 
def get_real_vector2():
  nqubits = 2
  init_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)  #
  num_angles = 21
  angles = [i * math.pi / num_angles for i in range(num_angles)]   
  
  # convolution
  k=0
  a = one_qubit_unitary(nqubits, bits[0], angles[k:k+3]).unitary()*init_state
  k+=3
  a = one_qubit_unitary(nqubits, bits[1], angles[k:k+3]).unitary()*a
  k+=3
  a = matrices.RZZ(bits[0], bits[1], angles[k])*a
  k+=1
  a = matrices.RYY(bits[0], bits[1], angles[k])*a
  k+=1
  a = matrices.RXX(bits[0], bits[1], angles[k])*a
  k+=1
  a = one_qubit_unitary(nqubits, bits[0], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[1], angles[k:k+3]).unitary()*a
  k+=3
  
  # pooling  
  ksink = k
  a = one_qubit_unitary(nqubits, bits[1], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[0], angles[k:k+3]).unitary()*a  
  a = matrices.CNOT(bits[0], bits[1])*a
  a = one_qubit_unitary(nqubits, bits[1], angles[ksink:ksink+3]).invert().unitary()*a
  
  return a

def test_classifier_circuit4():
  """
  """
  nqubits = 4
  nlayers = int(nqubits / 2)
  init_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)  #  
  
  qcnn = QuantumCNN(nqubits, nlayers, nclasses=2, RY=True)
  num_angles =  21 #qcnn.nparams_layer
  angles = [i * math.pi / num_angles for i in range(num_angles)] 
  
  circuit = qcnn.Classifier_circuit(angles)
  statevector = circuit(init_state).state()  
  real_vector = get_real_vector4()
  
  #to compare statevector and real_vector
  np.testing.assert_allclose(statevector.real, real_vector.real, atol=1e-5)
  np.testing.assert_allclose(statevector.imag, real_vector.imag, atol=1e-5)  
  
  
def get_real_vector4():
  nqubits = 4
  init_state = np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)  #
  num_angles = 21
  angles = [i * math.pi / num_angles for i in range(num_angles)] 
  
  
  # convolution - layer 1
  #to declare matrix array a 
  k=0
  a = one_qubit_unitary(nqubits, bits[0], angles[k:k+3]).unitary()*init_state
  k+=3
  a = one_qubit_unitary(nqubits, bits[1], angles[k:k+3]).unitary()*a
  k+=3
  a = matrices.RZZ(bits[0], bits[1], angles[k])*a
  k+=1
  a = matrices.RYY(bits[0], bits[1], angles[k])*a
  k+=1
  a = matrices.RXX(bits[0], bits[1], angles[k])*a
  k+=1
  a = one_qubit_unitary(nqubits, bits[0], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[1], angles[k:k+3]).unitary()*a
  
  k=0 #k+=3
  a = one_qubit_unitary(nqubits, bits[2], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[3], angles[k:k+3]).unitary()*a
  k+=3
  a = matrices.RZZ(bits[2], bits[3], angles[k])*a
  k+=1
  a = matrices.RYY(bits[2], bits[3], angles[k])*a
  k+=1
  a = matrices.RXX(bits[2], bits[3], angles[k])*a
  k+=1
  a = one_qubit_unitary(nqubits, bits[2], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[3], angles[k:k+3]).unitary()*a
  
  # pooling - layer 1
  k=15 #k+=3
  ksink = k
  a = one_qubit_unitary(nqubits, bits[2], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[0], angles[k:k+3]).unitary()*a  
  a = matrices.CNOT(bits[0], bits[2])*a
  a = one_qubit_unitary(nqubits, bits[2], angles[ksink:ksink+3]).invert().unitary()*a  
  
  k=15 #k+=3
  ksink = k
  a = one_qubit_unitary(nqubits, bits[3], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[1], angles[k:k+3]).unitary()*a
  a = matrices.CNOT(bits[1], bits[3])*a
  a = one_qubit_unitary(nqubits, bits[3], angles[ksink:ksink+3]).invert().unitary()*a 
 

  # convolution - layer 2
  k=0
  a = one_qubit_unitary(nqubits, bits[2], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[3], angles[k:k+3]).unitary()*a
  k+=3
  a = matrices.RZZ(bits[2], bits[3], angles[k])*a
  k+=1
  a = matrices.RYY(bits[2], bits[3], angles[k])*a
  k+=1
  a = matrices.RXX(bits[2], bits[3], angles[k])*a
  k+=1
  a = one_qubit_unitary(nqubits, bits[2], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[3], angles[k:k+3]).unitary()*a
  k+=3
  
  # pooling - layer 2  
  ksink = k
  a = one_qubit_unitary(nqubits, bits[3], angles[k:k+3]).unitary()*a
  k+=3
  a = one_qubit_unitary(nqubits, bits[2], angles[k:k+3]).unitary()*a
  a = matrices.CNOT(bits[2], bits[3])*a
  a = one_qubit_unitary(nqubits, bits[3], angles[ksink:ksink+3]).invert().unitary()*a  
  
  return a
  
    
def one_qubit_unitary(nqubits, bit, symbols):
  c = Circuit(nqubits)
  c.add(gates.RX(bit,symbols[0]))
  c.add(gates.RY(bit,symbols[1]))
  c.add(gates.RZ(bit,symbols[2]))

  return c    
    
 
  
  
