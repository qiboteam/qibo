# -*- coding: utf-8 -*-
import numpy as np
import math
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.models.qcnn import QuantumCNN


def test_classifier_circuit2():
  """
  """
  nqubits = 2
  nlayers = int(nqubits / 2)
  init_state = np.ones(nqubits) / 2.0 + i  #
  angles = [i * math.pi / num_angles for i in range(num_angles)] 
  
  qcnn = QuantumCNN(nqubits, nlayers, nclasses=2, RY=True)
  num_angles =  qcnn.nparams_layer
  
  circuit = qcnn.Classififer_circuit(angles)
  #circuit.set_circuit_params(angles) #this line is included in Classififer_circuit()  
  statevector = circuit(init_state).state()  
  real_vector = get_real_vector2()
  
  #to compare statevector and real_vector
  np.testing.assert_allclose(statevector.real, real_vector.real, atol=1e-5)
  np.testing.assert_allclose(statevector.imag, real_vector.imag, atol=1e-5) 
  
 
def get_real_vector2():
  nqubits = 2
  init_state = np.ones(nqubits) / 2.0 + i  #
  num_angles = 21
  angles = [i * math.pi / num_angles for i in range(num_angles)] 
  
  
  # convolution
  #to declare matrix array a  
  i=0
  k=0
  a[i] = one_qubit_unitary(bits[0], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[1], angles[k:k+3])
  i+=1
  k+=3
  a[i] = gates.RZZ(bits[0], bits[1], angles[k])
  i+=1
  k+=1
  a[i] = gates.RYY(bits[0], bits[1], angles[k])
  i+=1
  k+=1
  a[i] = gates.RXX(bits[0], bits[1], angles[k])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[0], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[1], angles[k:k+3])
  
  # pooling
  i+=1
  k+=3
  ksink = k
  a[i] = one_qubit_unitary(bits[1], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[0], angles[k:k+3])
  i+=1
  a[i] = gates.CNOT(bits[0], bits[1])
  i+=1
  #a[i] = sink_basis_selector.invert() #one_qubit_unitary(bits[1], angles[ksink:ksink+3]).invert()  
  

def test_classifier_circuit4():
  """
  """
  nqubits = 4
  nlayers = int(nqubits / 2)
  init_state = np.ones(nqubits) / 2.0 + i  #
  
  
  qcnn = QuantumCNN(nqubits, nlayers, nclasses=2, RY=True)
  num_angles =  qcnn.nparams_layer
  angles = [i * math.pi / num_angles for i in range(num_angles)] 
  
  circuit = qcnn.Classififer_circuit(angles)
  statevector = circuit(init_state).state()  
  real_vector = get_real_vector4()
  
  #to compare statevector and real_vector
  np.testing.assert_allclose(statevector.real, real_vector.real, atol=1e-5)
  np.testing.assert_allclose(statevector.imag, real_vector.imag, atol=1e-5)  
  
  
def get_real_vector4():
  nqubits = 4
  init_state = np.ones(nqubits) / 2.0 + i  #
  num_angles = 21*2
  angles = [i * math.pi / num_angles for i in range(num_angles)] 
  
  
  # convolution - layer 1
  #to declare matrix array a 
  i=0
  k=0
  a[i] = one_qubit_unitary(bits[0], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[1], angles[k:k+3])
  i+=1
  k+=3
  a[i] = gates.RZZ(bits[0], bits[1], angles[k])
  i+=1
  k+=1
  a[i] = gates.RYY(bits[0], bits[1], angles[k])
  i+=1
  k+=1
  a[i] = gates.RXX(bits[0], bits[1], angles[k])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[0], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[1], angles[k:k+3])
  
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[2], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[3], angles[k:k+3])
  i+=1
  k+=3
  a[i] = gates.RZZ(bits[2], bits[3], angles[k])
  i+=1
  k+=1
  a[i] = gates.RYY(bits[2], bits[3], angles[k])
  i+=1
  k+=1
  a[i] = gates.RXX(bits[2], bits[3], angles[k])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[2], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[3], angles[k:k+3])
  
  # pooling - layer 1
  i+=1
  k+=3
  ksink = k
  a[i] = one_qubit_unitary(bits[2], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[0], angles[k:k+3])
  i+=1
  k+=3
  a[i] = gates.CNOT(bits[0], bits[2])
  i+=1
  #a[i] = sink_basis_selector.invert() #one_qubit_unitary(bits[2], angles[ksink:ksink+3]).invert()  
  
  i+=1
  k+=3
  ksink = k
  a[i] = one_qubit_unitary(bits[3], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[1], angles[k:k+3])
  i+=1
  a[i] = gates.CNOT(bits[1], bits[3])
  i+=1
  #a[i] = sink_basis_selector.invert() #one_qubit_unitary(bits[3], angles[ksink:ksink+3]).invert() 
 

  # convolution - layer 2
  i+=1
  k=0
  a[i] = one_qubit_unitary(bits[2], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[3], angles[k:k+3])
  i+=1
  k+=3
  a[i] = gates.RZZ(bits[2], bits[3], angles[k])
  i+=1
  k+=1
  a[i] = gates.RYY(bits[2], bits[3], angles[k])
  i+=1
  k+=1
  a[i] = gates.RXX(bits[2], bits[3], angles[k])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[2], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[3], angles[k:k+3])
  
  # pooling - layer 2
  i+=1
  k+=3
  ksink = k
  a[i] = one_qubit_unitary(bits[3], angles[k:k+3])
  i+=1
  k+=3
  a[i] = one_qubit_unitary(bits[2], angles[k:k+3])
  i+=1
  a[i] = gates.CNOT(bits[2], bits[3])
  i+=1
  #a[i] = sink_basis_selector.invert() #one_qubit_unitary(bits[3], angles[ksink:ksink+3]).invert()  
  
    
def one_qubit_unitary(bit, symbols):
  c = Circuit(1)
  c.add(gates.RX(bit,symbols[0]))
  c.add(gates.RY(bit,symbols[1]))
  c.add(gates.RZ(bit,symbols[2]))

  return c    
    

  
  
