from qibo.models import Circuit
from qibo import gates
import numpy as np
import math
# In Qiskit -> index of qubit corresponds to how significant it is: left most bit = q1 and it goes
# |q1q0> ==> but in Qibo is other way around |q0q1> Right Most Significant bit

def iam_operator(n):
    qc = Circuit(n)

    for qubit in range(n):
        qc.add(gates.H(qubit)) # apply H-gate
        qc.add(gates.X(qubit)) # apply X-gate
        
    imaginary_I = np.matrix(np.identity(2**1), dtype=np.cfloat)
    for ix in range(imaginary_I.shape[0]):
        imaginary_I[ix, ix] = 1j
        
    qc.add(gates.Unitary(imaginary_I, 0))
    # apply H-gate to last qubit
    qc.add(gates.H(n-1))
    # apply multi-controlled toffoli gate to last qubit controlled by the ones before it (less significant)
    qc.add(gates.X(n-1).controlled_by(*list(range(0,n-1))))
    # apply H-gate to last qubit
    qc.add(gates.H(n-1))
    qc.add(gates.Unitary(imaginary_I, 0))
    
    for qubit in range(n):
        qc.add(gates.X(qubit))
        qc.add(gates.H(qubit))
    
    return qc
    

def grover_qc(qc, n, oracle, n_indices_flip):
    
    if n_indices_flip:
        r_i = int(math.floor(np.pi/4*np.sqrt(2**n/n_indices_flip)))
    else: r_i = 0
    
    for _ in range(r_i):
        qc.add(oracle.on_qubits(*(list(range(n))))) # apply oracle
        qc_diff = iam_operator(n)
        qc.add(qc_diff.on_qubits(*(list(range(n))))) # apply inversion around mean
        
    qc.add(gates.M(*list(range(n))))
    return qc