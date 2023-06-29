from qibo.models import Circuit
from qibo.gates import Unitary
import numpy as np

def f(x, threshold):
    return x < threshold

def create_oracle_circ(distances, threshold, n_qubits):
    """
    Create circuit for oracle - for one solution --> I - 2*|i0><i0| - extend to multi
        Args:
            distances: numpy.ndarray of shape (num_samples,)
                - Distances to be minimized
            threshold: int
            n_qubits: int
                - number of qubits oracle circuit is applied on.
        Returns:
            qc_oracle: Circuit()
                - oracle circuit
            marked_indices: int
                - number of indices which are marked as lower than threshold
    """
    solutions = []; marked_indices=0
    for index, d in enumerate(distances):
        if f(d, threshold):
            #index_bin = np.binary_repr(index, width=n)
            ket_i0 = np.zeros((2**n_qubits, 1)); ket_i0[index] = 1
            bra_i0 = np.conj(ket_i0).T
            solutions.append(np.dot(ket_i0, bra_i0))
            marked_indices+=1
    if marked_indices==0:
        ket_i0 = np.zeros((2**n_qubits, 1)); ket_i0[np.argmin(distances)] = 1
        bra_i0 = np.conj(ket_i0).T
        solutions.append(np.dot(ket_i0, bra_i0))
        marked_indices+=1
    
    i0 = sum(solutions)
    oracle_matrix = np.identity(2**n_qubits) - 2*i0
    qc_oracle = Circuit(n_qubits)
    qc_oracle.add(Unitary(oracle_matrix, *range(n_qubits)))
    return qc_oracle, marked_indices