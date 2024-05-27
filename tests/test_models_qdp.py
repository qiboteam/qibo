"""Test cases for DensityMatrixExponentiation model."""
import numpy as np
import pytest
from qibo.models.qdp.qunatum_dynamic_programming import DensityMatrixExponentiation
from qibo import gates
from qibo.models.circuit import Circuit

def test_increment_instruction_qubit():
    """Test incrementing the current instruction register index."""
    protocol = DensityMatrixExponentiation(theta=np.pi, N=1, num_work_qubits=1,
                                           num_instruction_qubits=2, number_muq_per_call=1)
    protocol.increment_current_instruction_register()
    assert protocol.id_current_instruction_reg == 2

def test_return_circuit():
    """Test returning the circuit generated by DensityMatrixExponentiation."""
    protocol = DensityMatrixExponentiation(theta=np.pi, N=2, num_work_qubits=1,
                                           num_instruction_qubits=2, number_muq_per_call=1)
    protocol.memory_call_circuit(num_instruction_qubits_per_query=2)
    assert isinstance(protocol.return_circuit(), Circuit)

def compute_swapped_probability(iteration, nshots=1000):
    """Compute the probability of observing '1' after DensityMatrixExponentiation.

    Args:
        iteration (int): The number of iterations for the exponentiation.
        nshots (int, optional): The number of shots for executing the circuit. Defaults to 1000.

    Returns:
        float: The probability of observing '1' after the exponentiation.
    """
    protocol = DensityMatrixExponentiation(theta=np.pi, N=iteration, num_work_qubits=1,
                                           num_instruction_qubits=iteration, number_muq_per_call=1)
    protocol.memory_call_circuit(num_instruction_qubits_per_query=iteration)
    protocol.c.add(gates.M(0))
    freq_dict = protocol.c.execute(nshots=nshots).frequencies(registers=True)
    counter = freq_dict['register0']
    return counter['1'] / nshots

def test_density_matrix_exponentiation_N_one():
    """Test case for DensityMatrixExponentiation with N=1."""
    assert compute_swapped_probability(1) == 1

def test_density_matrix_exponentiation_N_20():
    """Test case for DensityMatrixExponentiation with N=20."""
    probability = compute_swapped_probability(20)
    assert probability >= 0.9