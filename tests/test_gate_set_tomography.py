# test_gate_set_tomography.py

from itertools import product

import numpy as np
import pytest
from gate_set_tomography import (
    GST,
    GST_basis_operations,
    GST_execute_circuit,
    measurement_basis,
    prepare_states,
)

import qibo
from qibo import gates
from qibo.noise import AmplitudeDampingError, DepolarizingError, NoiseModel, PauliError

# import gate_set_tomography
# from gate_set_tomography import GST_1qb, GST_2qb, GST_1qb_basis_operations, GST_2qb_basis_operations


def test_prepare_states_valid_k_single_qubit():
    # Test for valid input with a single qubit
    k = 2
    nqubits = 1

    circuit = prepare_states(k, nqubits)

    assert isinstance(circuit, qibo.models.circuit.Circuit)
    assert circuit.nqubits == nqubits


def test_prepare_states_valid_k_two_qubits():
    # Test for valid input with two qubits
    k = 1
    nqubits = 2

    circuit = prepare_states(k, nqubits)

    assert isinstance(circuit, qibo.models.circuit.Circuit)
    assert circuit.nqubits == nqubits


def test_prepare_states_valid_k_invalid_nqubits():
    # Test for value input with invalid nqubits
    k = 0
    nqubits = 3

    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        circuit = prepare_states(k, nqubits)


def test_prepare_states_invalid_k_valid_nqubits():
    # Test for invalid input with valid nqubits
    k = 17
    nqubits = 2

    # Check if IndexError is raised
    with pytest.raises(IndexError):
        circuit = prepare_states(k, nqubits)


##################################################################################


def test_measurement_basis_value_j_single_qubit():
    # Test for valid input with a single qubit
    j = 2
    nqubits = 1

    test_circuit = qibo.models.Circuit(nqubits)
    new_circuit = measurement_basis(j, test_circuit)

    assert isinstance(new_circuit, qibo.models.circuit.Circuit)
    assert new_circuit.nqubits == nqubits


def test_measurement_basis_value_j_two_qubits():
    # Test for valid input with two qubits
    j = 1
    nqubits = 2

    test_circuit = qibo.models.Circuit(nqubits)
    new_circuit = measurement_basis(j, test_circuit)

    assert isinstance(new_circuit, qibo.models.circuit.Circuit)
    assert new_circuit.nqubits == nqubits


def test_measurement_basis_valid_j_invalid_nqubits():
    # Test for valid input with invalid qubits
    j = 0
    nqubits = 3

    test_circuit = qibo.models.Circuit(nqubits)

    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        new_circuit = measurement_basis(j, test_circuit)


def test_measurement_basis_invalid_j_valid_nqubits():
    # Test for invalid input with valid nqubits
    j = 17
    nqubits = 2

    test_circuit = qibo.models.Circuit(nqubits)

    # Check if IndexError is raised
    with pytest.raises(IndexError):
        new_circuit = measurement_basis(j, test_circuit)


##################################################################################


def test_GST_execute_circuit():
    nqubits = 1
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.M(0))
    k = 0
    j = 0
    result = GST_execute_circuit(circuit, k, j)
    if j == 0:
        # For j = 0, expect an exact match
        assert result == 1.0
    else:
        # For other values of j, use pytest.approx
        assert isinstance(result, float)


##################################################################################


def test_GST_one_qubit_empty_circuit():
    nqubits = 1
    result = GST(nqubits)
    assert np.shape(result) == (4, 4)  # and np.shape(result)[1] == 4


def test_GST_two_qubit_empty_circuit():
    nqubits = 2
    result = GST(nqubits)
    assert np.shape(result) == (16, 16)  # and np.shape(result)[1] == 4


def test_GST_one_qubit_with_gate():
    nqubits = 1
    test_gate = gates.H(0)
    result = GST(nqubits, gate=test_gate)
    assert np.shape(result) == (4, 4)  # and np.shape(result)[1] == 4


def test_GST_two_qubit_with_gate():
    nqubits = 2
    test_gate = gates.CNOT(0, 1)
    result = GST(nqubits, gate=test_gate)
    assert np.shape(result) == (16, 16)  # and np.shape(result)[1] == 4


##################################################################################


def test_GST_basis_operations_no_inputs():
    # Example 1: Test with default parameters
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        output1, output2, output3 = GST_basis_operations()


def test_GST_basis_operations_with_inputs_one_qubit():
    # Example 2: Test with specific parameters
    nqubits = 1
    gjk1 = GST(nqubits)

    lam = 0.2
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))

    output1, output2, output3 = GST_basis_operations(
        nqubits=1, gjk=gjk1, nshots=1000, noise_model=depol
    )

    assert (
        np.shape(output1) == (13, 4, 4)
        and np.shape(output2) == (16, 13)
        and np.shape(output3) == (13, 2, 2)
    )
    # assert np.shape(output1) == (241, 16, 16) and np.shape(output2) == (256, 241) and np.shape(output3) == (241, 4, 4)


def test_GST_basis_operations_with_inputs_two_qubit():
    # Example 2: Test with specific parameters
    nqubits = 2
    gjk2 = GST(nqubits)

    lam = 0.4
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))

    output1, output2, output3 = GST_basis_operations(
        nqubits=2, gjk=gjk2, nshots=1000, noise_model=depol
    )

    # assert np.shape(output1) == (13, 4, 4) and np.shape(output2) == (16, 13) and np.shape(output3) == (13, 2, 2)
    assert (
        np.shape(output1) == (241, 16, 16)
        and np.shape(output2) == (256, 241)
        and np.shape(output3) == (241, 4, 4)
    )
