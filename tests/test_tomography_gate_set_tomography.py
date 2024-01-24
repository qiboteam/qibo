import pytest

import qibo
from qibo import gates
from qibo.tomography.gate_set_tomography import (
    GST,
    GST_execute_circuit,
    measurement_basis,
    prepare_states,
    reset_register,
)


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


def test_reset_register_valid_string_1qb():
    # Test for valid string
    nqubits = 1
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))

    invert_register = "sp_0"
    inverted_circuit = reset_register(test_circuit, invert_register)

    assert isinstance(inverted_circuit, qibo.models.circuit.Circuit)


def test_reset_register_sp_0():
    # Test resetting qubit 0

    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.S(1))

    inverse_circuit = reset_register(test_circuit, "sp_0")

    assert isinstance(inverse_circuit, qibo.models.circuit.Circuit)


def test_reset_register_sp_1():
    # Test resetting qubit 1

    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.S(1))

    inverse_circuit = reset_register(test_circuit, "sp_1")

    assert isinstance(inverse_circuit, qibo.models.circuit.Circuit)


def test_reset_register_sp_t():
    # Test resetting both qubits

    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.CNOT(0, 1))

    inverse_circuit = reset_register(test_circuit, "sp_t")

    assert isinstance(inverse_circuit, qibo.models.circuit.Circuit)


def test_reset_register_invalid_string():
    # Test resetting both qubits

    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.CNOT(0, 1))

    # Check if NameError is raised
    with pytest.raises(NameError):
        inverse_circuit = reset_register(test_circuit, "sp_2")


#################################################################################


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


#################################################################################


def test_GST_one_qubit_empty_circuit():
    nqubits = 1
    result = GST(
        nqubits, gate=None, invert_register=None, noise_model=None, backend=None
    )
    assert np.shape(result) == (4, 4)


# def test_GST_two_qubit_empty_circuit():
#     nqubits = 2
#     result = GST(nqubits)
#     assert np.shape(result) == (16, 16)


# def test_GST_one_qubit_with_gate():
#     nqubits = 1
#     test_gate = gates.H(0)
#     result = GST(nqubits, gate=test_gate)
#     assert np.shape(result) == (4, 4)


# def test_GST_two_qubit_with_gate():
#     nqubits = 2
#     test_gate = gates.CNOT(0, 1)
#     result = GST(nqubits, gate=test_gate)
#     assert np.shape(result) == (16, 16)


# def test_GST_one_qubit_with_gate_with_valid_reset_register_string():
#     nqubits = 1
#     test_gate = gates.H(0)
#     invert_register = "sp_0"
#     result = GST(nqubits=nqubits, gate=test_gate, invert_register=invert_register)
#     assert np.shape(result) == (4, 4)


# def test_GST_two_qubit_with_gate_with_valid_reset_register_string():
#     nqubits = 2
#     test_gate = gates.CZ(0, 1)
#     invert_register = "sp_t"
#     result = GST(nqubits=nqubits, gate=test_gate, invert_register=invert_register)
#     assert np.shape(result) == (16, 16)


##################################################################################
