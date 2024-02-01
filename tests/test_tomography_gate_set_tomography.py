import numpy as np
import pytest

import qibo
from qibo import gates
from qibo.noise import DepolarizingError, NoiseModel
from qibo.tomography.gate_set_tomography import (
    GST_execute_circuit,
    execute_GST,
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


def test_reset_register_sp_0():
    # Test resetting qubit 0

    nqubits = 1
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.RX(0, np.pi / 3))

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


def test_reset_register_sp_1():
    # Test resetting qubit 1

    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.S(1))
    test_circuit.add(gates.RX(0, np.pi / 3))

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


def test_GST_execute_circuit_1qb_j0():
    np.random.seed(42)
    nqubits = 1
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.M(0))
    k = 0
    j = 0
    result = GST_execute_circuit(circuit, k, j)
    assert result == 1.0


def test_GST_execute_circuit_1qb_jnonzero():
    np.random.seed(42)
    nqubits = 1
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    k = 0
    j = 2
    result = GST_execute_circuit(circuit, k, j)
    seed42_value = 0.015200000000000047
    assert result == pytest.approx(seed42_value, abs=1e-12)


def test_GST_execute_circuit_2qb_j0():
    np.random.seed(42)
    nqubits = 2
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))
    k = 0
    j = 0
    result = GST_execute_circuit(circuit, k, j)
    assert result == 1.0


def test_GST_execute_circuit_2qb_jnonzero():
    np.random.seed(42)
    nqubits = 2
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))
    k = 0
    j = 4
    result = GST_execute_circuit(circuit, k, j)
    seed42_value = 0.015200000000000075
    assert result == pytest.approx(seed42_value, abs=1e-12)


def test_GST_execute_circuit_wrong_qb():
    nqubits = 4
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.M(0))
    k = 0
    j = 0

    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        result = GST_execute_circuit(circuit, k, j)


#################################################################################


def test_GST_one_qubit_empty_circuit():
    np.random.seed(42)
    nqubits = 1
    result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=None, backend=None
    )
    seed42_result = np.array(
        [
            [1, 1, 1, 1],
            [0.0152, 0, 1, 0.0052],
            [-0.0128, 0.0126, 0.0018, 1],
            [1, -1, -0.0114, -0.0034],
        ]
    )
    assert result == pytest.approx(seed42_result, abs=1e-12)


def test_GST_two_qubit_empty_circuit():
    np.random.seed(42)
    nqubits = 2
    result = execute_GST(nqubits)
    seed42_result = np.array(
        [
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0152,
                0.0110,
                1.0000,
                -0.0098,
                0.0158,
                0.0024,
                1.0000,
                -0.0020,
                -0.0124,
                -0.0222,
                1.0000,
                0.0064,
                -0.0046,
                0.0036,
                1.0000,
                0.0018,
            ],
            [
                -0.0128,
                -0.0062,
                0.0190,
                1.0000,
                0.0038,
                -0.0006,
                0.0034,
                1.0000,
                0.0042,
                -0.0008,
                0.0074,
                1.0000,
                -0.0054,
                -0.0078,
                0.0050,
                1.0000,
            ],
            [
                1.0000,
                -1.0000,
                0.0004,
                -0.0144,
                1.0000,
                -1.0000,
                0.0142,
                0.0082,
                1.0000,
                -1.0000,
                0.0080,
                0.0004,
                1.0000,
                -1.0000,
                0.0096,
                0.0076,
            ],
            [
                0.0000,
                -0.0150,
                0.0042,
                -0.0032,
                0.0092,
                -0.0106,
                0.0180,
                0.0080,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                0.0028,
                -0.0038,
                0.0200,
                -0.0008,
            ],
            [
                0.0114,
                0.0100,
                0.0020,
                0.0164,
                -0.0142,
                -0.0102,
                0.0096,
                -0.0154,
                -0.0050,
                0.0186,
                1.0000,
                0.0004,
                -0.0038,
                -0.0062,
                0.0080,
                0.0066,
            ],
            [
                -0.0066,
                0.0152,
                -0.0026,
                0.0112,
                -0.0080,
                0.0168,
                0.0094,
                -0.0050,
                0.0082,
                -0.0102,
                0.0062,
                1.0000,
                0.0096,
                -0.0070,
                0.0170,
                0.0050,
            ],
            [
                -0.0070,
                0.0128,
                -0.0024,
                -0.0040,
                0.0038,
                0.0014,
                0.0146,
                -0.0054,
                1.0000,
                -1.0000,
                0.0158,
                0.0036,
                -0.0020,
                -0.0042,
                0.0020,
                -0.0032,
            ],
            [
                0.0018,
                -0.0030,
                -0.0014,
                -0.0010,
                -0.0106,
                -0.0062,
                -0.0092,
                0.0030,
                -0.0060,
                0.0080,
                0.0110,
                0.0010,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0310,
                0.0036,
                0.0072,
                0.0008,
                -0.0148,
                -0.0104,
                -0.0028,
                0.0078,
                0.0018,
                0.0036,
                0.0068,
                0.0192,
                -0.0072,
                -0.0070,
                1.0000,
                -0.0142,
            ],
            [
                -0.0040,
                -0.0006,
                -0.0046,
                0.0026,
                -0.0028,
                0.0062,
                0.0118,
                0.0036,
                -0.0112,
                -0.0008,
                -0.0068,
                0.0028,
                0.0066,
                -0.0042,
                0.0150,
                1.0000,
            ],
            [
                -0.0058,
                -0.0068,
                0.0012,
                -0.0060,
                -0.0116,
                0.0164,
                0.0022,
                -0.0078,
                -0.0038,
                -0.0076,
                0.0166,
                0.0126,
                1.0000,
                -1.0000,
                0.0060,
                0.0062,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0116,
                0.0060,
                0.0028,
                -0.0108,
                0.0060,
                -0.0132,
                0.0098,
                -0.0180,
            ],
            [
                0.0182,
                -0.0204,
                1.0000,
                0.0050,
                0.0092,
                -0.0068,
                -1.0000,
                -0.0050,
                -0.0212,
                -0.0094,
                0.0056,
                -0.0042,
                0.0068,
                -0.0060,
                -0.0118,
                0.0012,
            ],
            [
                -0.0146,
                0.0026,
                0.0088,
                1.0000,
                -0.0230,
                0.0098,
                -0.0080,
                -1.0000,
                -0.0078,
                0.0014,
                -0.0130,
                0.0034,
                0.0010,
                -0.0102,
                0.0122,
                0.0052,
            ],
            [
                1.0000,
                -1.0000,
                -0.0074,
                -0.0034,
                -1.0000,
                1.0000,
                -0.0092,
                -0.0140,
                0.0044,
                0.0020,
                -0.0050,
                -0.0030,
                0.0194,
                0.0000,
                -0.0066,
                -0.0160,
            ],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_one_qubit_with_Hgate():
    np.random.seed(42)
    nqubits = 1
    test_gate = gates.H(0)
    result = execute_GST(nqubits, gate=test_gate)
    seed42_result = np.array(
        [
            [1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, -1.0000, -0.0070, 0.0052],
            [-0.0128, 0.0126, 0.0018, -1.0000],
            [-0.0044, -0.0124, 1.0000, -0.0034],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_one_qubit_with_RXgate():
    np.random.seed(42)
    nqubits = 1
    test_gate = gates.RX(0, np.pi / 7)
    result = execute_GST(nqubits, gate=test_gate)
    seed42_result = np.array(
        [
            [1.0000, 1.0000, 1.0000, 1.0000],
            [0.0152, 0.0000, 1.0000, 0.0052],
            [-0.4466, 0.4360, 0.0018, 0.9068],
            [0.9016, -0.8976, -0.0114, 0.4186],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_two_qubit_with_CNOTgate():
    np.random.seed(42)
    nqubits = 2
    test_gate = gates.CNOT(0, 1)
    result = execute_GST(nqubits, gate=test_gate)
    seed42_result = np.array(
        [
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0152,
                0.0110,
                1.0000,
                -0.0098,
                0.0158,
                0.0024,
                1.0000,
                -0.0020,
                -0.0124,
                -0.0222,
                1.0000,
                0.0064,
                -0.0046,
                0.0036,
                1.0000,
                0.0018,
            ],
            [
                -0.0128,
                -0.0062,
                0.0190,
                1.0000,
                0.0038,
                -0.0006,
                0.0034,
                -1.0000,
                0.0042,
                -0.0008,
                0.0074,
                0.0108,
                -0.0054,
                -0.0078,
                0.0050,
                -0.0026,
            ],
            [
                1.0000,
                -1.0000,
                0.0004,
                -0.0144,
                -1.0000,
                1.0000,
                0.0142,
                0.0082,
                -0.0024,
                -0.0064,
                0.0080,
                0.0004,
                0.0114,
                0.0034,
                0.0096,
                0.0076,
            ],
            [
                0.0000,
                -0.0150,
                0.0042,
                -0.0032,
                0.0092,
                -0.0106,
                0.0180,
                0.0080,
                0.0044,
                -0.0072,
                1.0000,
                -0.0074,
                0.0028,
                -0.0038,
                0.0200,
                -0.0008,
            ],
            [
                0.0114,
                0.0100,
                0.0020,
                0.0164,
                -0.0142,
                -0.0102,
                0.0096,
                -0.0154,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -0.0038,
                -0.0062,
                0.0080,
                0.0066,
            ],
            [
                -0.0066,
                0.0152,
                -0.0026,
                0.0112,
                -0.0080,
                0.0168,
                0.0094,
                0.0050,
                -0.0152,
                0.0090,
                0.0062,
                0.0100,
                1.0000,
                -1.0000,
                0.0170,
                -0.0154,
            ],
            [
                -0.0070,
                0.0128,
                -0.0024,
                -0.0040,
                -0.0038,
                -0.0014,
                0.0146,
                -0.0054,
                0.0124,
                0.0068,
                0.0158,
                0.0088,
                -0.0128,
                -0.0198,
                0.0020,
                -1.0000,
            ],
            [
                0.0018,
                -0.0030,
                -0.0014,
                -0.0010,
                -0.0106,
                -0.0062,
                -0.0092,
                0.0030,
                -0.0060,
                0.0080,
                0.0110,
                0.0010,
                0.0126,
                -0.0098,
                1.0000,
                0.0134,
            ],
            [
                0.0310,
                0.0036,
                0.0072,
                0.0008,
                -0.0148,
                -0.0104,
                -0.0028,
                0.0078,
                0.0018,
                0.0036,
                0.0068,
                0.0192,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                -0.0040,
                -0.0006,
                -0.0046,
                0.0026,
                -0.0028,
                0.0062,
                0.0118,
                -0.0036,
                -1.0000,
                1.0000,
                -0.0068,
                0.0110,
                -0.0148,
                0.0030,
                0.0150,
                0.0056,
            ],
            [
                -0.0058,
                -0.0068,
                0.0012,
                -0.0060,
                0.0116,
                -0.0164,
                0.0022,
                -0.0078,
                0.0012,
                -0.0158,
                0.0166,
                1.0000,
                -0.0080,
                -0.0012,
                0.0060,
                0.0018,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0116,
                0.0060,
                0.0028,
                -0.0108,
                0.0060,
                -0.0132,
                0.0098,
                -0.0180,
            ],
            [
                0.0182,
                -0.0204,
                1.0000,
                0.0050,
                0.0092,
                -0.0068,
                -1.0000,
                -0.0050,
                -0.0212,
                -0.0094,
                0.0056,
                -0.0042,
                0.0068,
                -0.0060,
                -0.0118,
                0.0012,
            ],
            [
                -0.0146,
                0.0026,
                0.0088,
                1.0000,
                -0.0230,
                0.0098,
                -0.0080,
                1.0000,
                -0.0078,
                0.0014,
                -0.0130,
                1.0000,
                0.0010,
                -0.0102,
                0.0122,
                1.0000,
            ],
            [
                1.0000,
                -1.0000,
                -0.0074,
                -0.0034,
                1.0000,
                -1.0000,
                -0.0092,
                -0.0140,
                1.0000,
                -1.0000,
                -0.0050,
                -0.0030,
                1.0000,
                -1.0000,
                -0.0066,
                -0.0160,
            ],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_two_qubit_with_CRXgate():
    np.random.seed(42)
    nqubits = 2
    test_gate = gates.CRX(0, 1, np.pi / 7)
    result = execute_GST(nqubits, gate=test_gate)
    seed42_result = np.array(
        [
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0152,
                0.0110,
                1.0000,
                -0.0098,
                0.0158,
                0.0024,
                1.0000,
                -0.0020,
                -0.0124,
                -0.0222,
                1.0000,
                0.0064,
                -0.0046,
                0.0036,
                1.0000,
                0.0018,
            ],
            [
                -0.0128,
                -0.0062,
                0.0190,
                1.0000,
                -0.4296,
                0.4342,
                0.0034,
                0.9042,
                -0.2128,
                0.2188,
                0.0074,
                0.9474,
                -0.2260,
                0.2188,
                0.0050,
                0.9504,
            ],
            [
                1.0000,
                -1.0000,
                0.0004,
                -0.0144,
                0.8928,
                -0.9052,
                0.0142,
                0.4342,
                0.9468,
                -0.9516,
                0.0080,
                0.2228,
                0.9502,
                -0.9490,
                0.0096,
                0.2142,
            ],
            [
                0.0000,
                -0.0150,
                0.0042,
                -0.0032,
                0.0092,
                -0.0106,
                0.0180,
                0.0080,
                0.9768,
                0.9760,
                0.9726,
                0.9706,
                0.0028,
                -0.0038,
                0.2424,
                -0.0008,
            ],
            [
                0.0114,
                0.0100,
                0.0020,
                0.0164,
                -0.0142,
                -0.0102,
                0.0096,
                -0.0154,
                -0.0018,
                0.0148,
                0.9774,
                0.0026,
                0.2090,
                0.2088,
                0.2268,
                0.2314,
            ],
            [
                -0.0066,
                0.0152,
                -0.0026,
                0.0112,
                -0.0002,
                -0.0108,
                0.0094,
                -0.0100,
                -0.2294,
                0.2218,
                0.0066,
                0.9742,
                0.0054,
                -0.0042,
                0.0098,
                0.0038,
            ],
            [
                -0.0070,
                0.0128,
                -0.0024,
                -0.0040,
                0.0018,
                0.0078,
                0.0146,
                -0.0084,
                0.9736,
                -0.9714,
                0.0162,
                0.2204,
                -0.0054,
                0.0024,
                0.0014,
                -0.0022,
            ],
            [
                0.0018,
                -0.0030,
                -0.0014,
                -0.0010,
                -0.0106,
                -0.0062,
                -0.0092,
                0.0030,
                -0.0060,
                0.0080,
                -0.2278,
                0.0010,
                0.9786,
                0.9726,
                0.9780,
                0.9760,
            ],
            [
                0.0310,
                0.0036,
                0.0072,
                0.0008,
                -0.0148,
                -0.0104,
                -0.0028,
                0.0078,
                -0.2252,
                -0.2200,
                -0.2116,
                -0.2044,
                -0.0058,
                -0.0038,
                0.9746,
                -0.0122,
            ],
            [
                -0.0040,
                -0.0006,
                -0.0046,
                0.0026,
                0.0034,
                0.0060,
                0.0118,
                0.0050,
                -0.0162,
                -0.0028,
                -0.0086,
                0.0046,
                -0.2156,
                0.2282,
                0.0132,
                0.9746,
            ],
            [
                -0.0058,
                -0.0068,
                0.0012,
                -0.0060,
                -0.0124,
                0.0158,
                0.0022,
                -0.0076,
                0.0008,
                -0.0048,
                0.0054,
                0.0104,
                0.9746,
                -0.9742,
                0.0060,
                0.2290,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0116,
                0.0060,
                0.0028,
                -0.0108,
                0.0060,
                -0.0132,
                0.0098,
                -0.0180,
            ],
            [
                0.0182,
                -0.0204,
                1.0000,
                0.0050,
                0.0092,
                -0.0068,
                -1.0000,
                -0.0050,
                -0.0212,
                -0.0094,
                0.0056,
                -0.0042,
                0.0068,
                -0.0060,
                -0.0118,
                0.0012,
            ],
            [
                -0.0146,
                0.0026,
                0.0088,
                1.0000,
                0.4144,
                -0.4252,
                -0.0080,
                -0.9050,
                0.2108,
                -0.2168,
                -0.0130,
                0.0592,
                0.2366,
                -0.2236,
                0.0122,
                0.0554,
            ],
            [
                1.0000,
                -1.0000,
                -0.0074,
                -0.0034,
                -0.9016,
                0.9040,
                -0.0092,
                -0.4478,
                0.0528,
                -0.0532,
                -0.0050,
                -0.2128,
                0.0704,
                -0.0548,
                -0.0066,
                -0.2200,
            ],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_one_qubit_with_gate_with_valid_reset_register_string():
    np.random.seed(42)
    nqubits = 1
    invert_register = "sp_0"
    result = execute_GST(nqubits=nqubits, gate=None, invert_register=invert_register)
    seed42_result = np.array(
        [
            [1.0000, 1.0000, 1.0000, 1.0000],
            [0.0152, 0.0000, -0.0070, 0.0052],
            [-0.0128, 0.0126, 0.0018, -0.0058],
            [1.0000, 1.0000, 1.0000, 1.0000],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_two_qubit_with_gate_with_valid_reset_register_string():
    np.random.seed(42)
    nqubits = 2
    invert_register = "sp_1"
    result = execute_GST(nqubits=nqubits, gate=None, invert_register=invert_register)
    seed42_result = np.array(
        [
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0152,
                0.0110,
                0.0070,
                -0.0098,
                0.0158,
                0.0024,
                -0.0102,
                -0.0020,
                -0.0124,
                -0.0222,
                -0.0260,
                0.0064,
                -0.0046,
                0.0036,
                0.0038,
                0.0018,
            ],
            [
                -0.0128,
                -0.0062,
                0.0190,
                -0.0174,
                0.0038,
                -0.0006,
                0.0034,
                -0.0036,
                0.0042,
                -0.0008,
                0.0074,
                -0.0072,
                -0.0054,
                -0.0078,
                0.0050,
                -0.0084,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0000,
                -0.0150,
                0.0042,
                -0.0032,
                0.0092,
                -0.0106,
                0.0180,
                0.0080,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                0.0028,
                -0.0038,
                0.0200,
                -0.0008,
            ],
            [
                0.0114,
                0.0100,
                -0.0102,
                0.0164,
                -0.0142,
                -0.0102,
                -0.0010,
                -0.0154,
                -0.0050,
                0.0186,
                -0.0032,
                0.0004,
                -0.0038,
                -0.0062,
                -0.0068,
                0.0066,
            ],
            [
                -0.0066,
                0.0152,
                -0.0026,
                0.0040,
                -0.0080,
                0.0168,
                0.0094,
                -0.0184,
                0.0082,
                -0.0102,
                0.0062,
                0.0096,
                0.0096,
                -0.0070,
                0.0170,
                -0.0154,
            ],
            [
                -0.0070,
                -0.0128,
                0.0022,
                0.0138,
                0.0038,
                -0.0014,
                -0.0008,
                -0.0054,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -0.0020,
                0.0042,
                -0.0040,
                0.0074,
            ],
            [
                0.0018,
                -0.0030,
                -0.0014,
                -0.0010,
                -0.0106,
                -0.0062,
                -0.0092,
                0.0030,
                -0.0060,
                0.0080,
                0.0110,
                0.0010,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0310,
                0.0036,
                0.0068,
                0.0008,
                -0.0148,
                -0.0104,
                -0.0064,
                0.0078,
                0.0018,
                0.0036,
                -0.0046,
                0.0192,
                -0.0072,
                -0.0070,
                0.0032,
                -0.0142,
            ],
            [
                -0.0040,
                -0.0006,
                -0.0046,
                -0.0070,
                -0.0028,
                0.0062,
                0.0118,
                -0.0028,
                -0.0112,
                -0.0008,
                -0.0068,
                0.0110,
                0.0066,
                -0.0042,
                0.0150,
                -0.0114,
            ],
            [
                -0.0058,
                0.0068,
                -0.0042,
                0.0128,
                -0.0116,
                -0.0164,
                -0.0052,
                -0.0002,
                -0.0038,
                0.0076,
                -0.0002,
                -0.0150,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0116,
                0.0060,
                0.0028,
                -0.0108,
                0.0060,
                -0.0132,
                0.0098,
                -0.0180,
            ],
            [
                0.0182,
                -0.0204,
                0.0016,
                0.0050,
                0.0092,
                -0.0068,
                -0.0092,
                -0.0050,
                -0.0212,
                -0.0094,
                -0.0064,
                -0.0042,
                0.0068,
                -0.0060,
                -0.0136,
                0.0012,
            ],
            [
                -0.0146,
                0.0026,
                0.0088,
                0.0082,
                -0.0230,
                0.0098,
                -0.0080,
                0.0048,
                -0.0078,
                0.0014,
                -0.0130,
                -0.0146,
                0.0010,
                -0.0102,
                0.0122,
                0.0084,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0044,
                -0.0020,
                0.0084,
                -0.0070,
                0.0194,
                0.0000,
                -0.0004,
                0.0044,
            ],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_one_qubit_with_param_gate_with_valid_reset_register_string():
    np.random.seed(42)
    nqubits = 1
    test_gate = gates.RX(0, np.pi / 7)
    invert_register = "sp_0"
    result = execute_GST(nqubits=nqubits, gate=None, invert_register=invert_register)
    seed42_result = np.array(
        [
            [1.0000, 1.0000, 1.0000, 1.0000],
            [0.0152, 0.0000, -0.0070, 0.0052],
            [-0.0128, 0.0126, 0.0018, -0.0058],
            [1.0000, 1.0000, 1.0000, 1.0000],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_two_qubit_with_param_gate_with_valid_reset_register_string():
    np.random.seed(42)
    nqubits = 2
    test_gate = gates.CNOT(0, 1)
    invert_register = "sp_1"
    result = execute_GST(nqubits=nqubits, gate=None, invert_register=invert_register)
    seed42_result = np.array(
        [
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0152,
                0.0110,
                0.0070,
                -0.0098,
                0.0158,
                0.0024,
                -0.0102,
                -0.0020,
                -0.0124,
                -0.0222,
                -0.0260,
                0.0064,
                -0.0046,
                0.0036,
                0.0038,
                0.0018,
            ],
            [
                -0.0128,
                -0.0062,
                0.0190,
                -0.0174,
                0.0038,
                -0.0006,
                0.0034,
                -0.0036,
                0.0042,
                -0.0008,
                0.0074,
                -0.0072,
                -0.0054,
                -0.0078,
                0.0050,
                -0.0084,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0000,
                -0.0150,
                0.0042,
                -0.0032,
                0.0092,
                -0.0106,
                0.0180,
                0.0080,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                0.0028,
                -0.0038,
                0.0200,
                -0.0008,
            ],
            [
                0.0114,
                0.0100,
                -0.0102,
                0.0164,
                -0.0142,
                -0.0102,
                -0.0010,
                -0.0154,
                -0.0050,
                0.0186,
                -0.0032,
                0.0004,
                -0.0038,
                -0.0062,
                -0.0068,
                0.0066,
            ],
            [
                -0.0066,
                0.0152,
                -0.0026,
                0.0040,
                -0.0080,
                0.0168,
                0.0094,
                -0.0184,
                0.0082,
                -0.0102,
                0.0062,
                0.0096,
                0.0096,
                -0.0070,
                0.0170,
                -0.0154,
            ],
            [
                -0.0070,
                -0.0128,
                0.0022,
                0.0138,
                0.0038,
                -0.0014,
                -0.0008,
                -0.0054,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -0.0020,
                0.0042,
                -0.0040,
                0.0074,
            ],
            [
                0.0018,
                -0.0030,
                -0.0014,
                -0.0010,
                -0.0106,
                -0.0062,
                -0.0092,
                0.0030,
                -0.0060,
                0.0080,
                0.0110,
                0.0010,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0310,
                0.0036,
                0.0068,
                0.0008,
                -0.0148,
                -0.0104,
                -0.0064,
                0.0078,
                0.0018,
                0.0036,
                -0.0046,
                0.0192,
                -0.0072,
                -0.0070,
                0.0032,
                -0.0142,
            ],
            [
                -0.0040,
                -0.0006,
                -0.0046,
                -0.0070,
                -0.0028,
                0.0062,
                0.0118,
                -0.0028,
                -0.0112,
                -0.0008,
                -0.0068,
                0.0110,
                0.0066,
                -0.0042,
                0.0150,
                -0.0114,
            ],
            [
                -0.0058,
                0.0068,
                -0.0042,
                0.0128,
                -0.0116,
                -0.0164,
                -0.0052,
                -0.0002,
                -0.0038,
                0.0076,
                -0.0002,
                -0.0150,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0116,
                0.0060,
                0.0028,
                -0.0108,
                0.0060,
                -0.0132,
                0.0098,
                -0.0180,
            ],
            [
                0.0182,
                -0.0204,
                0.0016,
                0.0050,
                0.0092,
                -0.0068,
                -0.0092,
                -0.0050,
                -0.0212,
                -0.0094,
                -0.0064,
                -0.0042,
                0.0068,
                -0.0060,
                -0.0136,
                0.0012,
            ],
            [
                -0.0146,
                0.0026,
                0.0088,
                0.0082,
                -0.0230,
                0.0098,
                -0.0080,
                0.0048,
                -0.0078,
                0.0014,
                -0.0130,
                -0.0146,
                0.0010,
                -0.0102,
                0.0122,
                0.0084,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0044,
                -0.0020,
                0.0084,
                -0.0070,
                0.0194,
                0.0000,
                -0.0004,
                0.0044,
            ],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_two_qubit_with_gate_with_valid_reset_register_string():
    np.random.seed(42)
    nqubits = 2
    test_gate = gates.CZ(0, 1)
    invert_register = "sp_t"
    result = execute_GST(
        nqubits=nqubits, gate=test_gate, invert_register=invert_register
    )
    seed42_result = np.array(
        [
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0152,
                0.0110,
                0.0070,
                -0.0098,
                0.0158,
                0.0024,
                -0.0102,
                -0.0020,
                0.0130,
                -0.0094,
                0.0056,
                0.0116,
                0.0024,
                0.0076,
                -0.0082,
                -0.0152,
            ],
            [
                -0.0128,
                -0.0062,
                0.0190,
                -0.0174,
                0.0038,
                -0.0006,
                0.0034,
                -0.0036,
                0.0142,
                -0.0022,
                0.0038,
                0.0108,
                -0.0152,
                -0.0226,
                -0.0088,
                -0.0026,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0000,
                -0.0150,
                0.0042,
                -0.0032,
                0.0092,
                -0.0106,
                0.0180,
                0.0080,
                0.0044,
                -0.0072,
                -0.0102,
                -0.0074,
                0.0028,
                -0.0038,
                0.0200,
                -0.0008,
            ],
            [
                0.0114,
                0.0100,
                -0.0102,
                0.0164,
                -0.0142,
                -0.0102,
                -0.0010,
                -0.0154,
                0.0112,
                -0.0196,
                -0.0112,
                -0.0110,
                -0.0038,
                -0.0062,
                -0.0068,
                0.0066,
            ],
            [
                -0.0066,
                0.0152,
                -0.0026,
                0.0040,
                -0.0080,
                0.0168,
                0.0094,
                -0.0184,
                -0.0152,
                0.0090,
                -0.0026,
                0.0100,
                0.0096,
                -0.0070,
                0.0170,
                -0.0154,
            ],
            [
                -0.0070,
                -0.0128,
                0.0022,
                0.0138,
                0.0038,
                -0.0014,
                -0.0008,
                -0.0054,
                0.0042,
                0.0004,
                0.0158,
                0.0036,
                -0.0020,
                0.0042,
                -0.0040,
                0.0074,
            ],
            [
                0.0018,
                -0.0030,
                -0.0014,
                -0.0010,
                -0.0106,
                -0.0062,
                -0.0092,
                0.0030,
                -0.0060,
                0.0080,
                0.0110,
                0.0010,
                0.0126,
                -0.0098,
                -0.0046,
                0.0134,
            ],
            [
                0.0310,
                0.0036,
                0.0068,
                0.0008,
                -0.0148,
                -0.0104,
                -0.0064,
                0.0078,
                0.0018,
                0.0036,
                -0.0046,
                0.0192,
                0.0056,
                0.0010,
                0.0072,
                -0.0102,
            ],
            [
                -0.0040,
                -0.0006,
                -0.0046,
                -0.0070,
                -0.0028,
                0.0062,
                0.0118,
                -0.0028,
                -0.0112,
                -0.0008,
                -0.0068,
                0.0110,
                -0.0148,
                0.0030,
                0.0058,
                0.0056,
            ],
            [
                -0.0058,
                0.0068,
                -0.0042,
                0.0128,
                -0.0116,
                -0.0164,
                -0.0052,
                -0.0002,
                -0.0038,
                0.0076,
                -0.0002,
                -0.0150,
                -0.0138,
                0.0176,
                0.0060,
                0.0062,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0182,
                -0.0204,
                0.0016,
                0.0050,
                -0.0092,
                0.0068,
                0.0092,
                0.0050,
                0.0068,
                -0.0066,
                0.0056,
                0.0022,
                -0.0038,
                -0.0054,
                -0.0118,
                -0.0022,
            ],
            [
                -0.0146,
                0.0026,
                0.0088,
                0.0082,
                0.0230,
                -0.0098,
                0.0080,
                -0.0048,
                -0.0020,
                0.0090,
                -0.0050,
                0.0034,
                -0.0108,
                -0.0076,
                0.0118,
                0.0052,
            ],
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_two_qubit_with_gate_with_invalid_reset_register_string():
    nqubits = 2
    test_gate = gates.CZ(0, 1)
    invert_register = "sp_3"
    with pytest.raises(NameError):
        result = execute_GST(
            nqubits=nqubits, gate=test_gate, invert_register=invert_register
        )


def test_GST_empty_circuit_with_invalid_qb():
    nqubits = 3
    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        result = execute_GST(
            nqubits, gate=None, invert_register=None, noise_model=None, backend=None
        )


def test_GST_with_gate_with_invalid_qb():
    nqubits = 3
    test_gate = gates.CNOT(0, 1)

    # Check if ValueError is raised
    with pytest.raises(ValueError):
        result = execute_GST(
            nqubits,
            gate=test_gate,
            invert_register=None,
            noise_model=None,
            backend=None,
        )


def test_GST_with_gate_with_invalid_qb():
    nqubits = 2
    test_gate = gates.H(0)

    # Check if ValueError is raised
    with pytest.raises(ValueError):
        result = execute_GST(
            nqubits,
            gate=test_gate,
            invert_register=None,
            noise_model=None,
            backend=None,
        )


def test_GST_one_qubit_empty_circuit_with_noise():
    np.random.seed(42)
    lam = 0.5
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    noise_model = depol
    nqubits = 1
    result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=None
    )
    seed42_result = np.array(
        [
            [1.0000, 1.0000, 1.0000, 1.0000],
            [0.0152, 0.0000, 0.2444, 0.0052],
            [-0.0128, 0.0126, 0.0018, 0.1272],
            [0.5106, -0.5128, -0.0114, -0.0034],
        ]
    )
    assert result == pytest.approx(seed42_result)


def test_GST_one_qubit_empty_circuit_with_noise():
    np.random.seed(42)
    nshots = int(1e4)
    lam = 0.5
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    noise_model = depol
    nqubits = 2
    result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=None
    )
    seed42_result = np.array(
        [
            [
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
            ],
            [
                0.0038,
                0.0104,
                0.2606,
                -0.0242,
                -0.0008,
                0.0038,
                0.2498,
                -0.0058,
                -0.0124,
                -0.0222,
                0.2302,
                0.0064,
                -0.0046,
                0.0036,
                0.2672,
                0.0018,
            ],
            [
                -0.0096,
                -0.0214,
                0.0134,
                0.1236,
                0.0072,
                0.0032,
                0.0100,
                0.1122,
                0.0042,
                -0.0008,
                0.0074,
                0.1170,
                -0.0054,
                -0.0078,
                0.0050,
                0.1118,
            ],
            [
                0.4854,
                -0.5056,
                0.0052,
                -0.0102,
                0.5066,
                -0.4948,
                0.0030,
                0.0056,
                0.4968,
                -0.4962,
                0.0080,
                0.0004,
                0.5128,
                -0.4946,
                0.0096,
                0.0076,
            ],
            [
                0.0000,
                -0.0150,
                0.0042,
                -0.0032,
                0.0092,
                -0.0106,
                0.0180,
                0.0080,
                0.2516,
                0.2574,
                0.2288,
                0.2446,
                0.0028,
                -0.0038,
                0.0200,
                -0.0008,
            ],
            [
                0.0114,
                0.0100,
                -0.0110,
                0.0164,
                -0.0142,
                -0.0102,
                -0.0022,
                -0.0154,
                0.0054,
                -0.0124,
                0.0566,
                -0.0066,
                -0.0038,
                -0.0062,
                -0.0058,
                0.0066,
            ],
            [
                -0.0066,
                0.0152,
                -0.0026,
                0.0028,
                -0.0080,
                0.0168,
                0.0094,
                -0.0156,
                -0.0140,
                0.0068,
                -0.0052,
                0.0384,
                0.0096,
                -0.0070,
                0.0170,
                -0.0188,
            ],
            [
                -0.0200,
                0.0042,
                -0.0024,
                -0.0040,
                0.0196,
                0.0084,
                0.0146,
                -0.0054,
                0.1376,
                -0.1034,
                0.0056,
                0.0098,
                -0.0120,
                -0.0202,
                0.0020,
                -0.0032,
            ],
            [
                0.0018,
                -0.0030,
                -0.0014,
                -0.0010,
                -0.0106,
                -0.0062,
                -0.0092,
                0.0030,
                -0.0060,
                0.0080,
                0.0110,
                0.0010,
                0.1394,
                0.1182,
                0.1246,
                0.1422,
            ],
            [
                0.0310,
                0.0036,
                0.0074,
                0.0008,
                -0.0148,
                -0.0104,
                -0.0086,
                0.0078,
                0.0018,
                0.0036,
                0.0032,
                0.0192,
                0.0054,
                -0.0028,
                0.0498,
                -0.0110,
            ],
            [
                -0.0040,
                -0.0006,
                -0.0046,
                -0.0160,
                -0.0028,
                0.0062,
                0.0118,
                -0.0064,
                -0.0112,
                -0.0008,
                -0.0068,
                0.0084,
                -0.0102,
                0.0096,
                0.0008,
                0.0252,
            ],
            [
                -0.0058,
                -0.0120,
                0.0012,
                -0.0060,
                -0.0098,
                0.0294,
                0.0022,
                -0.0078,
                0.0010,
                -0.0096,
                0.0166,
                0.0126,
                0.0466,
                -0.0632,
                0.0010,
                0.0032,
            ],
            [
                0.4912,
                0.4924,
                0.4960,
                0.5108,
                -0.5036,
                -0.4980,
                -0.5100,
                -0.4992,
                0.0116,
                0.0060,
                0.0028,
                -0.0108,
                0.0060,
                -0.0132,
                0.0098,
                -0.0180,
            ],
            [
                0.0076,
                -0.0110,
                0.1122,
                0.0104,
                -0.0036,
                -0.0086,
                -0.1172,
                -0.0030,
                -0.0212,
                -0.0094,
                0.0034,
                -0.0042,
                0.0068,
                -0.0060,
                -0.0100,
                0.0012,
            ],
            [
                -0.0038,
                0.0030,
                -0.0026,
                0.0682,
                -0.0100,
                0.0114,
                0.0116,
                -0.0586,
                -0.0078,
                0.0014,
                -0.0130,
                -0.0100,
                0.0010,
                -0.0102,
                0.0122,
                0.0078,
            ],
            [
                0.2340,
                -0.2598,
                0.0026,
                -0.0090,
                -0.2428,
                0.2532,
                -0.0006,
                -0.0170,
                -0.0004,
                -0.0094,
                -0.0050,
                -0.0030,
                0.0114,
                -0.0036,
                -0.0066,
                -0.0160,
            ],
        ]
    )

    assert result == pytest.approx(seed42_result)


##################################################################################
