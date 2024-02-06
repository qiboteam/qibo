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


@pytest.mark.parametrize("k", np.arange(0, 4, 1))
def test_prepare_states_valid_k_single_qubit(k):
    correct_gates = [[gates.I(0)], [gates.X(0)], [gates.H(0)], [gates.H(0), gates.S(0)]]
    nqubits = 1
    circuit = prepare_states(k, nqubits)

    for groundtruth, gate in zip(correct_gates[k], circuit.queue):
        assert isinstance(gate, type(groundtruth))


@pytest.mark.parametrize("k", np.arange(0, 16, 1))
def test_prepare_states_valid_k_two_qubits(k):
    correct_gates = [
        [gates.I(0), gates.I(1)],
        [gates.I(0), gates.X(1)],
        [gates.I(0), gates.H(1)],
        [gates.I(0), gates.H(1), gates.S(1)],
        [gates.X(0), gates.I(1)],
        [gates.X(0), gates.X(1)],
        [gates.X(0), gates.H(1)],
        [gates.X(0), gates.H(1), gates.S(1)],
        [gates.H(0), gates.I(1)],
        [gates.H(0), gates.X(1)],
        [gates.H(0), gates.H(1)],
        [gates.H(0), gates.H(1), gates.S(1)],
        [gates.H(0), gates.S(0), gates.I(1)],
        [gates.H(0), gates.S(0), gates.X(1)],
        [gates.H(0), gates.S(0), gates.H(1)],
        [gates.H(0), gates.S(0), gates.H(1), gates.S(1)],
    ]
    nqubits = 2
    circuit = prepare_states(k, nqubits)

    for groundtruth, gate in zip(correct_gates[k], circuit.queue):
        assert isinstance(gate, type(groundtruth))


@pytest.mark.parametrize("k, nqubits", [(0, 3), (1, 4), (2, 5), (3, 6)])
def test_prepare_states_valid_k_invalid_nqubits(k, nqubits):
    # Test for value input with invalid nqubits

    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        circuit = prepare_states(k, nqubits)


@pytest.mark.parametrize("k, nqubits", [(17, 1), (18, 1), (21, 2), (56, 2)])
def test_prepare_states_invalid_k_valid_nqubits(k, nqubits):
    # Check if IndexError is raised
    with pytest.raises(IndexError):
        circuit = prepare_states(k, nqubits)


# ##################################################################################


@pytest.mark.parametrize("j", np.arange(0, 4, 1))
def test_measurement_basis_value_j_single_qubit(j):
    # Test for valid input with a single qubit
    nqubits = 1

    test_circuit = qibo.models.Circuit(nqubits)
    new_circuit = measurement_basis(j, test_circuit)

    assert isinstance(new_circuit, qibo.models.circuit.Circuit)
    assert new_circuit.nqubits == nqubits


@pytest.mark.parametrize("j", np.arange(0, 16, 1))
def test_measurement_basis_value_j_two_qubits(j):
    # Test for valid input with two qubits
    nqubits = 2

    test_circuit = qibo.models.Circuit(nqubits)
    new_circuit = measurement_basis(j, test_circuit)

    assert isinstance(new_circuit, qibo.models.circuit.Circuit)
    assert new_circuit.nqubits == nqubits


@pytest.mark.parametrize(
    "j, nqubits",
    [
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 3),
        (4, 6),
        (5, 8),
        (6, 7),
        (7, 9),
        (8, 5),
        (9, 6),
        (10, 4),
        (11, 3),
        (12, 17),
        (13, 12),
        (14, 4),
        (15, 5),
    ],
)
def test_measurement_basis_valid_j_invalid_nqubits(j, nqubits):
    # Test for valid input with invalid qubits
    test_circuit = qibo.models.Circuit(nqubits)

    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        new_circuit = measurement_basis(j, test_circuit)


@pytest.mark.parametrize("j, nqubits", [(4, 1), (8, 1), (17, 2), (21, 2)])
def test_measurement_basis_invalid_j_valid_nqubits(j, nqubits):
    # Test for invalid input with valid nqubits
    test_circuit = qibo.models.Circuit(nqubits)

    # Check if IndexError is raised
    with pytest.raises(IndexError):
        new_circuit = measurement_basis(j, test_circuit)


# ##################################################################################


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


# #################################################################################


def test_GST_execute_circuit_1qb_j0():
    np.random.seed(42)
    nqubits = 1
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.M(0))
    k = 0
    j = 0
    result = GST_execute_circuit(circuit, k, j)
    assert result == 1.0


@pytest.mark.parametrize("k, j", [(0, 1), (0, 2), (1, 2), (2, 3)])
def test_GST_execute_circuit_1qb_jnonzero(backend, k, j):
    nqubits = 1
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    np.random.seed(42)
    control_result = GST_execute_circuit(circuit, k, j, backend=backend)
    np.random.seed(42)
    test_result = GST_execute_circuit(circuit, k, j, backend=backend)

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_execute_circuit_2qb_j0():
    np.random.seed(42)
    nqubits = 2
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))
    k = 0
    j = 0
    result = GST_execute_circuit(circuit, k, j)
    assert result == 1.0


@pytest.mark.parametrize("k, j", [(0, 1)])
def test_GST_execute_circuit_2qb_jnonzero(backend, k, j):
    nqubits = 2
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.M(0))
    circuit.add(gates.M(1))

    np.random.seed(42)
    control_result = GST_execute_circuit(circuit, k, j, backend=backend)
    np.random.seed(42)
    test_result = GST_execute_circuit(circuit, k, j, backend=backend)

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("nqubits", [3, 4, 5, 6, 7, 8])
def test_GST_execute_circuit_wrong_qb(nqubits):
    circuit = qibo.models.Circuit(nqubits)
    circuit.add(gates.M(0))
    k = 0
    j = 0

    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        result = GST_execute_circuit(circuit, k, j)


# #################################################################################


def test_GST_one_qubit_empty_circuit(backend):
    nqubits = 1
    control_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=None, backend=backend
    )
    test_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=None, backend=backend
    )
    backend.assert_allclose(control_result, test_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_empty_circuit(backend):
    nqubits = 2
    np.random.seed(42)
    control_result = execute_GST(nqubits)
    np.random.seed(42)
    test_result = execute_GST(nqubits)

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_Hgate(backend):
    nqubits = 1
    test_gate = gates.H(0)
    control_result = execute_GST(nqubits, gate=test_gate)
    test_result = execute_GST(nqubits, gate=test_gate)
    backend.assert_allclose(control_result, test_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_RXgate(backend):
    nqubits = 1
    test_gate = gates.RX(0, np.pi / 7)
    control_result = execute_GST(nqubits, gate=test_gate)
    test_result = execute_GST(nqubits, gate=test_gate)
    backend.assert_allclose(control_result, test_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_CNOTgate(backend):
    nqubits = 2
    test_gate = gates.CNOT(0, 1)
    np.random.seed(42)
    control_result = execute_GST(nqubits, gate=test_gate, backend=backend)
    np.random.seed(42)
    test_result = execute_GST(nqubits, gate=test_gate, backend=backend)

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_CRXgate(backend):
    nqubits = 2
    test_gate = gates.CRX(0, 1, np.pi / 7)
    np.random.seed(42)
    control_result = execute_GST(nqubits, gate=test_gate)
    np.random.seed(42)
    test_result = execute_GST(nqubits, gate=test_gate)
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_gate_with_valid_reset_register_string(backend):
    nqubits = 1
    invert_register = "sp_0"
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_gate_with_valid_reset_register_string(backend):
    nqubits = 2
    invert_register = "sp_1"
    np.random.seed(42)
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_param_gate_with_valid_reset_register_string(backend):
    nqubits = 1
    test_gate = gates.RX(0, np.pi / 7)
    invert_register = "sp_0"
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_param_gate_with_valid_reset_register_string(backend):
    nqubits = 2
    test_gate = gates.CNOT(0, 1)
    invert_register = "sp_1"
    np.random.seed(42)
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_gate_with_valid_reset_register_string(backend):
    nqubits = 2
    test_gate = gates.CZ(0, 1)
    invert_register = "sp_t"
    np.random.seed(42)
    control_result = execute_GST(
        nqubits=nqubits,
        gate=test_gate,
        invert_register=invert_register,
        backend=backend,
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits=nqubits,
        gate=test_gate,
        invert_register=invert_register,
        backend=backend,
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_gate_with_invalid_reset_register_string():
    nqubits = 2
    test_gate = gates.CZ(0, 1)
    invert_register = "sp_3"
    with pytest.raises(NameError):
        result = execute_GST(
            nqubits=nqubits, gate=test_gate, invert_register=invert_register
        )


def test_GST_empty_circuit_with_invalid_qb(backend):
    nqubits = 3
    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        result = execute_GST(
            nqubits, gate=None, invert_register=None, noise_model=None, backend=backend
        )


def test_GST_with_gate_with_invalid_qb(backend):
    nqubits = 3
    test_gate = gates.CNOT(0, 1)

    # Check if ValueError is raised
    with pytest.raises(ValueError):
        result = execute_GST(
            nqubits,
            gate=test_gate,
            invert_register=None,
            noise_model=None,
            backend=backend,
        )


def test_GST_with_gate_with_invalid_qb(backend):
    nqubits = 2
    test_gate = gates.H(0)

    # Check if ValueError is raised
    with pytest.raises(ValueError):
        result = execute_GST(
            nqubits,
            gate=test_gate,
            invert_register=None,
            noise_model=None,
            backend=backend,
        )


def test_GST_one_qubit_empty_circuit_with_noise(backend):
    lam = 0.5
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    noise_model = depol
    nqubits = 1
    control_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )
    test_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_empty_circuit_with_noise(backend):
    nshots = int(1e4)
    lam = 0.5
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    noise_model = depol
    nqubits = 2
    np.random.seed(42)
    control_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)
