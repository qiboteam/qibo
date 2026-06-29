import numpy as np
import pytest

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler.asserts import assert_decomposition
from qibo.transpiler.unroller import NativeGates, cz_dec, translate_gate

default_natives = NativeGates.Z | NativeGates.RZ | NativeGates.M | NativeGates.I


def assert_matrices_allclose(gate, natives, backend):
    target_matrix = gate.matrix(backend)
    # Remove global phase from target matrix
    normalisation = np.power(
        np.linalg.det(backend.to_numpy(target_matrix)),
        1 / float(target_matrix.shape[0]),
        dtype=complex,
    )
    target_unitary = target_matrix / normalisation

    circuit = Circuit(len(gate.qubits))
    circuit.add(translate_gate(gate, natives, backend=backend))
    native_matrix = circuit.unitary(backend)
    # Remove global phase from native matrix
    normalisation = np.power(
        np.linalg.det(backend.to_numpy(native_matrix)),
        1 / float(native_matrix.shape[0]),
        dtype=complex,
    )
    native_unitary = native_matrix / normalisation

    # There can still be phase differences of -1, -1j, 1j
    c = 0
    for phase in [1, -1, 1j, -1j]:
        if backend.allclose(
            phase * native_unitary,
            target_unitary,
            atol=1e-6,
        ):
            c = 1
    backend.assert_allclose(c, 1)
    assert_decomposition(circuit, natives)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize("gate_name", ["H", "X", "Y", "I"])
def test_pauli_to_native(backend, gate_name, natives):
    gate = getattr(gates, gate_name)(0)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize("gate_name", ["RX", "RY", "RZ"])
def test_rotations_to_native(backend, gate_name, natives):
    gate = getattr(gates, gate_name)(0, theta=0.1)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize("gate_name", ["S", "SDG", "T", "TDG", "SX"])
def test_special_single_qubit_to_native(backend, gate_name, natives):
    gate = getattr(gates, gate_name)(0)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_u1_to_native(backend, natives):
    gate = gates.U1(0, theta=0.5)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_u2_to_native(backend, natives):
    gate = gates.U2(0, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_u3_to_native(backend, natives):
    gate = gates.U3(0, theta=0.2, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_gpi2_to_native(backend, natives):
    gate = gates.GPI2(0, phi=0.123)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("gate_name", ["CNOT", "CZ", "SWAP", "iSWAP", "FSWAP"])
@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_two_qubit_to_native(backend, gate_name, natives_1q, natives_2q):
    gate = getattr(gates, gate_name)(0, 1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
@pytest.mark.parametrize("gate_name", ["CRX", "CRY", "CRZ"])
def test_controlled_rotations_to_native(backend, gate_name, natives_1q, natives_2q):
    gate = getattr(gates, gate_name)(0, 1, 0.3)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_cu1_to_native(backend, natives_1q, natives_2q):
    gate = gates.CU1(0, 1, theta=0.4)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_cu2_to_native(backend, natives_1q, natives_2q):
    gate = gates.CU2(0, 1, phi=0.2, lam=0.3)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_cu3_to_native(backend, natives_1q, natives_2q):
    gate = gates.CU3(0, 1, theta=0.2, phi=0.3, lam=0.4)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_fSim_to_native(backend, natives_1q, natives_2q):
    gate = gates.fSim(0, 1, theta=0.3, phi=0.1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize("seed", [None, 10])
@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_GeneralizedfSim_to_native(backend, natives_1q, natives_2q, seed):
    unitary = random_unitary(2, seed=seed, backend=backend)
    gate = gates.GeneralizedfSim(0, 1, unitary, phi=0.1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
@pytest.mark.parametrize("gate_name", ["RXX", "RZZ", "RYY"])
def test_rnn_to_native(backend, gate_name, natives_1q, natives_2q):
    gate = getattr(gates, gate_name)(0, 1, theta=0.1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_TOFFOLI_to_native(backend, natives_1q, natives_2q):
    gate = gates.TOFFOLI(0, 1, 2)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize("seed", [None, 10])
@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_to_native(backend, nqubits, natives_1q, natives_2q, seed):
    u = random_unitary(2**nqubits, seed=seed, backend=NumpyBackend())
    # transform to SU(2^nqubits) form
    u = u / np.sqrt(np.linalg.det(u))
    gate = gates.Unitary(u, *range(nqubits))
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


def test_count_1q(backend):
    np.testing.assert_allclose(cz_dec.count_1q(gates.CNOT(0, 1), backend), 2)
    np.testing.assert_allclose(cz_dec.count_1q(gates.CRX(0, 1, 0.1), backend), 2)


def test_count_2q(backend):
    np.testing.assert_allclose(cz_dec.count_2q(gates.CNOT(0, 1), backend), 1)
    np.testing.assert_allclose(cz_dec.count_2q(gates.CRX(0, 1, 0.1), backend), 2)


@pytest.mark.parametrize("seed", [None, 42])
def test_multi_controlled_su2_decomposition(backend, seed):
    # Test that the decomposed multi-controlled SU(2) circuit matches the original unitary.
    nqubits = 4

    # Create a random 1-qubit gate
    u_matrix = random_unitary(2, seed=seed, backend=backend)

    # Ensure it's in SU(2) by removing global phase, same as transpiler expects
    det = backend.det(u_matrix)
    u_su2 = u_matrix / backend.sqrt(det)

    # Original Circuit: 3-controlled SU(2) gate
    c_orig = Circuit(nqubits)
    c_orig.add(gates.Unitary(u_su2, 3).controlled_by(0, 1, 2))

    # Decomposed Circuit
    c_decomp = c_orig.decompose()

    # Assert unitaries match up to global phase
    matrix_orig = c_orig.unitary(backend)
    matrix_decomp = c_decomp.unitary(backend)

    # Remove global phase from both to compare them cleanly
    norm_orig = np.power(
        np.linalg.det(backend.to_numpy(matrix_orig)),
        1 / float(matrix_orig.shape[0]),
        dtype=complex,
    )
    norm_decomp = np.power(
        np.linalg.det(backend.to_numpy(matrix_decomp)),
        1 / float(matrix_decomp.shape[0]),
        dtype=complex,
    )

    u_orig_clean = matrix_orig / norm_orig
    u_decomp_clean = matrix_decomp / norm_decomp

    # Check for phase equivalencies
    match = False
    for phase in [1, -1, 1j, -1j]:
        if backend.allclose(phase * u_decomp_clean, u_orig_clean, atol=1e-6):
            match = True
            break

    assert (
        match
    ), "The decomposed multi-controlled SU(2) gate does not match the original unitary."

@pytest.mark.parametrize("seed", [None, 42])
def test_multi_controlled_su2_decomposition(backend, seed):
    #Test that the decomposed multi-controlled SU(2) circuit matches the original unitary.
    nqubits = 4
    
    # Create a random 1-qubit gate
    u_matrix = random_unitary(2, seed=seed, backend=backend)
    
    # Ensure it's in SU(2) by removing global phase
    det = backend.det(u_matrix)
    u_su2 = u_matrix / backend.sqrt(det)
    
    # Original Circuit: 3-controlled SU(2 gate
    c_orig = Circuit(nqubits)
    c_orig.add(gates.Unitary(u_su2, 3).controlled_by(0, 1, 2))
    
    # Decomposed Circuit
    c_decomp = c_orig.decompose() 
    
    # Assert unitaries match up to global phase
    matrix_orig = c_orig.unitary(backend)
    matrix_decomp = c_decomp.unitary(backend)
    
    # Remove global phase from both to compare them cleanly
    norm_orig = np.power(
        np.linalg.det(backend.to_numpy(matrix_orig)),
        1 / float(matrix_orig.shape[0]),
        dtype=complex,
    )
    norm_decomp = np.power(
        np.linalg.det(backend.to_numpy(matrix_decomp)),
        1 / float(matrix_decomp.shape[0]),
        dtype=complex,
    )
    
    u_orig_clean = matrix_orig / norm_orig
    u_decomp_clean = matrix_decomp / norm_decomp
    
    # Check for phase equivalencies
    match = False
    for phase in [1, -1, 1j, -1j]:
        if backend.allclose(phase * u_decomp_clean, u_orig_clean, atol=1e-6):
            match = True
            break
            
    assert match, "The decomposed multi-controlled SU(2) gate does not match the original unitary."