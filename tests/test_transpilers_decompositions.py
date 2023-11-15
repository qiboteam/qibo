import numpy as np
import pytest

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler.abstract import NativeType
from qibo.transpiler.unroller import translate_gate


def assert_matrices_allclose(
    gate,
    two_qubit_natives,
    backend,
    single_qubit_natives=(gates.I, gates.Z, gates.RZ, gates.U3),
):
    target_matrix = gate.matrix(backend)
    target_matrix = backend.cast(target_matrix, dtype=target_matrix.dtype)
    # Remove global phase from target matrix
    normalisation = np.power(
        np.linalg.det(target_matrix), 1 / float(target_matrix.shape[0]), dtype=complex
    )
    normalisation = backend.cast(normalisation, dtype=normalisation.dtype)
    target_unitary = target_matrix / normalisation

    circuit = Circuit(len(gate.qubits))
    circuit.add(
        translate_gate(
            gate, two_qubit_natives, single_qubit_natives=single_qubit_natives
        )
    )
    native_matrix = circuit.unitary(backend)
    # Remove global phase from native matrix
    normalisation = np.power(
        np.linalg.det(native_matrix), 1 / float(native_matrix.shape[0]), dtype=complex
    )
    normalisation = backend.cast(normalisation, dtype=normalisation.dtype)
    native_unitary = native_matrix / normalisation

    # There can still be phase differences of -1, -1j, 1j
    c = 0
    for phase in [1, -1, 1j, -1j]:
        if np.allclose(phase * native_unitary, target_unitary, atol=1e-12):
            c = 1
    backend.assert_allclose(c, 1)


@pytest.mark.parametrize("gatename", ["H", "X", "Y", "I"])
def test_pauli_to_native(backend, gatename):
    gate = getattr(gates, gatename)(0)
    assert_matrices_allclose(gate, two_qubit_natives=NativeType.CZ, backend=backend)


@pytest.mark.parametrize("gatename", ["RX", "RY", "RZ"])
def test_rotations_to_native(backend, gatename):
    gate = getattr(gates, gatename)(0, theta=0.1)
    assert_matrices_allclose(gate, two_qubit_natives=NativeType.CZ, backend=backend)


@pytest.mark.parametrize("gatename", ["S", "SDG", "T", "TDG"])
def test_special_single_qubit_to_native(backend, gatename):
    gate = getattr(gates, gatename)(0)
    assert_matrices_allclose(gate, two_qubit_natives=NativeType.CZ, backend=backend)


def test_u1_to_native(backend):
    gate = gates.U1(0, theta=0.5)
    assert_matrices_allclose(gate, two_qubit_natives=NativeType.CZ, backend=backend)


def test_u2_to_native(backend):
    gate = gates.U2(0, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate, two_qubit_natives=NativeType.CZ, backend=backend)


def test_u3_to_native(backend):
    gate = gates.U3(0, theta=0.2, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate, two_qubit_natives=NativeType.CZ, backend=backend)


def test_gpi2_to_native(backend):
    gate = gates.GPI2(0, phi=0.123)
    assert_matrices_allclose(gate, two_qubit_natives=NativeType.CZ, backend=backend)


@pytest.mark.parametrize("gatename", ["CNOT", "CZ", "SWAP", "iSWAP", "FSWAP"])
@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
def test_two_qubit_to_native(backend, gatename, natives):
    gate = getattr(gates, gatename)(0, 1)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
@pytest.mark.parametrize("gatename", ["CRX", "CRY", "CRZ"])
def test_controlled_rotations_to_native(backend, gatename, natives):
    gate = getattr(gates, gatename)(0, 1, 0.3)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
def test_cu1_to_native(backend, natives):
    gate = gates.CU1(0, 1, theta=0.4)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
def test_cu2_to_native(backend, natives):
    gate = gates.CU2(0, 1, phi=0.2, lam=0.3)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
def test_cu3_to_native(backend, natives):
    gate = gates.CU3(0, 1, theta=0.2, phi=0.3, lam=0.4)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
def test_fSim_to_native(backend, natives):
    gate = gates.fSim(0, 1, theta=0.3, phi=0.1)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
def test_GeneralizedfSim_to_native(backend, natives, seed):
    unitary = random_unitary(2, seed=seed, backend=backend)
    gate = gates.GeneralizedfSim(0, 1, unitary, phi=0.1)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
@pytest.mark.parametrize("gatename", ["RXX", "RZZ", "RYY"])
def test_rnn_to_native(backend, gatename, natives):
    gate = getattr(gates, gatename)(0, 1, theta=0.1)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
def test_TOFFOLI_to_native(backend, natives):
    gate = gates.TOFFOLI(0, 1, 2)
    assert_matrices_allclose(gate, natives, backend)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
@pytest.mark.parametrize(
    "natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_to_native(backend, nqubits, natives, seed):
    u = random_unitary(2**nqubits, seed=seed, backend=NumpyBackend())
    # transform to SU(2^nqubits) form
    u = u / np.sqrt(np.linalg.det(u))
    gate = gates.Unitary(u, *range(nqubits))
    assert_matrices_allclose(gate, natives, backend)


def test_count_1q():
    from qibo.transpiler.unroller import cz_dec

    np.testing.assert_allclose(cz_dec.count_1q(gates.CNOT(0, 1)), 2)
    np.testing.assert_allclose(cz_dec.count_1q(gates.CRX(0, 1, 0.1)), 2)


def test_count_2q():
    from qibo.transpiler.unroller import cz_dec

    np.testing.assert_allclose(cz_dec.count_2q(gates.CNOT(0, 1)), 1)
    np.testing.assert_allclose(cz_dec.count_2q(gates.CRX(0, 1, 0.1)), 2)
