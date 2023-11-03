import numpy as np
import pytest
from scipy.linalg import expm

from qibo import Circuit, gates, matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info.metrics import purity
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler.unitary_decompositions import (
    bell_basis,
    calculate_h_vector,
    calculate_psi,
    calculate_single_qubit_unitaries,
    cnot_decomposition,
    cnot_decomposition_light,
    magic_basis,
    magic_decomposition,
    to_bell_diagonal,
    two_qubit_decomposition,
)


def bell_unitary(hx, hy, hz):
    ham = (
        hx * np.kron(matrices.X, matrices.X)
        + hy * np.kron(matrices.Y, matrices.Y)
        + hz * np.kron(matrices.Z, matrices.Z)
    )
    return expm(-1j * ham)


def assert_single_qubits(backend, psi, ua, ub):
    """Assert UA, UB map the maximally entangled basis ``psi`` to the magic basis."""
    uaub = np.kron(ua, ub)
    for i, j in zip(range(4), [0, 1, 3, 2]):
        final_state = np.dot(uaub, psi[:, i])
        target_state = magic_basis[:, j]
        fidelity = np.abs(np.dot(np.conj(target_state), final_state))
        backend.assert_allclose(fidelity, 1)


def test_u3_decomposition(backend):
    theta, phi, lam = 0.1, 0.2, 0.3
    u3_matrix = gates.U3(0, theta, phi, lam).matrix(backend)

    rz1 = gates.RZ(0, phi).matrix(backend)
    rz2 = gates.RZ(0, theta).matrix(backend)
    rz3 = gates.RZ(0, lam).matrix(backend)
    rx1 = gates.RX(0, -np.pi / 2).matrix(backend)
    rx2 = gates.RX(0, np.pi / 2).matrix(backend)

    target_matrix = rz1 @ rx1 @ rz2 @ rx2 @ rz3

    backend.assert_allclose(u3_matrix, target_matrix)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_eigenbasis_entanglement(backend, seed):
    unitary = random_unitary(4, seed=seed, backend=backend)

    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            calculate_psi(unitary, backend=backend)
    else:
        """Check that the eigenvectors of UT_U are maximally entangled."""
        states, eigvals = calculate_psi(unitary, backend=backend)
        eigvals = backend.cast(eigvals, dtype=eigvals.dtype)
        backend.assert_allclose(np.abs(eigvals), np.ones(4))
        for state in np.transpose(states):
            state = backend.partial_trace(state, [1], 2)
            backend.assert_allclose(purity(state), 0.5)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_v_decomposition(backend, seed):
    """Check that V_A V_B |psi_k> = |phi_k> according to Lemma 1."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            calculate_psi(unitary, backend=backend)
    else:
        psi, _ = calculate_psi(unitary, backend=backend)
        va, vb = calculate_single_qubit_unitaries(psi)
        assert_single_qubits(backend, psi, va, vb)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_u_decomposition(backend, seed):
    r"""Check that U_A\dagger U_B\dagger |psi_k tilde> = |phi_k> according to Lemma 1."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            calculate_psi(unitary, backend=backend)
    else:
        psi, eigvals = calculate_psi(unitary, backend=backend)
        psi_tilde = np.conj(np.sqrt(eigvals)) * np.dot(unitary, psi)
        ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde)
        assert_single_qubits(backend, psi_tilde, ua_dagger, ub_dagger)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_ud_eigenvalues(backend, seed):
    """Check that U_d is diagonal in the Bell basis."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            magic_decomposition(unitary, backend=backend)
    else:
        ua, ub, ud, va, vb = magic_decomposition(unitary, backend=backend)

        unitary_recon = np.kron(ua, ub) @ ud @ np.kron(va, vb)
        backend.assert_allclose(unitary_recon, unitary)

        ud_bell = np.transpose(np.conj(bell_basis)) @ ud @ bell_basis
        ud_diag = np.diag(ud_bell)
        backend.assert_allclose(np.diag(ud_diag), ud_bell, atol=PRECISION_TOL)
        backend.assert_allclose(np.prod(ud_diag), 1)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_calculate_h_vector(backend, seed):
    unitary = random_unitary(4, seed=seed, backend=backend)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            magic_decomposition(unitary, backend=backend)
    else:
        _, _, ud, _, _ = magic_decomposition(unitary, backend=backend)
        ud_diag = to_bell_diagonal(ud, backend=backend)
        assert ud_diag is not None
        hx, hy, hz = calculate_h_vector(ud_diag)
        target_matrix = bell_unitary(hx, hy, hz)
        backend.assert_allclose(ud, target_matrix, atol=PRECISION_TOL)


def test_cnot_decomposition(backend):
    hx, hy, hz = np.random.random(3)
    target_matrix = bell_unitary(hx, hy, hz)
    c = Circuit(2)
    c.add(cnot_decomposition(0, 1, hx, hy, hz))
    final_matrix = c.unitary(backend)
    backend.assert_allclose(final_matrix, target_matrix, atol=PRECISION_TOL)


def test_cnot_decomposition_light(backend):
    hx, hy = np.random.random(2)
    target_matrix = bell_unitary(hx, hy, 0)
    c = Circuit(2)
    c.add(cnot_decomposition_light(0, 1, hx, hy))
    final_matrix = c.unitary(backend)
    backend.assert_allclose(final_matrix, target_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_two_qubit_decomposition(backend, seed):
    unitary = random_unitary(4, seed=seed, backend=backend)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            two_qubit_decomposition(0, 1, unitary, backend=backend)
    else:
        c = Circuit(2)
        c.add(two_qubit_decomposition(0, 1, unitary, backend=backend))
        final_matrix = c.unitary(backend)
        backend.assert_allclose(final_matrix, unitary, atol=PRECISION_TOL)


@pytest.mark.parametrize("gatename", ["CNOT", "CZ", "SWAP", "iSWAP", "fSim", "I"])
def test_two_qubit_decomposition_common_gates(backend, gatename):
    """Test general two-qubit decomposition on some common gates."""
    if gatename == "fSim":
        gate = gates.fSim(0, 1, theta=0.1, phi=0.2)
    else:
        gate = getattr(gates, gatename)(0, 1)
    matrix = gate.matrix(backend)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"] and gatename != "iSWAP":
        with pytest.raises(NotImplementedError):
            two_qubit_decomposition(0, 1, matrix, backend=backend)
    else:
        c = Circuit(2)
        c.add(two_qubit_decomposition(0, 1, matrix, backend=backend))
        final_matrix = c.unitary(backend)
        backend.assert_allclose(final_matrix, matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("hz_zero", [False, True])
def test_two_qubit_decomposition_bell_unitary(backend, hz_zero):
    hx, hy, hz = (2 * np.random.random(3) - 1) * np.pi
    if hz_zero:
        hz = 0
    unitary = bell_unitary(hx, hy, hz)
    c = Circuit(2)
    c.add(two_qubit_decomposition(0, 1, unitary))
    final_matrix = c.unitary(backend)
    backend.assert_allclose(final_matrix, unitary, atol=PRECISION_TOL)


def test_two_qubit_decomposition_no_entanglement(backend):
    """Test two-qubit decomposition on unitary that creates no entanglement."""
    matrix = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    c = Circuit(2)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            two_qubit_decomposition(0, 1, matrix, backend=backend)
    else:
        c.add(two_qubit_decomposition(0, 1, matrix, backend=backend))
        final_matrix = c.unitary(backend)
        backend.assert_allclose(final_matrix, matrix, atol=PRECISION_TOL)
