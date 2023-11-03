import numpy as np
import pytest
from scipy.linalg import expm

from qibo import gates, matrices
from qibo.backends import NumpyBackend
from qibo.models import Circuit
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

NREPS = 3  # number of repetitions to execute random tests
ATOL = 1e-12


# # TODO: confront with quantum_info function
# def purity(state, backend):
#     """Calculates the purity of the partial trace of a two-qubit state."""
#     # mat = np.reshape(state, (2, 2))
#     # reduced_rho = np.dot(mat, np.conj(mat.T))
#     reduced_rho = backend.calculate_
#     return np.trace(np.dot(reduced_rho, reduced_rho))


def bell_unitary(hx, hy, hz):
    ham = (
        hx * np.kron(matrices.X, matrices.X)
        + hy * np.kron(matrices.Y, matrices.Y)
        + hz * np.kron(matrices.Z, matrices.Z)
    )
    return expm(-1j * ham)


def assert_single_qubits(psi, ua, ub):
    """Assert UA, UB map the maximally entangled basis ``psi`` to the magic basis."""
    uaub = np.kron(ua, ub)
    for i, j in zip(range(4), [0, 1, 3, 2]):
        final_state = np.dot(uaub, psi[:, i])
        target_state = magic_basis[:, j]
        fidelity = np.abs(np.dot(np.conj(target_state), final_state))
        np.testing.assert_allclose(fidelity, 1)
        # np.testing.assert_allclose(final_state, target_state, atol=1e-12)


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
@pytest.mark.parametrize("run_number", range(NREPS))
def test_eigenbasis_entanglement(backend, run_number, seed):
    """Check that the eigenvectors of UT_U are maximally entangled."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    states, eigvals = calculate_psi(unitary)
    eigvals = backend.cast(eigvals, dtype=eigvals.dtype)
    backend.assert_allclose(np.abs(eigvals), np.ones(4))
    for state in np.transpose(states):
        state = backend.partial_trace(state, [1], 2)
        backend.assert_allclose(purity(state), 0.5)


@pytest.mark.parametrize("run_number", range(NREPS))
def test_v_decomposition(run_number):
    """Check that V_A V_B |psi_k> = |phi_k> according to Lemma 1."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    psi, eigvals = calculate_psi(unitary)
    va, vb = calculate_single_qubit_unitaries(psi)
    assert_single_qubits(psi, va, vb)


@pytest.mark.parametrize("run_number", range(NREPS))
def test_u_decomposition(run_number):
    r"""Check that U_A\dagger U_B\dagger |psi_k tilde> = |phi_k> according to Lemma 1."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    psi, eigvals = calculate_psi(unitary)
    psi_tilde = np.conj(np.sqrt(eigvals)) * np.dot(unitary, psi)
    ua_dagger, ub_dagger = calculate_single_qubit_unitaries(psi_tilde)
    assert_single_qubits(psi_tilde, ua_dagger, ub_dagger)


@pytest.mark.parametrize("run_number", range(NREPS))
def test_ud_eigenvalues(run_number):
    """Check that U_d is diagonal in the Bell basis."""
    unitary = random_unitary(4, seed=seed, backend=backend)
    ua, ub, ud, va, vb = magic_decomposition(unitary)

    unitary_recon = np.kron(ua, ub) @ ud @ np.kron(va, vb)
    np.testing.assert_allclose(unitary_recon, unitary)

    ud_bell = np.dot(np.dot(np.conj(bell_basis).T, ud), bell_basis)
    ud_diag = np.diag(ud_bell)
    np.testing.assert_allclose(np.diag(ud_diag), ud_bell, atol=ATOL)
    np.testing.assert_allclose(np.prod(ud_diag), 1)


@pytest.mark.parametrize("run_number", range(NREPS))
def test_calculate_h_vector(run_number):
    unitary = random_unitary(4, seed=seed, backend=backend)
    ua, ub, ud, va, vb = magic_decomposition(unitary)
    ud_diag = to_bell_diagonal(ud)
    assert ud_diag is not None
    hx, hy, hz = calculate_h_vector(ud_diag)
    target_matrix = bell_unitary(hx, hy, hz)
    np.testing.assert_allclose(ud, target_matrix, atol=ATOL)


@pytest.mark.parametrize("run_number", range(NREPS))
def test_cnot_decomposition(run_number):
    hx, hy, hz = np.random.random(3)
    target_matrix = bell_unitary(hx, hy, hz)
    c = Circuit(2)
    c.add(cnot_decomposition(0, 1, hx, hy, hz))
    final_matrix = c.unitary(NumpyBackend())
    np.testing.assert_allclose(final_matrix, target_matrix, atol=ATOL)


@pytest.mark.parametrize("run_number", range(NREPS))
def test_cnot_decomposition_light(run_number):
    hx, hy = np.random.random(2)
    target_matrix = bell_unitary(hx, hy, 0)
    c = Circuit(2)
    c.add(cnot_decomposition_light(0, 1, hx, hy))
    final_matrix = c.unitary(NumpyBackend())
    np.testing.assert_allclose(final_matrix, target_matrix, atol=ATOL)


@pytest.mark.parametrize("run_number", range(NREPS))
def test_two_qubit_decomposition(run_number):
    backend = NumpyBackend()
    unitary = random_unitary(4, seed=seed, backend=backend)
    c = Circuit(2)
    c.add(two_qubit_decomposition(0, 1, unitary))
    final_matrix = c.unitary(backend)
    np.testing.assert_allclose(final_matrix, unitary, atol=ATOL)


@pytest.mark.parametrize("gatename", ["CNOT", "CZ", "SWAP", "iSWAP", "fSim", "I"])
def test_two_qubit_decomposition_common_gates(gatename):
    """Test general two-qubit decomposition on some common gates."""
    backend = NumpyBackend()
    if gatename == "fSim":
        gate = gates.fSim(0, 1, theta=0.1, phi=0.2)
    else:
        gate = getattr(gates, gatename)(0, 1)
    matrix = gate.matrix(backend)
    c = Circuit(2)
    c.add(two_qubit_decomposition(0, 1, matrix))
    final_matrix = c.unitary(backend)
    np.testing.assert_allclose(final_matrix, matrix, atol=ATOL)


@pytest.mark.parametrize("run_number", range(NREPS))
@pytest.mark.parametrize("hz_zero", [False, True])
def test_two_qubit_decomposition_bell_unitary(run_number, hz_zero):
    backend = NumpyBackend()
    hx, hy, hz = (2 * np.random.random(3) - 1) * np.pi
    if hz_zero:
        hz = 0
    unitary = bell_unitary(hx, hy, hz)
    c = Circuit(2)
    c.add(two_qubit_decomposition(0, 1, unitary))
    final_matrix = c.unitary(backend)
    np.testing.assert_allclose(final_matrix, unitary, atol=ATOL)


def test_two_qubit_decomposition_no_entanglement():
    """Test two-qubit decomposition on unitary that creates no entanglement."""
    backend = NumpyBackend()
    matrix = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    c = Circuit(2)
    c.add(two_qubit_decomposition(0, 1, matrix))
    final_matrix = c.unitary(backend)
    np.testing.assert_allclose(final_matrix, matrix, atol=ATOL)
