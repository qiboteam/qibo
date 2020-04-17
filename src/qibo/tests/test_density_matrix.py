import numpy as np
import pytest
from qibo import gates, models

_EINSUM_BACKENDS = ["DefaultEinsum"]#, "MatmulEinsum"]


def random_density_matrix(nqubits: int) -> np.ndarray:
    shape = 2 * (2 ** nqubits,)
    rho = np.random.random(shape) + 1j * np.random.random(shape)
    # Make Hermitian
    rho = (rho + rho.T.conj()) / 2.0
    # Normalize
    ids = np.arange(2 ** nqubits)
    rho[ids, ids] = rho[ids, ids] / np.trace(rho)
    return rho


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_xgate_application_onequbit(einsum_choice):
    """Check applying one qubit gate to one qubit density matrix."""
    initial_rho = random_density_matrix(1)
    gate = gates.X(0).with_backend(einsum_choice)
    final_rho = gate(initial_rho, is_density_matrix=True).numpy()

    pauliX = np.array([[0, 1], [1, 0]])
    target_rho = pauliX.dot(initial_rho).dot(pauliX)

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_hgate_application_twoqubit(einsum_choice):
    """Check applying one qubit gate to two qubit density matrix."""
    initial_rho = random_density_matrix(2)
    gate = gates.H(1).with_backend(einsum_choice)
    final_rho = gate(initial_rho.reshape(4 * (2,)), is_density_matrix=True
                     ).numpy().reshape((4, 4))

    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    matrix = np.kron(np.eye(2), matrix)
    target_rho = matrix.dot(initial_rho).dot(matrix)

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_rygate_application_twoqubit(einsum_choice):
    """Check applying non-hermitian one qubit gate to one qubit density matrix."""
    theta = 0.1234
    initial_rho = random_density_matrix(1)

    gate = gates.RY(0, theta=theta).with_backend(einsum_choice)
    gate.nqubits = 1
    final_rho = gate(initial_rho, is_density_matrix=True).numpy()

    phase = np.exp(1j * np.pi * theta / 2.0)
    matrix = phase * np.array([[phase.real, -phase.imag], [phase.imag, phase.real]])
    target_rho = matrix.dot(initial_rho).dot(matrix.T.conj())

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_czpowgate_application_twoqubit(einsum_choice):
    """Check applying two qubit gate to three qubit density matrix."""
    theta = 0.1234
    initial_rho = random_density_matrix(3)

    gate = gates.CRZ(0, 1, theta=theta).with_backend(einsum_choice)
    final_rho = gate(initial_rho.reshape(6 * (2,)),
                     is_density_matrix=True).numpy().reshape(initial_rho.shape)

    matrix = np.eye(4, dtype=np.complex128)
    matrix[3, 3] = np.exp(1j * np.pi * theta)
    matrix = np.kron(matrix, np.eye(2))
    target_rho = matrix.dot(initial_rho).dot(matrix.T.conj())

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_circuit(einsum_choice):
    """Check passing density matrix as initial state to circuit."""
    theta = 0.1234
    initial_rho = random_density_matrix(3)

    c = models.Circuit(3)
    c.add(gates.X(2).with_backend(einsum_choice))
    c.add(gates.CRZ(0, 1, theta=theta).with_backend(einsum_choice))
    final_rho = c(initial_rho).numpy().reshape(initial_rho.shape)

    m1 = np.kron(np.eye(4), np.array([[0, 1], [1, 0]]))
    m2 = np.eye(4, dtype=np.complex128)
    m2[3, 3] = np.exp(1j * np.pi * theta)
    m2 = np.kron(m2, np.eye(2))
    target_rho = m1.dot(initial_rho).dot(m1)
    target_rho = m2.dot(target_rho).dot(m2.T.conj())

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_controlled_by_simple(einsum_choice):
    psi = np.zeros(4)
    psi[0] = 1
    initial_rho = np.outer(psi, psi.conj())

    c = models.Circuit(2)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.Y(1).with_backend(einsum_choice).controlled_by(0))
    final_rho = c(np.copy(initial_rho)).numpy()

    c = models.Circuit(2)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.Y(1).with_backend(einsum_choice))
    target_rho = c(np.copy(initial_rho)).numpy()

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_controlled_by_no_effect(einsum_choice):
    psi = np.zeros(2 ** 4)
    psi[0] = 1
    initial_rho = np.outer(psi, psi.conj())

    c = models.Circuit(4)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.SWAP(1, 3).with_backend(einsum_choice).controlled_by(0, 2))
    final_rho = c(np.copy(initial_rho)).numpy()

    c = models.Circuit(4)
    c.add(gates.X(0).with_backend(einsum_choice))
    target_rho = c(np.copy(initial_rho)).numpy()

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_controlled_with_effect(einsum_choice):
    psi = np.zeros(2 ** 4)
    psi[0] = 1
    initial_rho = np.outer(psi, psi.conj())

    c = models.Circuit(4)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.X(2).with_backend(einsum_choice))
    c.add(gates.SWAP(1, 3).with_backend(einsum_choice).controlled_by(0, 2))
    final_rho = c(np.copy(initial_rho)).numpy()

    c = models.Circuit(4)
    c.add(gates.X(0).with_backend(einsum_choice))
    c.add(gates.X(2).with_backend(einsum_choice))
    c.add(gates.SWAP(1, 3).with_backend(einsum_choice))
    target_rho = c(np.copy(initial_rho)).numpy()

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_bitflip_noise(einsum_choice):
    initial_rho = random_density_matrix(2)

    c = models.Circuit(2)
    c.add(gates.NoiseChannel(1, px=0.3).with_backend(einsum_choice))
    final_rho = c(np.copy(initial_rho)).numpy()

    c = models.Circuit(2)
    c.add(gates.X(1).with_backend(einsum_choice))
    target_rho = 0.3 * c(np.copy(initial_rho)).numpy()
    target_rho += 0.7 * initial_rho.reshape(target_rho.shape)

    np.testing.assert_allclose(final_rho, target_rho)
