import numpy as np
import pytest
from qibo import gates

_EINSUM_BACKENDS = ["DefaultEinsum"]#, "MatmulEinsum"]


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_xgate_application_onequbit(einsum_choice):
    """Check applying one qubit gate to one qubit density matrix."""
    initial_rho = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    gate = gates.X(0).with_backend(einsum_choice)
    gate.nqubits = 1
    final_rho = gate(initial_rho).numpy()

    pauliX = np.array([[0, 1], [1, 0]])
    target_rho = pauliX.dot(initial_rho).dot(pauliX)

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_hgate_application_twoqubit(einsum_choice):
    """Check applying one qubit gate to two qubit density matrix."""
    initial_rho = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
    gate = gates.H(1).with_backend(einsum_choice)
    gate.nqubits = 2
    final_rho = gate(initial_rho.reshape(4 * (2,))).numpy().reshape((4, 4))

    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    matrix = np.kron(np.eye(2), matrix)
    target_rho = matrix.dot(initial_rho).dot(matrix)

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_rygate_application_twoqubit(einsum_choice):
    """Check applying non-hermitian one qubit gate to one qubit density matrix."""
    theta = 0.1234

    initial_rho = np.random.random((2, 2)) + 1j * np.random.random((2, 2))
    gate = gates.RY(0, theta=theta).with_backend(einsum_choice)
    gate.nqubits = 1
    final_rho = gate(initial_rho).numpy()

    phase = np.exp(1j * np.pi * theta / 2.0)
    matrix = phase * np.array([[phase.real, -phase.imag], [phase.imag, phase.real]])
    target_rho = matrix.dot(initial_rho).dot(matrix.T.conj())

    np.testing.assert_allclose(final_rho, target_rho)


@pytest.mark.parametrize("einsum_choice", _EINSUM_BACKENDS)
def test_czpowgate_application_twoqubit(einsum_choice):
    """Check applying two qubit gate to three qubit density matrix."""
    theta = 0.1234
    nqubits = 3
    shape = 2 * (2 ** nqubits,)

    initial_rho = np.random.random(shape) + 1j * np.random.random(shape)
    gate = gates.CRZ(0, 1, theta=theta).with_backend(einsum_choice)
    gate.nqubits = nqubits
    final_rho = gate(initial_rho.reshape(2 * nqubits * (2,))).numpy().reshape(shape)

    matrix = np.eye(4, dtype=np.complex128)
    matrix[3, 3] = np.exp(1j * np.pi * theta)
    matrix = np.kron(matrix, np.eye(2))
    target_rho = matrix.dot(initial_rho).dot(matrix.T.conj())

    np.testing.assert_allclose(final_rho, target_rho)
