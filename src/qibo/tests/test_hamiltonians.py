import numpy as np
import pytest
from qibo.hamiltonians import Hamiltonian, XXZ, NUMERIC_TYPES


def test_hamiltonian_initialization():
    """Testing hamiltonian not implemented errors."""
    H1 = Hamiltonian(nqubits=2)

    with pytest.raises(RuntimeError):
        H2 = Hamiltonian(np.eye(2))


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_hamiltonian_overloading(dtype):
    """Test basic hamiltonian overloading."""

    def transformation_a(a, b):
        c1 = dtype(0.1)
        return a + c1 * b
    def transformation_b(a, b):
        c1 = dtype(2)
        c2 = dtype(3.5)
        return c1 * a - b * c2
    def transformation_c(a, b, use_eye=False):
        c1 = dtype(4.5)
        if use_eye:
            return a + c1 * np.eye(a.shape[0]) - b
        else:
            return a + c1 - b
    def transformation_d(a, b, use_eye=False):
        c1 = dtype(10.5)
        c2 = dtype(2)
        if use_eye:
            return c1 * np.eye(a.shape[0]) - a + c2 * b
        else:
            return c1 - a + c2 * b

    H1 = XXZ(nqubits=2, delta=0.5)
    H2 = XXZ(nqubits=2, delta=1)

    hH1 = transformation_a(H1.hamiltonian, H2.hamiltonian)
    hH2 = transformation_b(H1.hamiltonian, H2.hamiltonian)
    hH3 = transformation_c(H1.hamiltonian, H2.hamiltonian, use_eye=True)
    hH4 = transformation_d(H1.hamiltonian, H2.hamiltonian, use_eye=True)

    HT1 = transformation_a(H1, H2)
    HT2 = transformation_b(H1, H2)
    HT3 = transformation_c(H1, H2)
    HT4 = transformation_d(H1, H2)

    np.testing.assert_allclose(hH1, HT1.hamiltonian)
    np.testing.assert_allclose(hH2, HT2.hamiltonian)
    np.testing.assert_allclose(hH3, HT3.hamiltonian)
    np.testing.assert_allclose(hH4, HT4.hamiltonian)


def test_hamiltonian_runtime_errors():
    """Testing hamiltonian runtime errors."""
    H1 = XXZ(nqubits=2, delta=0.5)
    H2 = XXZ(nqubits=3, delta=0.1)

    with pytest.raises(RuntimeError):
        R = H1 + H2
    with pytest.raises(RuntimeError):
        R = H1 - H2


def test_hamiltonian_notimplemented_errors():
    """Testing hamiltonian not implemented errors."""
    H1 = XXZ(nqubits=2, delta=0.5)
    H2 = XXZ(nqubits=2, delta=0.1)

    with pytest.raises(NotImplementedError):
        R = H1 * H2


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_hamiltonian_eigenvalues(dtype):
    """Testing hamiltonian eigenvalues scaling."""
    H1 = XXZ(nqubits=2, delta=0.5)

    H1_eigen = H1.eigenvalues()
    hH1_eigen = np.linalg.eigvalsh(H1.hamiltonian)
    np.testing.assert_allclose(H1_eigen, hH1_eigen)

    c1 = dtype(2.5)
    H2 = c1 * H1
    H2_eigen = H2._eigenvalues
    hH2_eigen = np.linalg.eigvalsh(c1 * H1.hamiltonian)
    np.testing.assert_allclose(H2._eigenvalues, hH2_eigen)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    H3_eigen = H3._eigenvalues
    hH3_eigen = np.linalg.eigvalsh(H1.hamiltonian * c2)
    np.testing.assert_allclose(H3._eigenvalues, hH3_eigen)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_hamiltonian_eigenvectors(dtype):
    """Testing hamiltonian eigenvectors scaling."""
    H1 = XXZ(nqubits=2, delta=0.5)

    V1 = H1.eigenvectors().numpy()
    U1 = H1.eigenvalues().numpy()
    np.testing.assert_allclose(H1.hamiltonian, V1 @ np.diag(U1) @ V1.T)

    c1 = dtype(2.5)
    H2 = c1 * H1
    V2 = H2._eigenvectors.numpy()
    U2 = H2._eigenvalues.numpy()
    np.testing.assert_allclose(H2.hamiltonian, V2 @ np.diag(U2) @ V2.T)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    V3 = H3.eigenvectors().numpy()
    U3 = H3._eigenvalues.numpy()
    np.testing.assert_allclose(H3.hamiltonian, V3 @ np.diag(U3) @ V3.T)
