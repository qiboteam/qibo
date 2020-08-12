import numpy as np
import pytest
from qibo.hamiltonians import Hamiltonian, XXZ, TFIM, Y
from qibo.tensorflow.hamiltonians import NUMERIC_TYPES


def test_hamiltonian_initialization():
    """Testing hamiltonian initialization errors."""
    H1 = Hamiltonian(2, np.eye(4))
    with pytest.raises(RuntimeError):
        H2 = Hamiltonian(np.eye(2), np.eye(4))
    with pytest.raises(ValueError):
        H3 = Hamiltonian(4, np.eye(10))


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_overloading(dtype, numpy):
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

    H1 = XXZ(nqubits=2, delta=0.5, numpy=numpy)
    H2 = XXZ(nqubits=2, delta=1, numpy=numpy)

    hH1 = transformation_a(H1.matrix, H2.matrix)
    hH2 = transformation_b(H1.matrix, H2.matrix)
    hH3 = transformation_c(H1.matrix, H2.matrix, use_eye=True)
    hH4 = transformation_d(H1.matrix, H2.matrix, use_eye=True)

    HT1 = transformation_a(H1, H2)
    HT2 = transformation_b(H1, H2)
    HT3 = transformation_c(H1, H2)
    HT4 = transformation_d(H1, H2)

    np.testing.assert_allclose(hH1, HT1.matrix)
    np.testing.assert_allclose(hH2, HT2.matrix)
    np.testing.assert_allclose(hH3, HT3.matrix)
    np.testing.assert_allclose(hH4, HT4.matrix)


@pytest.mark.parametrize("numpy", [True, False])
def test_different_hamiltonian_addition(numpy):
    """Test adding Hamiltonians of different models."""
    H1 = Y(nqubits=3, numpy=numpy)
    H2 = TFIM(nqubits=3, h=1.0, numpy=numpy)
    H = H1 + H2
    matrix = H1.matrix + H2.matrix
    np.testing.assert_allclose(H.matrix, matrix)
    H = H1 - 0.5 * H2
    matrix = H1.matrix - 0.5 * H2.matrix
    np.testing.assert_allclose(H.matrix, matrix)


@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_matmul(numpy):
    """Test matrix multiplication between Hamiltonians and state vectors."""
    H1 = TFIM(nqubits=3, h=1.0, numpy=numpy)
    H2 = Y(nqubits=3, numpy=numpy)
    if numpy:
        m1 = H1.matrix
        m2 = H2.matrix
    else:
        m1 = H1.matrix.numpy()
        m2 = H2.matrix.numpy()

    np.testing.assert_allclose((H1 @ H2).matrix, m1 @ m2)
    np.testing.assert_allclose((H2 @ H1).matrix, m2 @ m1)

    v = (np.random.random(8) + 1j * np.random.random(8)).astype(m1.dtype)
    m = (np.random.random((8, 8)) + 1j * np.random.random((8, 8))).astype(m1.dtype)
    np.testing.assert_allclose(H1 @ v, m1.dot(v))
    np.testing.assert_allclose(H1 @ m, m1 @ m)

    with pytest.raises(ValueError):
        H1 @ np.zeros((8, 8, 8), dtype=m1.dtype)
    with pytest.raises(NotImplementedError):
        H1 @ 2


@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_exponentiation(numpy):
    from scipy.linalg import expm
    H = XXZ(nqubits=2, delta=0.5, numpy=numpy)
    if numpy:
        target_matrix = expm(-0.5j * H.matrix)
    else:
        target_matrix = expm(-0.5j * H.matrix.numpy())
    np.testing.assert_allclose(H.exp(0.5), target_matrix)

    _ = H.eigenvectors()
    np.testing.assert_allclose(H.exp(0.5), target_matrix)


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
    with pytest.raises(NotImplementedError):
        R = H1 + "a"
    with pytest.raises(NotImplementedError):
        R = H2 - (2,)
    with pytest.raises(NotImplementedError):
        R = [3] - H1


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_hamiltonian_eigenvalues(dtype):
    """Testing hamiltonian eigenvalues scaling."""
    H1 = XXZ(nqubits=2, delta=0.5)

    H1_eigen = H1.eigenvalues()
    hH1_eigen = np.linalg.eigvalsh(H1.matrix)
    np.testing.assert_allclose(H1_eigen, hH1_eigen)

    c1 = dtype(2.5)
    H2 = c1 * H1
    H2_eigen = H2._eigenvalues
    hH2_eigen = np.linalg.eigvalsh(c1 * H1.matrix)
    np.testing.assert_allclose(H2._eigenvalues, hH2_eigen)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    H3_eigen = H3._eigenvalues
    hH3_eigen = np.linalg.eigvalsh(H1.matrix * c2)
    np.testing.assert_allclose(H3._eigenvalues, hH3_eigen)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_hamiltonian_eigenvectors(dtype):
    """Testing hamiltonian eigenvectors scaling."""
    H1 = XXZ(nqubits=2, delta=0.5)

    V1 = H1.eigenvectors()
    U1 = H1.eigenvalues()
    np.testing.assert_allclose(H1.matrix, V1 @ np.diag(U1) @ V1.T)

    c1 = dtype(2.5)
    H2 = c1 * H1
    V2 = H2._eigenvectors
    U2 = H2._eigenvalues
    np.testing.assert_allclose(H2.matrix, V2 @ np.diag(U2) @ V2.T)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    V3 = H3.eigenvectors()
    U3 = H3._eigenvalues
    np.testing.assert_allclose(H3.matrix, V3 @ np.diag(U3) @ V3.T)

    c3 = dtype(0)
    H4 = c3 * H1
    V4 = H4._eigenvectors
    U4 = H4._eigenvalues
    np.testing.assert_allclose(H4.matrix, V4 @ np.diag(U4) @ V4.T)
