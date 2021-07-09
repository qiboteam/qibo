"""Test methods in `qibo/core/hamiltonians.py`."""
import pytest
import numpy as np
from qibo import hamiltonians, K
from qibo.tests.utils import random_complex


def test_hamiltonian_init():
    with pytest.raises(TypeError):
        H = hamiltonians.Hamiltonian(2, "test")
    H1 = hamiltonians.Hamiltonian(2, np.eye(4))
    H1 = hamiltonians.Hamiltonian(2, np.eye(4))
    H1 = hamiltonians.Hamiltonian(2, K.eye(4))
    H1 = hamiltonians.Hamiltonian(2, K.eye(4))
    with pytest.raises(ValueError):
        H1 = hamiltonians.Hamiltonian(-2, np.eye(4))
    with pytest.raises(RuntimeError):
        H2 = hamiltonians.Hamiltonian(np.eye(2), np.eye(4))
    with pytest.raises(ValueError):
        H3 = hamiltonians.Hamiltonian(4, np.eye(10))


@pytest.mark.parametrize("dtype", K.numeric_types)
def test_hamiltonian_algebraic_operations(dtype):
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

    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5)
    H2 = hamiltonians.XXZ(nqubits=2, delta=1)

    mH1, mH2 = K.to_numpy(H1.matrix), K.to_numpy(H2.matrix)
    hH1 = transformation_a(mH1, mH2)
    hH2 = transformation_b(mH1, mH2)
    hH3 = transformation_c(mH1, mH2, use_eye=True)
    hH4 = transformation_d(mH1, mH2, use_eye=True)

    HT1 = transformation_a(H1, H2)
    HT2 = transformation_b(H1, H2)
    HT3 = transformation_c(H1, H2)
    HT4 = transformation_d(H1, H2)

    K.assert_allclose(hH1, HT1.matrix)
    K.assert_allclose(hH2, HT2.matrix)
    K.assert_allclose(hH3, HT3.matrix)
    K.assert_allclose(hH4, HT4.matrix)


def test_hamiltonian_addition():
    H1 = hamiltonians.Y(nqubits=3)
    H2 = hamiltonians.TFIM(nqubits=3, h=1.0)
    H = H1 + H2
    matrix = H1.matrix + H2.matrix
    K.assert_allclose(H.matrix, matrix)
    H = H1 - 0.5 * H2
    matrix = H1.matrix - 0.5 * H2.matrix
    K.assert_allclose(H.matrix, matrix)

    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5)
    H2 = hamiltonians.XXZ(nqubits=3, delta=0.1)
    with pytest.raises(RuntimeError):
        R = H1 + H2
    with pytest.raises(RuntimeError):
        R = H1 - H2


def test_hamiltonian_operation_errors():
    """Testing hamiltonian not implemented errors."""
    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5)
    H2 = hamiltonians.XXZ(nqubits=2, delta=0.1)

    with pytest.raises(NotImplementedError):
        R = H1 * H2
    with pytest.raises(NotImplementedError):
        R = H1 + "a"
    with pytest.raises(NotImplementedError):
        R = H2 - (2,)
    with pytest.raises(NotImplementedError):
        R = [3] - H1


def test_hamiltonian_matmul():
    """Test matrix multiplication between Hamiltonians and state vectors."""
    H1 = hamiltonians.TFIM(nqubits=3, h=1.0)
    H2 = hamiltonians.Y(nqubits=3)
    m1 = K.to_numpy(H1.matrix)
    m2 = K.to_numpy(H2.matrix)

    K.assert_allclose((H1 @ H2).matrix, m1 @ m2)
    K.assert_allclose((H2 @ H1).matrix, m2 @ m1)

    v = random_complex(8, dtype=m1.dtype)
    m = random_complex((8, 8), dtype=m1.dtype)
    H1v = H1 @ K.cast(v)
    H1m = H1 @ K.cast(m)
    K.assert_allclose(H1v, m1.dot(v))
    K.assert_allclose(H1m, m1 @ m)

    from qibo.core.states import VectorState
    H1state = H1 @ VectorState.from_tensor(K.cast(v))
    K.assert_allclose(H1state, m1.dot(v))

    with pytest.raises(ValueError):
        H1 @ np.zeros((8, 8, 8), dtype=m1.dtype)
    with pytest.raises(NotImplementedError):
        H1 @ 2


@pytest.mark.parametrize("dense", [True, False])
def test_hamiltonian_exponentiation(dense):
    from scipy.linalg import expm
    H = hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense)
    target_matrix = expm(-0.5j * np.array(H.matrix))
    K.assert_allclose(H.exp(0.5), target_matrix)

    H = hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense)
    _ = H.eigenvectors()
    K.assert_allclose(H.exp(0.5), target_matrix)


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("density_matrix", [True, False])
def test_hamiltonian_expectation(dense, density_matrix):
    h = hamiltonians.XXZ(nqubits=3, delta=0.5, dense=dense)
    matrix = K.to_numpy(h.matrix)

    if density_matrix:
        state = random_complex((8, 8))
        state = state + state.T.conj()
        norm = np.trace(state)
        target_ev = np.trace(matrix.dot(state)).real
    else:
        state = random_complex(8)
        norm = np.sum(np.abs(state) ** 2)
        target_ev = np.sum(state.conj() * matrix.dot(state)).real

    K.assert_allclose(h.expectation(state), target_ev)
    K.assert_allclose(h.expectation(state, True), target_ev / norm)


def test_hamiltonian_expectation_errors():
    h = hamiltonians.XXZ(nqubits=3, delta=0.5)
    state = random_complex((4, 4, 4))
    with pytest.raises(ValueError):
        h.expectation(state)
    with pytest.raises(TypeError):
        h.expectation("test")


@pytest.mark.parametrize("dtype", K.numeric_types)
@pytest.mark.parametrize("dense", [True, False])
def test_hamiltonian_eigenvalues(dtype, dense):
    """Testing hamiltonian eigenvalues scaling."""
    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense)

    H1_eigen = H1.eigenvalues()
    hH1_eigen = np.linalg.eigvalsh(H1.matrix)
    K.assert_allclose(H1_eigen, hH1_eigen)

    c1 = dtype(2.5)
    H2 = c1 * H1
    hH2_eigen = np.linalg.eigvalsh(c1 * H1.matrix)
    K.assert_allclose(H2._eigenvalues, hH2_eigen)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    hH3_eigen = np.linalg.eigvalsh(H1.matrix * c2)
    K.assert_allclose(H3._eigenvalues, hH3_eigen)


@pytest.mark.parametrize("dtype", K.numeric_types)
@pytest.mark.parametrize("dense", [True, False])
def test_hamiltonian_eigenvectors(dtype, dense):
    """Testing hamiltonian eigenvectors scaling."""
    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense)

    V1 = K.to_numpy(H1.eigenvectors())
    U1 = K.to_numpy(H1.eigenvalues())
    K.assert_allclose(H1.matrix, V1 @ np.diag(U1) @ V1.T)
    # Check ground state
    K.assert_allclose(H1.ground_state(), V1[:, 0])

    c1 = dtype(2.5)
    H2 = c1 * H1
    V2 = K.to_numpy(H2._eigenvectors)
    U2 = K.to_numpy(H2._eigenvalues)
    K.assert_allclose(H2.matrix, V2 @ np.diag(U2) @ V2.T)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    V3 = K.to_numpy(H3.eigenvectors())
    U3 = K.to_numpy(H3._eigenvalues)
    K.assert_allclose(H3.matrix, V3 @ np.diag(U3) @ V3.T)

    c3 = dtype(0)
    H4 = c3 * H1
    V4 = K.to_numpy(H4._eigenvectors)
    U4 = K.to_numpy(H4._eigenvalues)
    K.assert_allclose(H4.matrix, V4 @ np.diag(U4) @ V4.T)
