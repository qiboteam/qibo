import numpy as np
import pytest
from qibo.hamiltonians import Hamiltonian, XXZ, TFIM, X, Y, Z
from qibo.tensorflow.hamiltonians import NUMERIC_TYPES
from qibo.tests import utils


@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_initialization(numpy):
    """Testing hamiltonian initialization errors."""
    if numpy:
        eye = lambda n: np.eye(n)
    else:
        import tensorflow as tf
        from qibo.config import DTYPES
        eye = lambda n: tf.eye(n, dtype=DTYPES.get('DTYPECPX'))

    with pytest.raises(TypeError):
        H = Hamiltonian(2, "test")
    H1 = Hamiltonian(2, eye(4))
    with pytest.raises(ValueError):
        H1 = Hamiltonian(-2, eye(4))
    with pytest.raises(RuntimeError):
        H2 = Hamiltonian(eye(2), eye(4))
    with pytest.raises(ValueError):
        H3 = Hamiltonian(4, eye(10))


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
def test_right_operations(numpy):
    """Tests operations not covered by ``test_hamiltonian_overloading``."""
    H1 = Y(nqubits=3, numpy=numpy)
    H2 = 2 + H1
    target_matrix = 2 * np.eye(8) + H1.matrix
    np.testing.assert_allclose(H2.matrix, target_matrix)
    H2 = H1 - 2
    target_matrix = H1.matrix - 2 * np.eye(8)
    np.testing.assert_allclose(H2.matrix, target_matrix)


@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_mul(numpy):
    """Test multiplication with ``np.array`` and ``tf.Tensor`` scalar."""
    import tensorflow as tf
    h = TFIM(nqubits=3, h=1.0, numpy=numpy)
    h2 = h * np.array(2)
    np.testing.assert_allclose(h2.matrix, 2 * np.array(h.matrix))
    _ = h.eigenvectors()
    h2 = h * tf.cast(2, dtype=tf.complex128)
    np.testing.assert_allclose(h2.matrix, 2 * np.array(h.matrix))


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
    target_matrix = expm(-0.5j * np.array(H.matrix))
    np.testing.assert_allclose(H.exp(0.5), target_matrix)

    H = XXZ(nqubits=2, delta=0.5, numpy=numpy)
    _ = H.eigenvectors()
    np.testing.assert_allclose(H.exp(0.5), target_matrix)


@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_expectation(numpy):
    h = XXZ(nqubits=3, delta=0.5, numpy=numpy)
    matrix = np.array(h.matrix)

    state = np.random.random(8) + 1j * np.random.random(8)
    norm = (np.abs(state) ** 2).sum()
    target_ev = (state.conj() * matrix.dot(state)).sum().real

    np.testing.assert_allclose(h.expectation(state), target_ev)
    np.testing.assert_allclose(h.expectation(state, True), target_ev / norm)


@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_runtime_errors(numpy):
    """Testing hamiltonian runtime errors."""
    H1 = XXZ(nqubits=2, delta=0.5, numpy=numpy)
    H2 = XXZ(nqubits=3, delta=0.1, numpy=numpy)

    with pytest.raises(RuntimeError):
        R = H1 + H2
    with pytest.raises(RuntimeError):
        R = H1 - H2


@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_notimplemented_errors(numpy):
    """Testing hamiltonian not implemented errors."""
    H1 = XXZ(nqubits=2, delta=0.5, numpy=numpy)
    H2 = XXZ(nqubits=2, delta=0.1, numpy=numpy)

    with pytest.raises(NotImplementedError):
        R = H1 * H2
    with pytest.raises(NotImplementedError):
        R = H1 + "a"
    with pytest.raises(NotImplementedError):
        R = H2 - (2,)
    with pytest.raises(NotImplementedError):
        R = [3] - H1


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_eigenvalues(dtype, numpy):
    """Testing hamiltonian eigenvalues scaling."""
    H1 = XXZ(nqubits=2, delta=0.5, numpy=numpy)

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
@pytest.mark.parametrize("numpy", [True, False])
def test_hamiltonian_eigenvectors(dtype, numpy):
    """Testing hamiltonian eigenvectors scaling."""
    H1 = XXZ(nqubits=2, delta=0.5, numpy=numpy)

    V1 = np.array(H1.eigenvectors())
    U1 = np.array(H1.eigenvalues())
    np.testing.assert_allclose(H1.matrix, V1 @ np.diag(U1) @ V1.T)

    c1 = dtype(2.5)
    H2 = c1 * H1
    V2 = np.array(H2._eigenvectors)
    U2 = np.array(H2._eigenvalues)
    np.testing.assert_allclose(H2.matrix, V2 @ np.diag(U2) @ V2.T)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    V3 = np.array(H3.eigenvectors())
    U3 = np.array(H3._eigenvalues)
    np.testing.assert_allclose(H3.matrix, V3 @ np.diag(U3) @ V3.T)

    c3 = dtype(0)
    H4 = c3 * H1
    V4 = np.array(H4._eigenvectors)
    U4 = np.array(H4._eigenvalues)
    np.testing.assert_allclose(H4.matrix, V4 @ np.diag(U4) @ V4.T)


models_config = [
    (TFIM, {"nqubits": 3, "h": 0.0}, "tfim_N3h0.0.out"),
    (TFIM, {"nqubits": 3, "h": 0.5}, "tfim_N3h0.5.out"),
    (TFIM, {"nqubits": 3, "h": 1.0}, "tfim_N3h1.0.out"),
    (XXZ, {"nqubits": 3, "delta": 0.0}, "heisenberg_N3delta0.0.out"),
    (XXZ, {"nqubits": 3, "delta": 0.5}, "heisenberg_N3delta0.5.out"),
    (XXZ, {"nqubits": 3, "delta": 1.0}, "heisenberg_N3delta1.0.out"),
    (X, {"nqubits": 3}, "x_N3.out"),
    (Y, {"nqubits": 4}, "y_N4.out"),
    (Z, {"nqubits": 5}, "z_N5.out")
]
@pytest.mark.parametrize(("model", "kwargs", "filename"), models_config)
@pytest.mark.parametrize("numpy", [True, False])
def test_tfim_model_hamiltonian(model, kwargs, filename, numpy):
    """Test pre-coded Hamiltonian models generate the proper matrices."""
    kwargs["numpy"] = numpy
    H = model(**kwargs)
    matrix = np.array(H.matrix).ravel().real
    utils.assert_regression_fixture(matrix, filename)
