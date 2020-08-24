import numpy as np
import pytest
from qibo.hamiltonians import Hamiltonian, TrotterHamiltonian
from qibo.hamiltonians import XXZ, TFIM, X, Y, Z
from qibo.tensorflow.hamiltonians import NUMERIC_TYPES
from qibo.tests import utils


def test_hamiltonian_initialization():
    """Testing hamiltonian initialization errors."""
    import tensorflow as tf
    from qibo.config import DTYPES
    dtype = DTYPES.get('DTYPECPX')
    with pytest.raises(TypeError):
        H = Hamiltonian(2, "test")
    H1 = Hamiltonian(2, np.eye(4))
    H1 = Hamiltonian(2, np.eye(4), numpy=True)
    H1 = Hamiltonian(2, tf.eye(4, dtype=dtype))
    H1 = Hamiltonian(2, tf.eye(4, dtype=dtype), numpy=True)
    with pytest.raises(ValueError):
        H1 = Hamiltonian(-2, np.eye(4))
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

    v = utils.random_numpy_complex(8, dtype=m1.dtype)
    m = utils.random_numpy_complex((8, 8), dtype=m1.dtype)
    np.testing.assert_allclose(H1 @ v, m1.dot(v))
    np.testing.assert_allclose(H1 @ m, m1 @ m)

    with pytest.raises(ValueError):
        H1 @ np.zeros((8, 8, 8), dtype=m1.dtype)
    with pytest.raises(NotImplementedError):
        H1 @ 2


@pytest.mark.parametrize("numpy", [True, False])
@pytest.mark.parametrize("trotter", [True, False])
def test_hamiltonian_exponentiation(numpy, trotter):
    from scipy.linalg import expm
    H = XXZ(nqubits=2, delta=0.5, numpy=numpy, trotter=trotter)
    target_matrix = expm(-0.5j * np.array(H.matrix))
    np.testing.assert_allclose(H.exp(0.5), target_matrix)

    H = XXZ(nqubits=2, delta=0.5, numpy=numpy, trotter=trotter)
    _ = H.eigenvectors()
    np.testing.assert_allclose(H.exp(0.5), target_matrix)


@pytest.mark.parametrize("numpy", [True, False])
@pytest.mark.parametrize("trotter", [True, False])
def test_hamiltonian_expectation(numpy, trotter):
    h = XXZ(nqubits=3, delta=0.5, numpy=numpy, trotter=trotter)
    matrix = np.array(h.matrix)

    state = utils.random_numpy_complex(8)
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
@pytest.mark.parametrize("trotter", [True, False])
def test_hamiltonian_eigenvalues(dtype, numpy, trotter):
    """Testing hamiltonian eigenvalues scaling."""
    H1 = XXZ(nqubits=2, delta=0.5, numpy=numpy, trotter=trotter)

    H1_eigen = H1.eigenvalues()
    hH1_eigen = np.linalg.eigvalsh(H1.matrix)
    np.testing.assert_allclose(H1_eigen, hH1_eigen)

    c1 = dtype(2.5)
    H2 = c1 * H1
    hH2_eigen = np.linalg.eigvalsh(c1 * H1.matrix)
    np.testing.assert_allclose(H2._eigenvalues, hH2_eigen)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    hH3_eigen = np.linalg.eigvalsh(H1.matrix * c2)
    np.testing.assert_allclose(H3._eigenvalues, hH3_eigen)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("numpy", [True, False])
@pytest.mark.parametrize("trotter", [True, False])
def test_hamiltonian_eigenvectors(dtype, numpy, trotter):
    """Testing hamiltonian eigenvectors scaling."""
    H1 = XXZ(nqubits=2, delta=0.5, numpy=numpy, trotter=trotter)

    V1 = np.array(H1.eigenvectors())
    U1 = np.array(H1.eigenvalues())
    np.testing.assert_allclose(H1.matrix, V1 @ np.diag(U1) @ V1.T)
    # Check ground state
    np.testing.assert_allclose(H1.ground_state(), V1[:, 0])

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


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("model", [TFIM, XXZ, Y])
def test_trotter_hamiltonian_to_dense(nqubits, model):
    """Test that Trotter Hamiltonian dense form agrees with normal Hamiltonian."""
    local_ham = model(nqubits, trotter=True)
    target_ham = model(nqubits, numpy=True)
    final_ham = local_ham.dense
    np.testing.assert_allclose(final_ham.matrix, target_ham.matrix)


def test_trotter_hamiltonian_scalar_mul(nqubits=3):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 * TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 * local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham * 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_scalar_add(nqubits=4):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 + TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 + local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham + 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_scalar_sub(nqubits=3):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 - TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 - local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = TFIM(nqubits, h=1.0, numpy=True) - 2
    local_ham = TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham - 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_operator_add_and_sub(nqubits=3):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham1 = TFIM(nqubits, h=1.0, trotter=True)
    local_ham2 = TFIM(nqubits, h=0.5, trotter=True)

    local_ham = local_ham1 + local_ham2
    target_ham = (TFIM(nqubits, h=1.0, numpy=True) +
                  TFIM(nqubits, h=0.5, numpy=True))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham = local_ham1 - local_ham2
    target_ham = (TFIM(nqubits, h=1.0, numpy=True) -
                  TFIM(nqubits, h=0.5, numpy=True))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
def test_trotter_hamiltonian_matmul(nqubits, normalize):
    """Test Trotter Hamiltonian expectation value."""
    local_ham = TFIM(nqubits, h=1.0, trotter=True)
    dense_ham = TFIM(nqubits, h=1.0)

    state = utils.random_tensorflow_complex((2 ** nqubits,))
    trotter_ev = dense_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(trotter_ev, target_ev)

    state = utils.random_numpy_complex((2 ** nqubits,))
    trotter_ev = dense_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(trotter_ev, target_ev)


def test_trotter_hamiltonian_initialization_errors():
    """Test errors in initialization of ``TrotterHamiltonian``."""
    # Wrong type of terms
    with pytest.raises(TypeError):
        ham = TrotterHamiltonian({(0, 1): "abc"})
    # Wrong type of parts
    with pytest.raises(TypeError):
        ham = TrotterHamiltonian([(0, 1)])
    # Wrong number of target qubits
    with pytest.raises(ValueError):
        ham = TrotterHamiltonian({(0, 1): TFIM(nqubits=3, numpy=True)})
    # Same targets multiple times
    h = TFIM(nqubits=2, numpy=True)
    with pytest.raises(ValueError):
        ham = TrotterHamiltonian({(0, 1): h}, {(0, 1): h})
    # Different term matrix types
    h2 = Hamiltonian(2, np.eye(4, dtype=np.float32), numpy=True)
    with pytest.raises(TypeError):
        ham = TrotterHamiltonian({(0, 1): h, (1, 2): h2})
    # ``from_twoqubit_term`` initialization with nqubits < 0
    with pytest.raises(ValueError):
        ham = TrotterHamiltonian.from_twoqubit_term(-2, h)
    # ``from_twoqubit_term`` initialization with more than 2 targets
    h = TFIM(nqubits=3, numpy=True)
    with pytest.raises(ValueError):
        ham = TrotterHamiltonian.from_twoqubit_term(4, h)


def test_trotter_hamiltonian_operation_errors():
    """Test errors in ``TrotterHamiltonian`` addition and subtraction."""
    # test addition with different number of parts
    h1 = TFIM(nqubits=5, trotter=True)
    term = TFIM(nqubits=2, numpy=True)
    h2 = TrotterHamiltonian({(0, 1): term, (2, 3): term},
                            {(1, 2): term, (3, 4): term},
                            {(4, 0): term})
    with pytest.raises(ValueError):
        h = h1 + h2
    # test subtraction with incompatible parts
    h2 = TrotterHamiltonian({(0, 1): term, (2, 3): term},
                            {(1, 2): term, (3, 4): term})
    with pytest.raises(ValueError):
        h = h1 - h2
    # test matmul with bad type
    with pytest.raises(NotImplementedError):
        s = h1 @ "abc"
    # test matmul with bad shape
    with pytest.raises(ValueError):
        s = h1 @ np.zeros((2, 2))


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
