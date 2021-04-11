import numpy as np
import pytest
from qibo import matrices, K
from qibo.hamiltonians import Hamiltonian, TrotterHamiltonian
from qibo.hamiltonians import XXZ, TFIM, X, Y, Z, MaxCut
from qibo.tests import utils


def test_hamiltonian_initialization():
    """Testing hamiltonian initialization errors."""
    import tensorflow as tf
    dtype = K.dtypes('DTYPECPX')
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


@pytest.mark.parametrize("dtype", K.numeric_types)
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

    from qibo.core.states import VectorState
    state = VectorState.from_tensor(v)
    np.testing.assert_allclose(H1 @ state, m1.dot(v))

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
@pytest.mark.parametrize("density_matrix", [True, False])
def test_hamiltonian_expectation(numpy, trotter, density_matrix):
    h = XXZ(nqubits=3, delta=0.5, numpy=numpy, trotter=trotter)
    matrix = np.array(h.matrix)

    if density_matrix:
        state = utils.random_numpy_complex((8, 8))
        state = state + state.T.conj()
        norm = np.trace(state)
        target_ev = np.trace(matrix.dot(state)).real
    else:
        state = utils.random_numpy_complex(8)
        norm = np.sum(np.abs(state) ** 2)
        target_ev = np.sum(state.conj() * matrix.dot(state)).real

    np.testing.assert_allclose(h.expectation(state), target_ev)
    np.testing.assert_allclose(h.expectation(state, True), target_ev / norm)


def test_hamiltonian_expectation_errors():
    h = XXZ(nqubits=3, delta=0.5)
    state = utils.random_numpy_complex((4, 4, 4))
    with pytest.raises(ValueError):
        h.expectation(state)
    with pytest.raises(TypeError):
        h.expectation("test")


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


@pytest.mark.parametrize("dtype", K.numeric_types)
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


@pytest.mark.parametrize("dtype", K.numeric_types)
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


models_config = [
    (TFIM, {"nqubits": 3, "h": 0.0}, "tfim_N3h0.0.out"),
    (TFIM, {"nqubits": 3, "h": 0.5}, "tfim_N3h0.5.out"),
    (TFIM, {"nqubits": 3, "h": 1.0}, "tfim_N3h1.0.out"),
    (XXZ, {"nqubits": 3, "delta": 0.0}, "heisenberg_N3delta0.0.out"),
    (XXZ, {"nqubits": 3, "delta": 0.5}, "heisenberg_N3delta0.5.out"),
    (XXZ, {"nqubits": 3, "delta": 1.0}, "heisenberg_N3delta1.0.out"),
    (X, {"nqubits": 3}, "x_N3.out"),
    (Y, {"nqubits": 4}, "y_N4.out"),
    (Z, {"nqubits": 5}, "z_N5.out"),
    (MaxCut, {"nqubits": 3}, "maxcut_N3.out"),
    (MaxCut, {"nqubits": 4}, "maxcut_N4.out"),
    (MaxCut, {"nqubits": 5}, "maxcut_N5.out"),
]
@pytest.mark.parametrize(("model", "kwargs", "filename"), models_config)
@pytest.mark.parametrize("numpy", [True, False])
def test_tfim_model_hamiltonian(model, kwargs, filename, numpy):
    """Test pre-coded Hamiltonian models generate the proper matrices."""
    from qibo.tests_new.test_models_variational import assert_regression_fixture
    kwargs["numpy"] = numpy
    H = model(**kwargs)
    matrix = np.array(H.matrix).ravel().real
    assert_regression_fixture(matrix, filename)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("model", [TFIM, XXZ, Y, MaxCut])
def test_trotter_hamiltonian_to_dense(nqubits, model):
    """Test that Trotter Hamiltonian dense form agrees with normal Hamiltonian."""
    local_ham = model(nqubits, trotter=True)
    target_ham = model(nqubits, numpy=True)
    final_ham = local_ham.dense
    np.testing.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


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
    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(trotter_ev, target_ev)

    state = utils.random_numpy_complex((2 ** nqubits,))
    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(trotter_ev, target_ev)

    from qibo.core.states import VectorState
    state = VectorState.from_tensor(state)
    trotter_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    np.testing.assert_allclose(trotter_matmul, target_matmul)


def test_trotter_hamiltonian_three_qubit_term(backend):
    """Test creating ``TrotterHamiltonian`` with three qubit term."""
    import qibo
    from scipy.linalg import expm
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    m1 = utils.random_numpy_hermitian(3)
    m2 = utils.random_numpy_hermitian(2)
    m3 = utils.random_numpy_hermitian(1)

    term1 = Hamiltonian(3, m1, numpy=True)
    term2 = Hamiltonian(2, m2, numpy=True)
    term3 = Hamiltonian(1, m3, numpy=True)
    parts = [{(0, 1, 2): term1}, {(2, 3): term2, (1,): term3}]
    trotter_h = TrotterHamiltonian(*parts)

    # Test that the `TrotterHamiltonian` dense matrix is correct
    eye = np.eye(2, dtype=m1.dtype)
    mm1 = np.kron(m1, eye)
    mm2 = np.kron(np.kron(eye, eye), m2)
    mm3 = np.kron(np.kron(eye, m3), np.kron(eye, eye))
    target_h = Hamiltonian(4, mm1 + mm2 + mm3)
    np.testing.assert_allclose(trotter_h.dense.matrix, target_h.matrix)

    dt = 1e-2
    initial_state = utils.random_numpy_state(4)
    if backend == "custom":
        with pytest.raises(NotImplementedError):
            circuit = trotter_h.circuit(dt=dt)
    else:
        circuit = trotter_h.circuit(dt=dt)
        final_state = circuit(np.copy(initial_state))

        u = [expm(-0.5j * dt * m) for m in [mm1, mm2, mm3]]
        target_state = u[2].dot(u[1].dot(u[0])).dot(initial_state)
        target_state = u[0].dot(u[1].dot(u[2])).dot(target_state)
        np.testing.assert_allclose(final_state, target_state)

    qibo.set_backend(original_backend)


def test_trotter_hamiltonian_make_compatible_simple():
    """Test ``make_compatible`` on a simple 3-qubit example."""
    h0target = X(3)
    h0 = X(3, trotter=True)
    term1 = Y(1, numpy=True)
    term2 = TFIM(2, numpy=True)
    parts = [{(0, 1): term2, (1, 2): term2, (0, 2): term2, (2,): term1}]
    h1 = TrotterHamiltonian(*parts)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    np.testing.assert_allclose(h0c.matrix, h0target.matrix)


def test_trotter_hamiltonian_make_compatible_redundant():
    """Test ``make_compatible`` with redudant two-qubit terms."""
    h0 = X(2, trotter=True)
    target_matrix = h0.dense.matrix.numpy()
    target_matrix = np.kron(target_matrix, np.eye(2, dtype=target_matrix.dtype))
    parts = [{(0, 1, 2): TFIM(3, numpy=True)}]
    h1 = TrotterHamiltonian(*parts)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    np.testing.assert_allclose(h0c.matrix, target_matrix)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_trotter_hamiltonian_make_compatible(nqubits):
    """Test that ``make_compatible`` method works for ``X`` Hamiltonian."""
    h0target = X(nqubits)
    h0 = X(nqubits, trotter=True)
    h1 = XXZ(nqubits, delta=0.5, trotter=True)
    assert not h1.is_compatible(h0)
    assert not h0.is_compatible(h1)
    np.testing.assert_allclose(h0.matrix, h0target.matrix)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    assert h0c.is_compatible(h1)
    np.testing.assert_allclose(h0.matrix, h0target.matrix)
    np.testing.assert_allclose(h0c.matrix, h0target.matrix)
    # for coverage
    h0c = h1.make_compatible(h0c)
    assert not h1.is_compatible("test")
    h2 = XXZ(nqubits, delta=0.5, trotter=True)
    h2.parts[0].pop((0, 1))
    assert not h1.is_compatible(h2)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_trotter_hamiltonian_make_compatible_repeating(nqubits):
    """Check ``make_compatible`` when first target is repeated in parts."""
    h0target = X(nqubits)
    h0 = X(nqubits, trotter=True)
    term = TFIM(2, numpy=True)
    parts = [{(0, i): term} for i in range(1, nqubits)]
    parts.extend(({(i, 0): term} for i in range(1, nqubits)))
    h1 = TrotterHamiltonian(*parts)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    np.testing.assert_allclose(h0c.matrix, h0target.matrix)


def test_trotter_hamiltonian_make_compatible_onequbit_terms():
    """Check ``make_compatible`` when the two-qubit Hamiltonian has one-qubit terms."""
    term1 = Hamiltonian(1, matrices.Z, numpy=True)
    term2 = Hamiltonian(2, np.kron(matrices.Z, matrices.Z), numpy=True)
    terms = {(0, 1): term2,
             (0, 2): -0.5 * term2,
             (1, 2): 2 * term2,
             (1,): 0.35 * term1,
             (2, 3): 0.25 * term2,
             (2,): 0.5 * term1,
             (3,): term1}
    tham = TrotterHamiltonian.from_dictionary(terms) + 1.5
    xham = X(nqubits=4, trotter=True)
    cxham = tham.make_compatible(xham)
    assert not tham.is_compatible(xham)
    assert tham.is_compatible(cxham)
    np.testing.assert_allclose(xham.dense.matrix, cxham.dense.matrix)


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
    # Different term Hamiltonian types
    h2 = TFIM(nqubits=2, numpy=False)
    with pytest.raises(TypeError):
        ham = TrotterHamiltonian({(0, 1): h, (1, 2): h2})


def test_trotter_hamiltonian_operation_errors():
    """Test errors in ``TrotterHamiltonian`` addition and subtraction."""
    # test addition with different number of parts
    h1 = TFIM(nqubits=5, trotter=True)
    term = TFIM(nqubits=2, numpy=True)
    h2 = TrotterHamiltonian({(0, 1): term, (2, 3): term, (4, 0): term},
                            {(1, 2): term, (3, 4): term})
    with pytest.raises(ValueError):
        h = h1 + h2
    # test subtraction with incompatible parts
    h2 = TrotterHamiltonian({(0, 1): term, (2, 3): term},
                            {(1, 2): term}, {(4, 0): term})
    with pytest.raises(ValueError):
        h = h1 - h2
    # test matmul with bad type
    with pytest.raises(NotImplementedError):
        s = h1 @ "abc"
    # test matmul with bad shape
    with pytest.raises(ValueError):
        s = h1 @ np.zeros((2, 2))
    # test ``make_compatible`` with non-Trotter Hamiltonian
    with pytest.raises(TypeError):
        h2 = h1.make_compatible("test")
    # test ``make_compatible`` with interacting Hamiltonian
    with pytest.raises(NotImplementedError):
        h2 = h1.make_compatible(h2)
    # test ``make_compatible`` with insufficient two-qubit terms
    h3 = X(nqubits=7, trotter=True)
    with pytest.raises(ValueError):
        h3 = h1.make_compatible(h3)


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("trotter", [False, True])
def test_tfim_hamiltonian_from_symbols(nqubits, trotter):
    """Check creating TFIM Hamiltonian using sympy."""
    import sympy
    h = 0.5
    z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(nqubits))))
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(nqubits))))

    symham = sum(z_symbols[i] * z_symbols[i + 1] for i in range(nqubits - 1))
    symham += z_symbols[0] * z_symbols[-1]
    symham += h * sum(x_symbols)
    symmap = {z: (i, matrices.Z) for i, z in enumerate(z_symbols)}
    symmap.update({x: (i, matrices.X) for i, x in enumerate(x_symbols)})

    target_matrix = TFIM(nqubits, h=h).matrix
    if trotter:
        trotter_ham = TrotterHamiltonian.from_symbolic(-symham, symmap)
        final_matrix = trotter_ham.dense.matrix
    else:
        full_ham = Hamiltonian.from_symbolic(-symham, symmap)
        final_matrix = full_ham.matrix
    np.testing.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("trotter", [False, True])
def test_from_symbolic_with_power(trotter):
    """Check ``from_symbolic`` when the expression contains powers."""
    import sympy
    z = sympy.symbols(" ".join((f"Z{i}" for i in range(3))))
    symham =  z[0] ** 2 - z[1] ** 2 + 3 * z[1] - 2 * z[0] * z[2] + + 1
    matrix = utils.random_numpy_hermitian(1)
    symmap = {x: (i, matrix) for i, x in enumerate(z)}
    if trotter:
        ham = TrotterHamiltonian.from_symbolic(symham, symmap)
        final_matrix = ham.dense.matrix
    else:
        ham = Hamiltonian.from_symbolic(symham, symmap)
        final_matrix = ham.matrix

    matrix2 = matrix.dot(matrix)
    eye = np.eye(2, dtype=matrix.dtype)
    target_matrix = np.kron(np.kron(matrix2, eye), eye)
    target_matrix -= np.kron(np.kron(eye, matrix2), eye)
    target_matrix += 3 * np.kron(np.kron(eye, matrix), eye)
    target_matrix -= 2 * np.kron(np.kron(matrix, eye), matrix)
    target_matrix += np.eye(8, dtype=matrix.dtype)
    np.testing.assert_allclose(final_matrix, target_matrix)


def test_from_symbolic_application_hamiltonian():
    """Check ``from_symbolic`` for a specific four-qubit Hamiltonian."""
    import sympy
    z1, z2, z3, z4 = sympy.symbols("z1 z2 z3 z4")
    symmap = {z: (i, matrices.Z) for i, z in enumerate([z1, z2, z3, z4])}
    symham = (z1 * z2 - 0.5 * z1 * z3 + 2 * z2 * z3 + 0.35 * z2
              + 0.25 * z3 * z4 + 0.5 * z3 + z4 - z1)
    # Check that Trotter dense matrix agrees will full Hamiltonian matrix
    fham = Hamiltonian.from_symbolic(symham, symmap)
    tham = TrotterHamiltonian.from_symbolic(symham, symmap)
    np.testing.assert_allclose(tham.dense.matrix, fham.matrix)
    # Check that no one-qubit terms exist in the Trotter Hamiltonian
    # (this means that merging was successful)
    first_targets = set()
    for part in tham.parts:
        for targets, term in part.items():
            first_targets.add(targets[0])
            assert len(targets) == 2
            assert term.nqubits == 2
    assert first_targets == set(range(4))
    # Check making an ``X`` Hamiltonian compatible with ``tham``
    xham = X(nqubits=4, trotter=True)
    cxham = tham.make_compatible(xham)
    assert not tham.is_compatible(xham)
    assert tham.is_compatible(cxham)
    np.testing.assert_allclose(xham.dense.matrix, cxham.dense.matrix)


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("trotter", [False, True])
def test_x_hamiltonian_from_symbols(nqubits, trotter):
    """Check creating sum(X) Hamiltonian using sympy."""
    import sympy
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(nqubits))))
    symham =  -sum(x_symbols)
    symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}

    target_matrix = X(nqubits).matrix
    if trotter:
        trotter_ham = TrotterHamiltonian.from_symbolic(symham, symmap)
        final_matrix = trotter_ham.dense.matrix
    else:
        full_ham = Hamiltonian.from_symbolic(symham, symmap)
        final_matrix = full_ham.matrix
    np.testing.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("trotter", [False, True])
def test_three_qubit_term_hamiltonian_from_symbols(trotter):
    """Check creating Hamiltonian with three-qubit interaction using sympy."""
    import sympy
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(4))))
    y_symbols = sympy.symbols(" ".join((f"Y{i}" for i in range(4))))
    z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(4))))
    symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}
    symmap.update({x: (i, matrices.Y) for i, x in enumerate(y_symbols)})
    symmap.update({x: (i, matrices.Z) for i, x in enumerate(z_symbols)})

    symham = x_symbols[0] * y_symbols[1] * z_symbols[2]
    symham += 0.5 * y_symbols[0] * z_symbols[1] * x_symbols[3]
    symham += z_symbols[0] * x_symbols[2]
    symham += -3 * x_symbols[1] * y_symbols[3]
    symham += y_symbols[2]
    symham += 1.5 * z_symbols[1]
    symham -= 2

    target_matrix = np.kron(np.kron(matrices.X, matrices.Y),
                            np.kron(matrices.Z, matrices.I))
    target_matrix += 0.5 * np.kron(np.kron(matrices.Y, matrices.Z),
                                   np.kron(matrices.I, matrices.X))
    target_matrix += np.kron(np.kron(matrices.Z, matrices.I),
                             np.kron(matrices.X, matrices.I))
    target_matrix += -3 * np.kron(np.kron(matrices.I, matrices.X),
                             np.kron(matrices.I, matrices.Y))
    target_matrix += np.kron(np.kron(matrices.I, matrices.I),
                             np.kron(matrices.Y, matrices.I))
    target_matrix += 1.5 * np.kron(np.kron(matrices.I, matrices.Z),
                                   np.kron(matrices.I, matrices.I))
    target_matrix -= 2 * np.eye(2**4, dtype=target_matrix.dtype)
    if trotter:
        trotter_ham = TrotterHamiltonian.from_symbolic(symham, symmap)
        final_matrix = trotter_ham.dense.matrix
    else:
        full_ham = Hamiltonian.from_symbolic(symham, symmap)
        final_matrix = full_ham.matrix
    np.testing.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("sufficient", [True, False])
def test_symbolic_hamiltonian_merge_one_qubit(sufficient):
    """Check that ``merge_one_qubit`` works both when two-qubit are sufficient and no."""
    import sympy
    from qibo.hamiltonians import SymbolicHamiltonian
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(5))))
    z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(5))))
    symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}
    symmap.update({x: (i, matrices.Z) for i, x in enumerate(z_symbols)})
    symham = sum(z_symbols[i] * z_symbols[i + 1] for i in range(4))
    symham += sum(x_symbols)
    if sufficient:
        symham += z_symbols[0] * z_symbols[-1]
    symham = SymbolicHamiltonian(symham, symmap)
    terms = {t: m for t, m in symham.partial_matrices()}
    merged = symham.merge_one_qubit(terms)

    two_qubit_keys = {(i, i + 1) for i in range(4)}
    if sufficient:
        target_matrix = (np.kron(matrices.Z, matrices.Z) +
                         np.kron(matrices.X, matrices.I))
        two_qubit_keys.add((4, 0))
        assert set(merged.keys()) == two_qubit_keys
        for matrix in merged.values():
            np.testing.assert_allclose(matrix, target_matrix)
    else:
        one_qubit_keys = {(i,) for i in range(5)}
        assert set(merged.keys()) == one_qubit_keys | two_qubit_keys
        target_matrix = matrices.X
        for t in one_qubit_keys:
            np.testing.assert_allclose(merged[t], target_matrix)
        target_matrix = np.kron(matrices.Z, matrices.Z)
        for t in two_qubit_keys:
            np.testing.assert_allclose(merged[t], target_matrix)


def test_symbolic_hamiltonian_errors():
    """Check errors raised by `SymbolicHamiltonian`."""
    import sympy
    from qibo.hamiltonians import SymbolicHamiltonian
    a, b = sympy.symbols("a b")
    ham = a * b
    # Bad hamiltonian type
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian("test", "test")
    # Bad symbol map type
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian(ham, "test")
    # Bad symbol map key
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian(ham, {"a": 2})
    # Bad symbol map value
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian(ham, {a: 2})
    with pytest.raises(ValueError):
        sh = SymbolicHamiltonian(ham, {a: (1, 2, 3)})
    # Missing symbol
    with pytest.raises(ValueError):
        sh = SymbolicHamiltonian(ham, {a: (0, matrices.X)})
    # Factor that cannot be parsed
    ham = a * b + sympy.cos(a) * b
    with pytest.raises(ValueError):
        sh = SymbolicHamiltonian(ham, {a: (0, matrices.X), b: (1, matrices.Z)})


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("numpy", [True, False])
def test_maxcut(nqubits, numpy):
    size = 2 ** nqubits
    ham = np.zeros(shape=(size, size), dtype=np.complex128)
    for i in range(nqubits):
        for j in range(nqubits):
            h = np.eye(1)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    h = np.kron(h, matrices.Z)
                else:
                    h = np.kron(h, matrices.I)
            M = np.eye(2**nqubits) - h
            ham += M
    target_ham = K.cast(- ham / 2)
    final_ham = MaxCut(nqubits, numpy=numpy)
    np.testing.assert_allclose(final_ham.matrix, target_ham)
