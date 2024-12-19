"""Test methods in `qibo/core/hamiltonians.py`."""

import numpy as np
import pytest

from qibo import Circuit, gates, hamiltonians
from qibo.hamiltonians.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.quantum_info.random_ensembles import random_density_matrix, random_statevector
from qibo.symbols import I, X, Y, Z

from .utils import random_sparse_matrix


def test_hamiltonian_init(backend):
    with pytest.raises(TypeError):
        H = hamiltonians.Hamiltonian(2, "test", backend=backend)
    with pytest.raises(ValueError):
        H1 = hamiltonians.Hamiltonian(-2, np.eye(4), backend=backend)
    with pytest.raises(RuntimeError):
        H2 = hamiltonians.Hamiltonian(np.eye(2), np.eye(4), backend=backend)
    with pytest.raises(ValueError):
        H3 = hamiltonians.Hamiltonian(4, np.eye(10), backend=backend)


@pytest.mark.parametrize(
    "dtype",
    [
        int,
        float,
        complex,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ],
)
@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_hamiltonian_algebraic_operations(backend, dtype, sparse_type):
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
            return a + c1 * backend.to_numpy(backend.matrices.I(a.shape[0])) - b
        else:
            return a + c1 - b

    def transformation_d(a, b, use_eye=False):
        c1 = dtype(10.5)
        c2 = dtype(2)
        if use_eye:
            return c1 * backend.to_numpy(backend.matrices.I(a.shape[0])) - a + c2 * b
        else:
            return c1 - a + c2 * b

    if sparse_type is None:
        H1 = hamiltonians.XXZ(nqubits=2, delta=0.5, backend=backend)
        H2 = hamiltonians.XXZ(nqubits=2, delta=1, backend=backend)
        mH1, mH2 = backend.to_numpy(H1.matrix), backend.to_numpy(H2.matrix)
    else:
        if backend.platform in ["tensorflow", "pytorch"]:
            pytest.skip(
                "Tensorflow and Pytorch do not support operations with sparse matrices."
            )

        mH1 = random_sparse_matrix(backend, 64, sparse_type=sparse_type)
        mH2 = random_sparse_matrix(backend, 64, sparse_type=sparse_type)
        H1 = hamiltonians.Hamiltonian(6, mH1, backend=backend)
        H2 = hamiltonians.Hamiltonian(6, mH2, backend=backend)

    hH1 = transformation_a(mH1, mH2)
    hH2 = transformation_b(mH1, mH2)
    hH3 = transformation_c(mH1, mH2, use_eye=True)
    hH4 = transformation_d(mH1, mH2, use_eye=True)

    HT1 = transformation_a(H1, H2)
    HT2 = transformation_b(H1, H2)
    HT3 = transformation_c(H1, H2)
    HT4 = transformation_d(H1, H2)

    backend.assert_allclose(hH1, HT1.matrix)
    backend.assert_allclose(hH2, HT2.matrix)
    backend.assert_allclose(hH3, HT3.matrix)
    backend.assert_allclose(hH4, HT4.matrix)


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_hamiltonian_addition(backend, sparse_type):
    if sparse_type is None:
        H1 = hamiltonians.Y(nqubits=3, backend=backend)
        H2 = hamiltonians.TFIM(nqubits=3, h=1.0, backend=backend)
    else:
        if backend.platform in ["tensorflow", "pytorch"]:
            pytest.skip(
                "Tensorflow and Pytorch do not support operations with sparse matrices."
            )
        H1 = hamiltonians.Hamiltonian(
            6,
            random_sparse_matrix(backend, 64, sparse_type=sparse_type),
            backend=backend,
        )
        H2 = hamiltonians.Hamiltonian(
            6,
            random_sparse_matrix(backend, 64, sparse_type=sparse_type),
            backend=backend,
        )

    H = H1 + H2
    matrix = H1.matrix + H2.matrix
    backend.assert_allclose(H.matrix, matrix)
    H = H1 - 0.5 * H2
    matrix = H1.matrix - 0.5 * H2.matrix
    backend.assert_allclose(H.matrix, matrix)

    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5, backend=backend)
    H2 = hamiltonians.XXZ(nqubits=3, delta=0.1, backend=backend)
    with pytest.raises(RuntimeError):
        R = H1 + H2
    with pytest.raises(RuntimeError):
        R = H1 - H2


def test_hamiltonian_operation_errors(backend):
    """Testing hamiltonian not implemented errors."""
    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5, backend=backend)
    H2 = hamiltonians.XXZ(nqubits=2, delta=0.1, backend=backend)

    with pytest.raises(NotImplementedError):
        R = H1 * H2
    with pytest.raises(NotImplementedError):
        R = H1 + "a"
    with pytest.raises(NotImplementedError):
        R = H2 - (2,)
    with pytest.raises(NotImplementedError):
        R = [3] - H1


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_hamiltonian_matmul(backend, sparse_type):
    """Test matrix multiplication between Hamiltonians."""
    if backend.platform in ["tensorflow", "pytorch"]:
        pytest.skip(
            "Tensorflow and Pytorch do not support operations with sparse matrices."
        )
    if sparse_type is None:
        nqubits = 3
        H1 = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
        H2 = hamiltonians.Y(nqubits, backend=backend)
    else:
        nqubits = 5
        nstates = 2**nqubits
        H1 = hamiltonians.Hamiltonian(
            nqubits,
            random_sparse_matrix(backend, nstates, sparse_type),
            backend=backend,
        )
        H2 = hamiltonians.Hamiltonian(
            nqubits,
            random_sparse_matrix(backend, nstates, sparse_type),
            backend=backend,
        )

    m1 = backend.to_numpy(H1.matrix)
    m2 = backend.to_numpy(H2.matrix)
    if backend.platform == "tensorflow" and sparse_type is not None:
        with pytest.raises(NotImplementedError):
            _ = H1 @ H2
    else:
        backend.assert_allclose((H1 @ H2).matrix, (m1 @ m2))
        backend.assert_allclose((H2 @ H1).matrix, (m2 @ m1))

    with pytest.raises(ValueError):
        H1 @ np.zeros(3 * (2**nqubits,), dtype=m1.dtype)
    with pytest.raises(NotImplementedError):
        H1 @ 2


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_hamiltonian_matmul_states(backend, sparse_type):
    """Test matrix multiplication between Hamiltonian and states."""
    if sparse_type is None:
        nqubits = 3
        H = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    else:
        if backend.platform in ["tensorflow", "pytorch"]:
            pytest.skip(
                "Tensorflow and Pytorch do not support operations with sparse matrices."
            )
        nqubits = 3
        nstates = 2**nqubits
        matrix = random_sparse_matrix(backend, nstates, sparse_type)
        H = hamiltonians.Hamiltonian(nqubits, matrix, backend=backend)

    hm = H.matrix
    v = random_statevector(2**nqubits, backend=backend)
    v = backend.cast(v, dtype=hm.dtype)
    m = random_density_matrix(2**nqubits, backend=backend)
    m = backend.cast(m, dtype=hm.dtype)
    Hv = H @ backend.cast(v)
    Hm = H @ backend.cast(m)
    backend.assert_allclose(Hv, hm @ v)  # needs atol for cuquantum
    backend.assert_allclose(Hm, hm @ m)


@pytest.mark.parametrize("density_matrix", [True, False])
@pytest.mark.parametrize(
    "sparse_type,dense",
    [
        (None, True),
        (None, False),
        ("coo", True),
        ("csr", True),
        ("csc", True),
        ("dia", True),
    ],
)
def test_hamiltonian_expectation(backend, dense, density_matrix, sparse_type):
    """Test Hamiltonian expectation value calculation."""
    if sparse_type is None:
        h = hamiltonians.XXZ(nqubits=3, delta=0.5, dense=dense, backend=backend)
    else:
        if backend.platform in ["tensorflow", "pytorch"]:
            pytest.skip(
                "Tensorflow and Pytorch do not support operations with sparse matrices."
            )
        h = hamiltonians.Hamiltonian(
            6, random_sparse_matrix(backend, 64, sparse_type), backend=backend
        )

    matrix = backend.to_numpy(h.matrix)
    if density_matrix:
        state = random_density_matrix(2**h.nqubits, backend=backend)
        state = backend.to_numpy(state)
        state = state + state.T.conj()
        norm = np.trace(state)
        target_ev = np.trace(matrix.dot(state)).real
    else:
        state = random_statevector(2**h.nqubits, backend=backend)
        state = backend.to_numpy(state)
        norm = np.sum(np.abs(state) ** 2)
        target_ev = np.sum(state.conj() * matrix.dot(state)).real

    backend.assert_allclose(h.expectation(state), target_ev)
    backend.assert_allclose(h.expectation(state, True), target_ev / norm)


def test_hamiltonian_expectation_errors(backend):
    h = hamiltonians.XXZ(nqubits=3, delta=0.5, backend=backend)
    state = np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4)
    with pytest.raises(ValueError):
        h.expectation(state)
    with pytest.raises(TypeError):
        h.expectation("test")


def non_exact_expectation_test_setup(backend, observable):

    nqubits = 3
    c = Circuit(nqubits)
    for q in range(nqubits):
        c.add(gates.RX(q, np.random.rand()))

    H = hamiltonians.SymbolicHamiltonian(observable, nqubits=nqubits, backend=backend)
    final_state = backend.execute_circuit(c.copy(True)).state()
    exp = H.expectation(final_state)
    return exp, H, c


def test_hamiltonian_expectation_from_samples(backend):
    """Test Hamiltonian expectation value calculation."""
    backend.set_seed(12)

    nshots = 4 * 10**6
    observable = 2 * Z(0) * (1 - Z(1)) ** 2 + Z(0) * Z(2)
    exp, H, c = non_exact_expectation_test_setup(backend, observable)
    c.add(gates.M(*range(c.nqubits)))
    freq = backend.execute_circuit(c, nshots=nshots).frequencies()
    exp_from_samples = H.expectation_from_samples(freq)
    backend.assert_allclose(exp, exp_from_samples, atol=1e-2)


def test_hamiltonian_expectation_from_circuit(backend):
    """Test Hamiltonian expectation value calculation."""
    backend.set_seed(12)

    nshots = 4 * 10**6
    observable = I(0) * Z(1) + X(0) * Z(1) + Y(0) * X(2) / 2 - Z(0) * (1 - Y(1)) ** 3
    exp, H, c = non_exact_expectation_test_setup(backend, observable)
    exp_from_samples = H.expectation_from_circuit(c, nshots=nshots)
    backend.assert_allclose(exp, exp_from_samples, atol=1e-2)


def test_hamiltonian_expectation_from_samples_errors(backend):
    obs = random_density_matrix(4, backend=backend)
    h = hamiltonians.Hamiltonian(2, obs, backend=backend)
    with pytest.raises(NotImplementedError):
        h.expectation_from_samples(None, qubit_map=None)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize(
    "sparse_type,dense",
    [
        (None, True),
        (None, False),
        ("coo", True),
        ("csr", True),
        ("csc", True),
        ("dia", True),
    ],
)
def test_hamiltonian_eigenvalues(backend, dtype, sparse_type, dense):
    """Testing hamiltonian eigenvalues scaling."""
    if sparse_type is None:
        H1 = hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense, backend=backend)
    else:
        if backend.platform in ["tensorflow", "pytorch"]:
            pytest.skip(
                "Tensorflow and Pytorch do not support operations with sparse matrices."
            )
        from scipy import sparse

        H1 = hamiltonians.XXZ(nqubits=5, delta=0.5, backend=backend)
        m = getattr(sparse, f"{sparse_type}_matrix")(backend.to_numpy(H1.matrix))
        H1 = hamiltonians.Hamiltonian(5, m, backend=backend)

    H1_eigen = sorted(backend.to_numpy(H1.eigenvalues()))
    hH1_eigen = sorted(backend.to_numpy(backend.calculate_eigenvalues(H1.matrix)))
    backend.assert_allclose(sorted(H1_eigen), hH1_eigen)

    c1 = dtype(2.5)
    H2 = c1 * H1
    H2_eigen = sorted(backend.to_numpy(H2.eigenvalues()))
    hH2_eigen = sorted(backend.to_numpy(backend.calculate_eigenvalues(c1 * H1.matrix)))
    backend.assert_allclose(H2_eigen, hH2_eigen)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    if sparse_type is None:
        H3_eigen = sorted(backend.to_numpy(H3.eigenvalues()))
        hH3_eigen = sorted(
            backend.to_numpy(backend.calculate_eigenvalues(H1.matrix * c2))
        )
        backend.assert_allclose(H3_eigen, hH3_eigen)
    else:
        assert H3._eigenvalues is None


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
@pytest.mark.parametrize("dense", [True, False])
def test_hamiltonian_eigenvectors(backend, dtype, dense):
    """Testing hamiltonian eigenvectors scaling."""
    H1 = hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense, backend=backend)

    V1 = backend.to_numpy(H1.eigenvectors())
    U1 = backend.to_numpy(H1.eigenvalues())
    backend.assert_allclose(H1.matrix, V1 @ np.diag(U1) @ V1.T)

    c1 = dtype(2.5)
    H2 = c1 * H1
    V2 = backend.to_numpy(H2.eigenvectors())
    U2 = backend.to_numpy(H2.eigenvalues())
    backend.assert_allclose(H2.matrix, V2 @ np.diag(U2) @ V2.T)

    c2 = dtype(-11.1)
    H3 = H1 * c2
    V3 = backend.to_numpy(H3.eigenvectors())
    U3 = backend.to_numpy(H3.eigenvalues())
    backend.assert_allclose(H3.matrix, V3 @ np.diag(U3) @ V3.T)

    c3 = dtype(0)
    H4 = c3 * H1
    V4 = backend.to_numpy(H4.eigenvectors())
    U4 = backend.to_numpy(H4.eigenvalues())
    backend.assert_allclose(H4.matrix, V4 @ np.diag(U4) @ V4.T)


@pytest.mark.parametrize(
    "sparse_type,dense",
    [
        (None, True),
        (None, False),
        ("coo", True),
        ("csr", True),
        ("csc", True),
        ("dia", True),
    ],
)
def test_hamiltonian_ground_state(backend, sparse_type, dense):
    """Test Hamiltonian ground state."""
    if sparse_type is None:
        H = hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense, backend=backend)
    else:
        if backend.platform in ["tensorflow", "pytorch"]:
            pytest.skip(
                "Tensorflow and Pytorch do not support operations with sparse matrices."
            )
        from scipy import sparse

        H = hamiltonians.XXZ(nqubits=5, delta=0.5, backend=backend)
        m = getattr(sparse, f"{sparse_type}_matrix")(backend.to_numpy(H.matrix))
        H = hamiltonians.Hamiltonian(5, m, backend=backend)
    V = backend.to_numpy(H.eigenvectors())
    backend.assert_allclose(H.ground_state(), V[:, 0])


@pytest.mark.parametrize(
    "sparse_type,dense",
    [
        (None, True),
        (None, False),
        ("coo", True),
        ("csr", True),
        ("csc", True),
        ("dia", True),
    ],
)
def test_hamiltonian_exponentiation(backend, sparse_type, dense):
    """Test matrix exponentiation of Hamiltonians ``exp(1j * t * H)``."""
    from scipy.linalg import expm

    def construct_hamiltonian():
        if sparse_type is None:
            return hamiltonians.XXZ(nqubits=2, delta=0.5, dense=dense, backend=backend)
        else:
            if backend.platform in ["tensorflow", "pytorch"]:
                pytest.skip(
                    "Tensorflow and Pytorch do not support operations with sparse matrices."
                )
            from scipy import sparse

            ham = hamiltonians.XXZ(nqubits=5, delta=0.5, backend=backend)
            m = getattr(sparse, f"{sparse_type}_matrix")(backend.to_numpy(ham.matrix))
            return hamiltonians.Hamiltonian(5, m, backend=backend)

    H = construct_hamiltonian()
    target_matrix = expm(-0.5j * backend.to_numpy(H.matrix))
    H1 = construct_hamiltonian()
    _ = H1.eigenvectors()

    backend.assert_allclose(H.exp(0.5), target_matrix, atol=1e-6)
    backend.assert_allclose(H1.exp(0.5), target_matrix, atol=1e-6)


def test_hamiltonian_energy_fluctuation(backend):
    """Test energy fluctuation."""
    # define hamiltonian
    ham = hamiltonians.XXZ(nqubits=2, backend=backend)
    # take ground state and zero state
    ground_state = ham.ground_state()
    zero_state = backend.np.ones(2**2) / np.sqrt(2**2)
    # collect energy fluctuations
    gs_energy_fluctuation = ham.energy_fluctuation(ground_state)
    zs_energy_fluctuation = ham.energy_fluctuation(zero_state)

    assert np.isclose(backend.to_numpy(gs_energy_fluctuation), 0, atol=1e-5)
    assert gs_energy_fluctuation < zs_energy_fluctuation
