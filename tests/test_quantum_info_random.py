"""Tests for the quantum_info.random_ensembles module."""

from functools import reduce
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qibo import Circuit, gates, matrices
from qibo.config import PRECISION_TOL
from qibo.models.encodings import entangling_layer
from qibo.quantum_info.basis import pauli_to_comp_basis
from qibo.quantum_info.metrics import purity
from qibo.quantum_info.random_ensembles import (
    random_clifford,
    random_density_matrix,
    random_gaussian_matrix,
    random_hermitian,
    random_iqp,
    random_pauli,
    random_pauli_hamiltonian,
    random_quantum_channel,
    random_statevector,
    random_stochastic_matrix,
    random_unitary,
    uniform_sampling_U3,
)


@pytest.mark.parametrize("seed", [None, 10])
def test_uniform_sampling_U3(backend, seed):
    with pytest.raises(TypeError):
        uniform_sampling_U3("1", seed=seed, backend=backend)
    with pytest.raises(ValueError):
        uniform_sampling_U3(0, seed=seed, backend=backend)

    X = backend.cast(matrices.X, dtype=matrices.X.dtype)
    Y = backend.cast(matrices.Y, dtype=matrices.Y.dtype)
    Z = backend.cast(matrices.Z, dtype=matrices.Z.dtype)

    ngates = int(1e3)
    phases = uniform_sampling_U3(ngates, seed=seed, backend=backend)

    # expectation values in the 3 directions should be the same
    expectation_values = []
    for row in phases:
        row = [float(phase) for phase in row]
        circuit = Circuit(1)
        circuit.add(gates.U3(0, *row))
        state = backend.execute_circuit(circuit).state()

        expectation_values.append(
            [
                backend.conj(state) @ X @ state,
                backend.conj(state) @ Y @ state,
                backend.conj(state) @ Z @ state,
            ]
        )
    expectation_values = backend.cast(expectation_values)

    expectation_values = backend.mean(expectation_values, axis=0)

    backend.assert_allclose(expectation_values[0], expectation_values[1], atol=1e-1)
    backend.assert_allclose(expectation_values[0], expectation_values[2], atol=1e-1)


@pytest.mark.parametrize("seed", [None, 10])
def test_random_gaussian_matrix(backend, seed):
    if backend.platform != "tensorflow":
        # tensorflow raises a custom exception
        with pytest.raises(TypeError):
            dims = np.array([2])
            random_gaussian_matrix(dims, backend=backend)
        with pytest.raises(TypeError):
            dims = 2
            rank = np.array([2])
            random_gaussian_matrix(dims, rank, backend=backend)
        with pytest.raises((ValueError, RuntimeError)):
            dims = -1
            random_gaussian_matrix(dims, backend=backend)
    with pytest.raises(ValueError):
        dims, rank = 2, 4
        random_gaussian_matrix(dims, rank, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_gaussian_matrix(dims, seed=0.1, backend=backend)

    # just runs the function with no tests
    random_gaussian_matrix(4, seed=seed, backend=backend)
    # we should probably check whether the real and im part of the matrix
    # are gaussian


def test_random_hermitian(backend):

    if backend.platform != "tensorflow":
        global PRECISION_TOL
    else:
        PRECISION_TOL = 4e-7

    # test if function returns Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, backend=backend)
    matrix_dagger = backend.conj(matrix).T
    norm = float(backend.matrix_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    # test if function returns semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True, backend=backend)
    matrix_dagger = backend.conj(matrix).T
    norm = float(backend.matrix_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(backend.to_numpy(matrix))
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues >= 0), True)

    # test if function returns normalized Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, normalize=True, backend=backend)
    matrix_dagger = backend.conj(matrix).T
    norm = float(backend.matrix_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(backend.to_numpy(matrix))
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues <= 1), True)

    # test if function returns normalized and semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True, normalize=True, backend=backend)
    matrix_dagger = backend.conj(matrix).T
    norm = float(backend.vector_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(backend.to_numpy(matrix))
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues >= 0), True)
    backend.assert_allclose(all(eigenvalues <= 1), True)


@pytest.mark.parametrize("seed", [None, 10])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize(
    "architecture", ["all-to-all", "diagonal", "even_layer", "pyramid"]
)
def test_random_iqp(backend, architecture, closed_boundary, seed):
    nqubits = 4

    with pytest.raises(NotImplementedError):
        random_iqp(nqubits, architecture="nonexistent", seed=seed, backend=backend)

    circuit = random_iqp(
        nqubits, architecture, closed_boundary, seed=seed, backend=backend
    )

    nentangling = len(
        entangling_layer(nqubits, architecture, "RZZ", closed_boundary).queue
    )

    assert len(circuit.get_parameters()) == nqubits + nentangling
    assert len(circuit.queue) == 3 * nqubits + nentangling

    # Hadamard gates only at the beginning and at the end of the circuit
    names = [gate.__class__.__name__ for gate in circuit.queue]
    assert names[:nqubits] == ["H"] * nqubits
    assert names[-nqubits:] == ["H"] * nqubits
    assert set(names[nqubits:-nqubits]) == {"RZ", "RZZ"}


@pytest.mark.parametrize("measure", [None, "haar"])
def test_random_unitary(backend, measure):
    with pytest.raises(ValueError):
        dims = 2
        random_unitary(dims, measure="gaussian", backend=backend)

    # tests if operator is unitary (measure == "haar")
    dims = 4
    matrix = random_unitary(dims, measure=measure, backend=backend)
    matrix_dagger = backend.conj(matrix).T
    matrix_inv = backend.inv(matrix)
    norm = float(backend.matrix_norm(matrix_inv - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)


@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("rank", [None, 4])
@pytest.mark.parametrize("measure", [None, "haar", "bcsz"])
@pytest.mark.parametrize(
    "representation",
    [
        "chi",
        "chi-IZXY",
        "choi",
        "kraus",
        "liouville",
        "pauli",
        "pauli-IZXY",
        "stinespring",
    ],
)
def test_random_quantum_channel(backend, representation, measure, rank, order):
    with pytest.raises(TypeError):
        random_quantum_channel(4, representation=True, backend=backend)
    with pytest.raises(ValueError):
        random_quantum_channel(4, representation="Choi", backend=backend)
    with pytest.raises(ValueError):
        random_quantum_channel(4, measure="bcsz", order="system", backend=backend)

    # All subroutines are already tested elsewhere,
    # so here we only execute them once for coverage
    random_quantum_channel(
        4,
        rank=rank,
        representation=representation,
        measure=measure,
        order=order,
        backend=backend,
    )


@pytest.mark.parametrize("seed", [None, 1])
@pytest.mark.parametrize(
    "dtype", [None, "complex128", "complex64", "float64", "float32"]
)
def test_random_statevector(backend, dtype, seed):

    # tests if random statevector is a pure state
    dims = 4
    state = random_statevector(dims, dtype=dtype, seed=seed, backend=backend)
    assert abs(purity(state, backend=backend) - 1.0) < 9 * PRECISION_TOL

    if dtype is not None:
        dtype = getattr(backend.engine, dtype)
        state = random_statevector(dims, dtype=dtype, seed=seed, backend=backend)
        assert abs(purity(state, backend=backend) - 1.0) < 9 * PRECISION_TOL


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("basis", [None, "pauli-IXYZ", "pauli-IZXY"])
@pytest.mark.parametrize("metric", ["hilbert-schmidt", "ginibre", "bures"])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("dims", [2, 4])
def test_random_density_matrix(backend, dims, pure, metric, basis, normalize):
    with pytest.raises(ValueError):
        random_density_matrix(dims=2, rank=3, backend=backend)
    with pytest.raises(ValueError):
        random_density_matrix(dims=2, metric="gaussian", backend=backend)
    with pytest.raises(ValueError):
        random_density_matrix(dims=2, metric=metric, basis="Pauli")

    if basis is None and normalize is True:
        with pytest.raises(ValueError):
            random_density_matrix(dims=dims, normalize=True)
    else:
        norm_function = backend.matrix_norm if basis is None else backend.vector_norm
        state = random_density_matrix(
            dims,
            pure=pure,
            metric=metric,
            basis=basis,
            normalize=normalize,
            backend=backend,
        )
        if basis is None and normalize is False:
            backend.assert_allclose(
                np.real(np.trace(backend.to_numpy(state))) <= 1.0 + PRECISION_TOL, True
            )
            backend.assert_allclose(
                np.real(np.trace(backend.to_numpy(state))) >= 1.0 - PRECISION_TOL, True
            )
            backend.assert_allclose(
                purity(state, backend=backend) <= 1.0 + PRECISION_TOL, True
            )
            if pure is True:
                backend.assert_allclose(
                    purity(state, backend=backend) >= 1.0 - PRECISION_TOL, True
                )
            norm = np.abs(
                backend.to_numpy(norm_function(state - backend.conj(state).T, order=2))
            )
            backend.assert_allclose(norm < PRECISION_TOL, True)
        else:
            normalization = 1.0 if normalize is False else 1.0 / np.sqrt(dims)
            backend.assert_allclose(state[0], normalization)
            assert all(
                np.abs(backend.to_numpy(exp_value)) <= normalization
                for exp_value in state[1:]
            )


@pytest.mark.parametrize("nqubits,nsamples", zip((1, 2), (int(3e2), int(3e3))))
def test_random_clifford(backend, nqubits, nsamples):

    backend.set_seed(42)

    # errors tests
    with pytest.raises(TypeError):
        random_clifford(nqubits="1", backend=backend)
    with pytest.raises(ValueError):
        random_clifford(nqubits=-1, backend=backend)
    with pytest.raises(TypeError):
        random_clifford(nqubits, return_circuit="True", backend=backend)
    with pytest.raises(TypeError):
        random_clifford(nqubits, seed=0.1, backend=backend)

    n_cliffords = 24 if nqubits == 1 else 11520
    expected_prob = 1 / n_cliffords
    samples = {str(random_clifford(nqubits, return_circuit=False, backend=backend)): 1}
    for _ in range(nsamples):
        sample = str(random_clifford(nqubits, return_circuit=False, backend=backend))
        if sample in samples:
            samples[sample] += 1
        else:
            samples[sample] = 1

    avg_prob = (np.array(list(samples.values())) / (nsamples + 1)).mean()
    backend.assert_allclose(expected_prob, avg_prob, atol=1e-2, rtol=0.7)
    if nqubits == 1:
        assert len(samples) == n_cliffords
    else:
        assert int(1e3) <= len(samples) <= n_cliffords


def test_random_pauli_errors(backend):
    with pytest.raises(ValueError):
        q, depth = 1, 0
        random_pauli(q, depth, backend=backend)
    with pytest.raises(ValueError):
        q, depth, max_qubits = 4, 1, 3
        random_pauli(q, depth, max_qubits=max_qubits, backend=backend)
    with pytest.raises(ValueError):
        q, depth, max_qubits = [0, 5], 1, 3
        random_pauli(q, depth, max_qubits=max_qubits, backend=backend)
    with pytest.raises(TypeError):
        q, depth = 2, 1
        subset = ["I", 0]
        random_pauli(q, depth, subset=subset, backend=backend)


def test_pauli_single(backend):
    target = (
        backend.matrices.Z
        if backend.platform in ("cupy", "cuquantum")
        else backend.matrices.X
    )

    matrix = random_pauli(0, 1, 1, seed=10, backend=backend)
    matrix = matrix.unitary(backend=backend)
    matrix = backend.cast(matrix, dtype=matrix.dtype)

    backend.assert_allclose(matrix, target)


@pytest.mark.parametrize("qubits", [2, [0, 1]])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("max_qubits", [None])
@pytest.mark.parametrize("subset", [None, ["I", "X"]])
@pytest.mark.parametrize("return_circuit", [True, False])
@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("seed", [10])
def test_random_pauli(
    backend, qubits, depth, max_qubits, subset, return_circuit, density_matrix, seed
):
    result_complete_set = (
        backend.kron(backend.matrices.I(), backend.matrices.I())
        if backend.platform in ("cupy", "cuquantum")
        else backend.kron(backend.matrices.I(), backend.matrices.Z)
    )

    result_subset = (
        backend.kron(backend.matrices.I(), backend.matrices.I())
        if backend.platform in ("cupy", "cuquantum")
        else backend.kron(backend.matrices.I(), backend.matrices.X)
    )

    matrix = random_pauli(
        qubits, depth, max_qubits, subset, return_circuit, density_matrix, seed, backend
    )

    if return_circuit:
        matrix = matrix.unitary(backend=backend)
        if subset is None:
            backend.assert_allclose(matrix, result_complete_set)
        else:
            backend.assert_allclose(matrix, result_subset)
    else:
        matrix = backend.transpose(matrix, (1, 0, 2, 3))
        matrix = [reduce(backend.kron, row) for row in matrix]
        matrix = reduce(backend.matmul, matrix)

        if subset is None:
            backend.assert_allclose(
                float(backend.matrix_norm(matrix - result_complete_set, order=2))
                < PRECISION_TOL,
                True,
            )
        else:
            assert (
                float(backend.matrix_norm(matrix - result_subset, order=2))
                < PRECISION_TOL
            )


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("max_eigenvalue", [2, 3])
@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_random_pauli_hamiltonian(
    backend, nqubits, max_eigenvalue, normalize, pauli_order
):
    with pytest.raises(TypeError):
        random_pauli_hamiltonian(nqubits=[1], backend=backend)
    with pytest.raises(ValueError):
        random_pauli_hamiltonian(nqubits=0, backend=backend)
    with pytest.raises(TypeError):
        random_pauli_hamiltonian(nqubits=2, max_eigenvalue=[2], backend=backend)
    with pytest.raises(TypeError):
        random_pauli_hamiltonian(nqubits=2, normalize="True", backend=backend)
    with pytest.raises(ValueError):
        random_pauli_hamiltonian(
            nqubits=2, normalize=True, max_eigenvalue=1, backend=backend
        )
    with pytest.raises(TypeError):
        random_pauli_hamiltonian(
            nqubits, max_eigenvalue=None, normalize=True, backend=backend
        )

    _, eigenvalues = random_pauli_hamiltonian(
        nqubits, max_eigenvalue, normalize, pauli_order, backend=backend
    )

    if normalize is True:
        backend.assert_allclose(
            np.abs(backend.to_numpy(eigenvalues[0])) < PRECISION_TOL, True
        )
        backend.assert_allclose(
            np.abs(backend.to_numpy(eigenvalues[1]) - 1) < PRECISION_TOL, True
        )
        backend.assert_allclose(
            np.abs(backend.to_numpy(eigenvalues[-1]) - max_eigenvalue) < PRECISION_TOL,
            True,
        )


def test_random_stochastic_matrix(backend):
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_stochastic_matrix(dims, backend=backend)
    with pytest.raises(ValueError):
        dims = 0
        random_stochastic_matrix(dims, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_stochastic_matrix(dims, bistochastic="True", backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_stochastic_matrix(dims, diagonally_dominant="True", backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_stochastic_matrix(dims, precision_tol=1, backend=backend)
    with pytest.raises(ValueError):
        dims, precision_tol = 2, -0.1
        random_stochastic_matrix(dims, precision_tol=precision_tol, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        max_iterations = 1.1
        random_stochastic_matrix(dims, max_iterations=max_iterations, backend=backend)
    with pytest.raises(ValueError):
        dims = 2
        max_iterations = -1
        random_stochastic_matrix(dims, max_iterations=max_iterations, backend=backend)

    # tests if matrix is row-stochastic
    dims = 4
    matrix = random_stochastic_matrix(dims, backend=backend)
    sum_rows = backend.sum(matrix, axis=1)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    # tests if matrix is diagonally dominant
    dims = 4
    matrix = random_stochastic_matrix(
        dims, diagonally_dominant=True, max_iterations=1000, backend=backend
    )

    sum_rows = backend.sum(matrix, axis=1)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(2 * backend.diag(matrix) - sum_rows > 0), True)

    # tests if matrix is bistochastic
    dims = 4
    matrix = random_stochastic_matrix(dims, bistochastic=True, backend=backend)
    sum_rows = backend.sum(matrix, axis=1)
    column_rows = backend.sum(matrix, axis=0)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(column_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(column_rows > 1 - PRECISION_TOL), True)

    # tests if matrix is bistochastic and diagonally dominant
    dims = 4
    matrix = random_stochastic_matrix(
        dims,
        bistochastic=True,
        diagonally_dominant=True,
        max_iterations=1000,
        backend=backend,
    )
    sum_rows = backend.sum(matrix, axis=1)
    column_rows = backend.sum(matrix, axis=0)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(column_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(column_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(2 * backend.diag(matrix) - sum_rows > 0), True)
    backend.assert_allclose(all(2 * backend.diag(matrix) - column_rows > 0), True)

    # tests warning for max_iterations
    dims = 4
    random_stochastic_matrix(dims, bistochastic=True, max_iterations=1, backend=backend)


def test_random_pauli_hamiltonian_tensorflow(backend):
    """Test ``random_pauli_hamiltonian`` with a tensorflow backend to cover
    the ``to_numpy`` conversion."""
    # Create a mock backend with platform="tensorflow"
    mock_backend = MagicMock()
    mock_backend.platform = "tensorflow"
    mock_backend.name = "tensorflow"
    mock_backend.set_seed = MagicMock()

    # eigenvectors returns (eigenvalues, eigenvectors)
    d = 4  # 2 qubits
    mock_eigenvalues = np.array([0.0, 1.0, 2.0, 3.0])
    mock_eigenvectors = np.eye(4)
    mock_backend.eigenvectors.return_value = (mock_eigenvalues, mock_eigenvectors)

    # to_numpy returns the input as-is
    mock_backend.to_numpy.side_effect = lambda x: x

    # random_hermitian needs to work - patch it to return a simple matrix
    mock_hamiltonian = np.eye(d)
    with (
        patch(
            "qibo.quantum_info.random_ensembles.random_hermitian",
            return_value=mock_hamiltonian,
        ),
        patch(
            "qibo.quantum_info.random_ensembles._check_backend",
            return_value=mock_backend,
        ),
    ):
        result = random_pauli_hamiltonian(
            nqubits=2, normalize=False, seed=42, backend=mock_backend
        )
        assert result is not None


@pytest.mark.parametrize(
    "function",
    [
        lambda seed, backend: random_gaussian_matrix(3, seed=seed, backend=backend),
        lambda seed, backend: random_hermitian(3, seed=seed, backend=backend),
        lambda seed, backend: random_unitary(3, seed=seed, backend=backend),
        lambda seed, backend: random_statevector(4, seed=seed, backend=backend),
        lambda seed, backend: random_density_matrix(4, seed=seed, backend=backend),
        lambda seed, backend: random_quantum_channel(4, seed=seed, backend=backend),
        lambda seed, backend: random_pauli(
            2, 2, return_circuit=False, seed=seed, backend=backend
        ),
        lambda seed, backend: random_pauli_hamiltonian(2, seed=seed, backend=backend)[
            0
        ],
    ],
)
def test_random_functions_accept_generator_seed(backend, function):
    first = function(np.random.default_rng(1234), backend)
    second = function(np.random.default_rng(1234), backend)
    other = function(np.random.default_rng(4321), backend)

    backend.assert_allclose(first, second)
    assert not np.allclose(backend.to_numpy(first), backend.to_numpy(other))

    # integer seeds are unchanged
    backend.assert_allclose(function(7, backend), function(7, backend))


@pytest.mark.parametrize("metric", ["ginibre", "bures"])
@pytest.mark.parametrize("rank", [1, 2, 4])
def test_random_density_matrix_rank(backend, metric, rank):
    state = random_density_matrix(4, rank=rank, metric=metric, seed=3, backend=backend)
    eigenvalues = np.linalg.eigvalsh(backend.to_numpy(state))

    assert np.sum(eigenvalues > 1e-10) == rank
    backend.assert_allclose(np.trace(backend.to_numpy(state)), 1.0, atol=1e-10)


def test_random_density_matrix_rank_errors(backend):
    with pytest.raises(ValueError):
        random_density_matrix(4, rank=2, metric="hilbert-schmidt", backend=backend)
    with pytest.raises(ValueError):
        random_density_matrix(4, rank=0, metric="ginibre", backend=backend)

    # the rank equal to the dimension is the default for the Hilbert-Schmidt metric
    random_density_matrix(4, rank=4, backend=backend)


def test_random_density_matrix_pauli_basis_default_order(backend):
    default = random_density_matrix(4, basis="pauli", seed=5, backend=backend)
    explicit = random_density_matrix(4, basis="pauli-IXYZ", seed=5, backend=backend)

    backend.assert_allclose(default, explicit)


def test_random_quantum_channel_invalid_measure(backend):
    with pytest.raises(ValueError):
        random_quantum_channel(4, measure="invalid", backend=backend)


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("max_eigenvalue", [1.2, 1.5, 3.0, 10.0])
def test_random_pauli_hamiltonian_spectrum(backend, nqubits, max_eigenvalue):
    hamiltonian, eigenvalues = random_pauli_hamiltonian(
        nqubits, max_eigenvalue=max_eigenvalue, normalize=True, seed=2, backend=backend
    )
    dims = 2**nqubits

    # back to the computational basis
    vector = backend.to_numpy(
        pauli_to_comp_basis(nqubits, normalize=True, backend=backend)
    ) @ backend.to_numpy(hamiltonian)
    matrix = vector.reshape(dims, dims)
    matrix = (matrix + matrix.conj().T) / 2
    spectrum = np.sort(np.linalg.eigvalsh(matrix))

    # returned eigenvalues are the spectrum of the Hamiltonian, with unit gap
    np.testing.assert_allclose(
        spectrum, np.sort(backend.to_numpy(eigenvalues).real), atol=1e-8
    )
    np.testing.assert_allclose(spectrum[:2], [0.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(spectrum[-1], max_eigenvalue, atol=1e-8)
