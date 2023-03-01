import numpy as np
import pytest

from qibo import get_backend
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


@pytest.mark.parametrize("seed", [None, 10, np.random.Generator(np.random.MT19937(10))])
def test_random_gaussian_matrix(seed):
    with pytest.raises(TypeError):
        dims = np.array([2])
        random_gaussian_matrix(dims)
    with pytest.raises(TypeError):
        dims = 2
        rank = np.array([2])
        random_gaussian_matrix(dims, rank)
    with pytest.raises(ValueError):
        dims = -1
        random_gaussian_matrix(dims)
    with pytest.raises(ValueError):
        dims, rank = 2, 4
        random_gaussian_matrix(dims, rank)
    with pytest.raises(ValueError):
        dims, rank = 2, -1
        random_gaussian_matrix(dims, rank)
    with pytest.raises(ValueError):
        dims, stddev = 2, -1
        random_gaussian_matrix(dims, stddev=stddev)
    with pytest.raises(TypeError):
        dims = 2
        random_gaussian_matrix(dims, seed=0.1)

    # just runs the function with no tests
    random_gaussian_matrix(4, seed=seed)


def test_random_hermitian():
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian(dims, semidefinite="True")
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian(dims, normalize="True")
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_hermitian(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_hermitian(dims)
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian(dims, seed=0.1)

    # test if function returns Hermitian operator
    dims = 4
    matrix = random_hermitian(dims)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = np.linalg.norm(matrix - matrix_dagger)
    assert norm < PRECISION_TOL

    # test if function returns semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = np.linalg.norm(matrix - matrix_dagger)
    assert norm < PRECISION_TOL

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    assert all(eigenvalues >= 0)

    # test if function returns normalized Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, normalize=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = np.linalg.norm(matrix - matrix_dagger)
    assert norm < PRECISION_TOL

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    assert all(eigenvalues <= 1)

    # test if function returns normalized and semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True, normalize=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = np.linalg.norm(matrix - matrix_dagger)
    assert norm < PRECISION_TOL

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    assert all(eigenvalues >= 0)
    assert all(eigenvalues <= 1)


def test_random_unitary():
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_unitary(dims)
    with pytest.raises(TypeError):
        dims = 2
        measure = 1
        random_unitary(dims, measure)
    with pytest.raises(ValueError):
        dims = 0
        random_unitary(dims)
    with pytest.raises(ValueError):
        dims = 2
        random_unitary(dims, measure="gaussian")
    with pytest.raises(TypeError):
        dims = 2
        random_unitary(dims=2, seed=0.1)

    # tests if operator is unitary (measure == "haar")
    dims = 4
    matrix = random_unitary(dims)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix_inv = np.linalg.inv(matrix)
    norm = np.linalg.norm(matrix_inv - matrix_dagger)
    assert norm < PRECISION_TOL

    # tests if operator is unitary (measure == None)
    dims, measure = 4, None
    matrix = random_unitary(dims, measure)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix_inv = np.linalg.inv(matrix)
    norm = np.linalg.norm(matrix_inv - matrix_dagger)
    assert norm < PRECISION_TOL


@pytest.mark.parametrize("haar", [False, True])
@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_random_statevector(haar, seed):
    with pytest.raises(TypeError):
        dims = "10"
        random_statevector(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_statevector(dims)
    with pytest.raises(TypeError):
        dims = 2
        random_statevector(dims, haar=1)
    with pytest.raises(TypeError):
        dims = 2
        random_statevector(dims, seed=0.1)

    # tests if random statevector is a pure state
    dims = 4
    state = random_statevector(dims, haar=haar, seed=seed)
    assert purity(state) <= 1.0 + PRECISION_TOL
    assert purity(state) >= 1.0 - PRECISION_TOL


@pytest.mark.parametrize("metric", ["Hilbert-Schmidt", "Bures"])
def test_random_density_matrix(metric):
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_density_matrix(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_density_matrix(dims)
    with pytest.raises(TypeError):
        dims = 2
        rank = np.array([1])
        random_density_matrix(dims, rank)
    with pytest.raises(ValueError):
        dims, rank = 2, 3
        random_density_matrix(dims, rank)
    with pytest.raises(ValueError):
        dims, rank = 2, 0
        random_density_matrix(dims, rank)
    with pytest.raises(TypeError):
        dims = 2
        random_density_matrix(dims, pure="True")
    with pytest.raises(TypeError):
        dims = 2
        random_density_matrix(dims, metric=1)
    with pytest.raises(ValueError):
        dims = 2
        random_density_matrix(dims, metric="gaussian")
    with pytest.raises(TypeError):
        dims = 4
        random_density_matrix(dims, seed=0.1)

    # for pure=True, tests if it is a density matrix and if state is pure
    dims = 4
    state = random_density_matrix(dims, pure=True, metric=metric)
    assert np.real(np.trace(state)) <= 1.0 + PRECISION_TOL
    assert np.real(np.trace(state)) >= 1.0 - PRECISION_TOL
    assert purity(state) <= 1.0 + PRECISION_TOL
    assert purity(state) >= 1.0 - PRECISION_TOL

    state_dagger = np.transpose(np.conj(state))
    norm = np.linalg.norm(state - state_dagger)
    assert norm < PRECISION_TOL

    # for pure=False, tests if it is a density matrix and if state is mixed
    dims = 4
    state = random_density_matrix(dims, metric=metric)
    assert np.real(np.trace(state)) <= 1.0 + PRECISION_TOL
    assert np.real(np.trace(state)) >= 1.0 - PRECISION_TOL
    assert purity(state) <= 1.0 + PRECISION_TOL

    state_dagger = np.transpose(np.conj(state))
    norm = np.linalg.norm(state - state_dagger)
    assert norm < PRECISION_TOL


@pytest.mark.parametrize("qubits", [1, 2, [0, 1], np.array([0, 1])])
@pytest.mark.parametrize("return_circuit", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("seed", [10])
def test_random_clifford(qubits, return_circuit, fuse, seed):
    with pytest.raises(TypeError):
        q = "1"
        random_clifford(q)
    with pytest.raises(ValueError):
        q = -1
        random_clifford(q)
    with pytest.raises(ValueError):
        q = [0, 1, -3]
        random_clifford(q)
    with pytest.raises(TypeError):
        q = 1
        random_clifford(q, return_circuit="True")
    with pytest.raises(TypeError):
        q = 2
        random_clifford(q, fuse="True")
    with pytest.raises(TypeError):
        q = 1
        random_clifford(q, seed=0.1)

    result_single = np.array([[0.5 - 0.5j, -0.5 + 0.5j], [0.5 + 0.5j, 0.5 + 0.5j]])

    result_two = np.array(
        [
            [0.5 + 0.0j, -0.5 - 0.0j, -0.5 + 0.0j, 0.5 + 0.0j],
            [0.0 - 0.5j, 0.0 - 0.5j, 0.0 + 0.5j, 0.0 + 0.5j],
            [0.0 + 0.5j, 0.0 - 0.5j, 0.0 + 0.5j, 0.0 - 0.5j],
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 - 0.0j, 0.5 + 0.0j],
        ]
    )

    result = result_single if (isinstance(qubits, int) and qubits == 1) else result_two

    matrix = random_clifford(
        qubits, return_circuit=return_circuit, fuse=fuse, seed=seed
    )

    if return_circuit:
        assert np.linalg.norm(matrix.matrix - result) < PRECISION_TOL

    if not return_circuit and fuse:
        assert np.linalg.norm(matrix - result) < PRECISION_TOL

    if not return_circuit and not fuse:
        if isinstance(qubits, int) and qubits == 1:
            assert np.linalg.norm(matrix - result) < PRECISION_TOL
        else:
            from functools import reduce

            assert np.linalg.norm(reduce(np.kron, matrix) - result) < PRECISION_TOL


def test_random_pauli_errors():
    with pytest.raises(TypeError):
        q, depth = "1", 1
        random_pauli(q, depth)
    with pytest.raises(ValueError):
        q, depth = -1, 1
        random_pauli(q, depth)
    with pytest.raises(ValueError):
        q = [0, 1, -3]
        depth = 1
        random_pauli(q, depth)
    with pytest.raises(TypeError):
        q, depth = 1, "1"
        random_pauli(q, depth)
    with pytest.raises(ValueError):
        q, depth = 1, 0
        random_pauli(q, depth)
    with pytest.raises(TypeError):
        q, depth, max_qubits = 1, 1, "1"
        random_pauli(q, depth, max_qubits=max_qubits)
    with pytest.raises(ValueError):
        q, depth, max_qubits = 1, 1, 0
        random_pauli(q, depth, max_qubits=max_qubits)
    with pytest.raises(ValueError):
        q, depth, max_qubits = 4, 1, 3
        random_pauli(q, depth, max_qubits=max_qubits)
    with pytest.raises(ValueError):
        q = [0, 1, 3]
        depth = 1
        max_qubits = 2
        random_pauli(q, depth, max_qubits=max_qubits)
    with pytest.raises(TypeError):
        q, depth = 1, 1
        random_pauli(q, depth, return_circuit="True")
    with pytest.raises(TypeError):
        q, depth = 2, 1
        subset = np.array([0, 1])
        random_pauli(q, depth, subset=subset)
    with pytest.raises(TypeError):
        q, depth = 2, 1
        subset = ["I", 0]
        random_pauli(q, depth, subset=subset)
    with pytest.raises(TypeError):
        q, depth = 1, 1
        random_pauli(q, depth, seed=0.1)


def test_pauli_single(backend):
    result = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]])
    result = backend.cast(result, dtype=result.dtype)
    matrix = random_pauli(0, 1, 1, seed=10).unitary()
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    backend.assert_allclose(
        backend.calculate_norm(matrix - result) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("qubits", [2, [0, 1], np.array([0, 1])])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("max_qubits", [None])
@pytest.mark.parametrize("subset", [None, ["I", "X"]])
@pytest.mark.parametrize("return_circuit", [True, False])
@pytest.mark.parametrize("seed", [10])
def test_random_pauli(backend, qubits, depth, max_qubits, subset, return_circuit, seed):
    result_complete_set = np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        ]
    )
    result_complete_set = backend.cast(
        result_complete_set, dtype=result_complete_set.dtype
    )
    result_subset = np.eye(4)
    result_subset = backend.cast(result_subset, dtype=result_subset.dtype)

    matrix = random_pauli(qubits, depth, max_qubits, subset, return_circuit, seed)

    if return_circuit:
        matrix = matrix.unitary()
        matrix = backend.cast(matrix, dtype=matrix.dtype)
        if subset is None:
            backend.assert_allclose(
                backend.calculate_norm(matrix - result_complete_set) < PRECISION_TOL,
                True,
            )
        else:
            backend.assert_allclose(
                backend.calculate_norm(matrix - result_subset) < PRECISION_TOL, True
            )
    else:
        matrix = np.transpose(matrix, (1, 0, 2, 3))
        matrix = [reduce(np.kron, row) for row in matrix]
        matrix = reduce(np.dot, matrix)
        matrix = backend.cast(matrix, dtype=matrix.dtype)
        if subset is None:
            backend.assert_allclose(
                backend.calculate_norm(matrix - result_complete_set) < PRECISION_TOL,
                True,
            )
        else:
            backend.assert_allclose(
                backend.calculate_norm(matrix - result_subset) < PRECISION_TOL, True
            )


def test_random_stochastic_matrix():
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_stochastic_matrix(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_stochastic_matrix(dims)
    with pytest.raises(TypeError):
        dims = 2
        random_stochastic_matrix(dims, bistochastic="True")
    with pytest.raises(TypeError):
        dims = 2
        random_stochastic_matrix(dims, precision_tol=1)
    with pytest.raises(ValueError):
        dims, precision_tol = 2, -0.1
        random_stochastic_matrix(dims, precision_tol=precision_tol)
    with pytest.raises(TypeError):
        dims = 2
        max_iterations = 1.1
        random_stochastic_matrix(dims, max_iterations=max_iterations)
    with pytest.raises(ValueError):
        dims = 2
        max_iterations = -1
        random_stochastic_matrix(dims, max_iterations=max_iterations)
    with pytest.raises(TypeError):
        dims = 4
        random_stochastic_matrix(dims, seed=0.1)

    # tests if matrix is row-stochastic
    dims = 4
    matrix = random_stochastic_matrix(dims)
    sum_rows = np.sum(matrix, axis=1)
    assert all(sum_rows < 1 + PRECISION_TOL)
    assert all(sum_rows > 1 - PRECISION_TOL)

    # tests if matrix is bistochastic
    dims = 4
    matrix = random_stochastic_matrix(dims, bistochastic=True)
    sum_rows = np.sum(matrix, axis=1)
    column_rows = np.sum(matrix, axis=0)
    assert all(sum_rows < 1 + PRECISION_TOL)
    assert all(sum_rows > 1 - PRECISION_TOL)
    assert all(column_rows < 1 + PRECISION_TOL)
    assert all(column_rows > 1 - PRECISION_TOL)

    # tests warning for max_iterations
    dims = 4
    random_stochastic_matrix(dims, bistochastic=True, max_iterations=1)
