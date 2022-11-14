import numpy as np
import pytest

from qibo import get_backend
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


def test_random_ginibre_unitary_matrix(backend):
    with pytest.raises(TypeError):
        dims = np.array([2])
        dims = backend.cast(dims, dtype=dims.dtype)
        random_ginibre_unitary_matrix(dims)
    with pytest.raises(TypeError):
        dims = 2
        rank = np.array([2])
        rank = backend.cast(rank, dtype=rank.dtype)
        random_ginibre_unitary_matrix(dims, rank)
    with pytest.raises(ValueError):
        dims = -1
        random_ginibre_unitary_matrix(dims)
    with pytest.raises(ValueError):
        dims, rank = 2, 4
        random_ginibre_unitary_matrix(dims, rank)
    with pytest.raises(ValueError):
        dims, rank = 2, -1
        random_ginibre_unitary_matrix(dims, rank)

    # just runs the fucntion with no tests
    random_ginibre_unitary_matrix(4)


def test_random_hermitian_operator(backend):
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian_operator(dims, semidefinite="True")
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian_operator(dims, normalize="True")
    with pytest.raises(TypeError):
        dims = np.array([1])
        dims = backend.cast(dims, dtype=dims.dtype)
        random_hermitian_operator(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_hermitian_operator(dims)

    # test if function returns Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    matrix_dagger = backend.cast(matrix_dagger, dtype=matrix_dagger.dtype)
    norm = np.linalg.norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    # test if function returns semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims, semidefinite=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    matrix_dagger = backend.cast(matrix_dagger, dtype=matrix_dagger.dtype)
    norm = np.linalg.norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = backend.cast(eigenvalues, dtype=eigenvalues.dtype)
    backend.assert_allclose(all(eigenvalues >= 0), True)

    # test if function returns normalized Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims, normalize=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    matrix_dagger = backend.cast(matrix_dagger, dtype=matrix_dagger.dtype)
    norm = np.linalg.norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = backend.cast(np.absolute(eigenvalues), dtype=eigenvalues.dtype)
    backend.assert_allclose(all(eigenvalues <= 1), True)

    # test if function returns normalized and semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims, semidefinite=True, normalize=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    matrix_dagger = backend.cast(matrix_dagger, dtype=matrix_dagger.dtype)
    norm = np.linalg.norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = backend.cast(eigenvalues, dtype=eigenvalues.dtype)
    backend.assert_allclose(all(eigenvalues >= 0), True)
    backend.assert_allclose(all(eigenvalues <= 1), True)


def test_random_unitary(backend):
    with pytest.raises(TypeError):
        dims = np.array([1])
        dims = backend.cast(dims, dtype=dims.dtype)
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

    # tests if operator is unitary (measure == "haar")
    dims = 4
    matrix = random_unitary(dims)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix_inv = np.linalg.inv(matrix)
    matrix = backend.cast(matrix, dtype=matrix.dtype)
    matrix_dagger = backend.cast(matrix_dagger, dtype=matrix_dagger.dtype)
    matrix_inv = backend.cast(matrix_inv, dtype=matrix_inv.dtype)
    norm = np.linalg.norm(matrix_inv - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    # tests if operator is unitary (measure == None)
    dims, measure = 4, None
    if get_backend() == "qibojit (cupy)":
        with pytest.raises(NotImplementedError):
            matrix = random_unitary(dims, measure)
    else:
        matrix = random_unitary(dims, measure)
        matrix_dagger = np.transpose(np.conj(matrix))
        matrix_inv = np.linalg.inv(matrix)
        matrix = backend.cast(matrix, dtype=matrix.dtype)
        matrix_dagger = backend.cast(matrix_dagger, dtype=matrix_dagger.dtype)
        matrix_inv = backend.cast(matrix_inv, dtype=matrix_inv.dtype)
        norm = np.linalg.norm(matrix_inv - matrix_dagger)
        backend.assert_allclose(norm < PRECISION_TOL, True)


def test_random_statevector(backend):
    with pytest.raises(TypeError):
        dims = np.array([1])
        dims = backend.cast(dims, dtype=dims.dtype)
        random_statevector(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_statevector(dims)

    dims = 4
    state = random_statevector(dims)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0)


def test_random_density_matrix(backend):
    with pytest.raises(TypeError):
        dims = np.array([1])
        dims = backend.cast(dims, dtype=dims.dtype)
        random_density_matrix(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_density_matrix(dims)
    with pytest.raises(TypeError):
        dims = 2
        rank = np.array([1])
        rank = backend.cast(rank, dtype=rank.dtype)
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
        random_density_matrix(dims, method=1)
    with pytest.raises(ValueError):
        dims = 2
        random_density_matrix(dims, method="gaussian")

    # for pure=True, tests if it is a density matrix and if state is pure
    dims = 4
    state = random_density_matrix(dims, pure=True)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(np.trace(state), 1.0)
    backend.assert_allclose(purity(state), 1.0)

    state_dagger = np.transpose(np.conj(state))
    state_dagger = backend.cast(state, dtype=state.dtype)
    norm = np.linalg.norm(state - state_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    state = random_density_matrix(dims, pure=True, method="Bures")
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(np.trace(state), 1.0)
    backend.assert_allclose(purity(state), 1.0)

    state_dagger = np.transpose(np.conj(state))
    state_dagger = backend.cast(state, dtype=state.dtype)
    norm = np.linalg.norm(state - state_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    # for pure=False, tests if it is a density matrix and if state is mixed
    dims = 4
    state = random_density_matrix(dims)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(np.trace(state), 1.0)
    backend.assert_allclose(purity(state) <= 1.0, True)

    state_dagger = np.transpose(np.conj(state))
    state_dagger = backend.cast(state, dtype=state.dtype)
    norm = np.linalg.norm(state - state_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    dims = 4
    state = random_density_matrix(dims, method="Bures")
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(np.trace(state), 1.0)
    backend.assert_allclose(purity(state) <= 1.0, True)

    state_dagger = np.transpose(np.conj(state))
    state_dagger = backend.cast(state, dtype=state.dtype)
    norm = np.linalg.norm(state - state_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)


def test_stochastic_matrix(backend):
    with pytest.raises(TypeError):
        dims = np.array([1])
        dims = backend.cast(dims, dtype=dims.dtype)
        stochastic_matrix(dims)
    with pytest.raises(ValueError):
        dims = 0
        stochastic_matrix(dims)
    with pytest.raises(TypeError):
        dims = 2
        stochastic_matrix(dims, bistochastic="True")
    with pytest.raises(TypeError):
        dims = 2
        stochastic_matrix(dims, precision_tol=1)
    with pytest.raises(ValueError):
        dims, precision_tol = 2, -0.1
        stochastic_matrix(dims, precision_tol=precision_tol)
    with pytest.raises(TypeError):
        dims = 2
        max_iterations = 1.1
        stochastic_matrix(dims, max_iterations=max_iterations)
    with pytest.raises(ValueError):
        dims = 2
        max_iterations = -1
        stochastic_matrix(dims, max_iterations=max_iterations)

    # tests if matrix is row-stochastic
    dims = 4
    matrix = stochastic_matrix(dims)
    sum_rows = np.sum(matrix, axis=1)
    sum_rows = backend.cast(sum_rows, dtype=sum_rows.dtype)
    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    # tests if matrix is bistochastic
    dims = 4
    matrix = stochastic_matrix(dims, bistochastic=True)
    sum_rows = np.sum(matrix, axis=1)
    column_rows = np.sum(matrix, axis=0)
    sum_rows = backend.cast(sum_rows, dtype=sum_rows.dtype)
    column_rows = backend.cast(column_rows, dtype=sum_rows.dtype)
    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)
    backend.assert_allclose(all(column_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(column_rows > 1 - PRECISION_TOL), True)

    # tests warning for max_iterations
    dims = 4
    stochastic_matrix(dims, bistochastic=True, max_iterations=1)
