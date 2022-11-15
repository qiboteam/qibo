import numpy as np
import pytest

from qibo import get_backend
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


def test_random_gaussian_matrix():
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
    with pytest.raises(TypeError):
        dims = 2
        random_gaussian_matrix(dims, seed=0.1)

    # just runs the fucntion with no tests
    random_gaussian_matrix(4)


def test_random_hermitian_operator():
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian_operator(dims, semidefinite="True")
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian_operator(dims, normalize="True")
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_hermitian_operator(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_hermitian_operator(dims)
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian_operator(dims, seed=0.1)

    # test if function returns Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = np.linalg.norm(matrix - matrix_dagger)
    assert norm < PRECISION_TOL

    # test if function returns semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims, semidefinite=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = np.linalg.norm(matrix - matrix_dagger)
    assert norm < PRECISION_TOL

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    assert all(eigenvalues >= 0)

    # test if function returns normalized Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims, normalize=True)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = np.linalg.norm(matrix - matrix_dagger)
    assert norm < PRECISION_TOL

    eigenvalues, _ = np.linalg.eigh(matrix)
    eigenvalues = np.real(eigenvalues)
    assert all(eigenvalues <= 1)

    # test if function returns normalized and semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian_operator(dims, semidefinite=True, normalize=True)
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


def test_random_statevector():
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_statevector(dims)
    with pytest.raises(ValueError):
        dims = 0
        random_statevector(dims)
    with pytest.raises(TypeError):
        dims, haar = 2, 1
        random_statevector(dims, haar)
    with pytest.raises(TypeError):
        dims = 2
        random_statevector(dims, seed=0.1)

    # tests if random statevector is a pure state
    dims = 4
    state = random_statevector(dims)
    assert purity(state) <= 1.0 + PRECISION_TOL
    assert purity(state) >= 1.0 - PRECISION_TOL

    # tests if haar random statevector is a pure state
    dims = 4
    state = random_statevector(dims, haar=True)
    assert purity(state) <= 1.0 + PRECISION_TOL
    assert purity(state) >= 1.0 - PRECISION_TOL


@pytest.mark.parametrize("method", ["Hilbert-Schmidt", "Bures"])
def test_random_density_matrix(method):
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
        random_density_matrix(dims, method=1)
    with pytest.raises(ValueError):
        dims = 2
        random_density_matrix(dims, method="gaussian")
    with pytest.raises(TypeError):
        dims = 4
        random_density_matrix(dims, seed=0.1)

    # for pure=True, tests if it is a density matrix and if state is pure
    dims = 4
    state = random_density_matrix(dims, pure=True, method=method)
    assert np.real(np.trace(state)) <= 1.0 + PRECISION_TOL
    assert np.real(np.trace(state)) >= 1.0 - PRECISION_TOL
    assert purity(state) <= 1.0 + PRECISION_TOL
    assert purity(state) >= 1.0 - PRECISION_TOL

    state_dagger = np.transpose(np.conj(state))
    norm = np.linalg.norm(state - state_dagger)
    assert norm < PRECISION_TOL

    # for pure=False, tests if it is a density matrix and if state is mixed
    dims = 4
    state = random_density_matrix(dims, method=method)
    assert np.real(np.trace(state)) <= 1.0 + PRECISION_TOL
    assert np.real(np.trace(state)) >= 1.0 - PRECISION_TOL
    assert purity(state) <= 1.0 + PRECISION_TOL

    state_dagger = np.transpose(np.conj(state))
    norm = np.linalg.norm(state - state_dagger)
    assert norm < PRECISION_TOL


@pytest.mark.parametrize("qubits", [2, [0, 1], np.array([0, 1])])
@pytest.mark.parametrize("return_circuit", [False, True])
@pytest.mark.parametrize("seed", [10])
def test_random_clifford(qubits, return_circuit, seed):
    with pytest.raises(TypeError):
        q = "1"
        random_clifford_gate(q)
    with pytest.raises(ValueError):
        q = -1
        random_clifford_gate(q)
    with pytest.raises(ValueError):
        q = [0, 1, -3]
        random_clifford_gate(q)
    with pytest.raises(TypeError):
        q = 1
        random_clifford_gate(q, return_circuit="True")
    with pytest.raises(TypeError):
        q = 1
        random_clifford_gate(q, seed=0.1)

    result = np.array(
        [
            [0.5 + 0.5j, 0.5 - 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.5 - 0.5j, 0.5 + 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.5 - 0.5j, -0.5 - 0.5j],
            [0.0 + 0.0j, 0.0 + 0.0j, -0.5 - 0.5j, 0.5 - 0.5j],
        ]
    )

    matrix = random_clifford_gate(qubits, return_circuit=return_circuit, seed=seed)

    if return_circuit == True:
        assert np.linalg.norm(matrix.matrix - result) < PRECISION_TOL
    else:
        from functools import reduce

        assert np.linalg.norm(reduce(np.kron, matrix) - result) < PRECISION_TOL


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
