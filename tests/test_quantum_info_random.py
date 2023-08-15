"""Tests for the quantum_info.random_ensembles module."""

from functools import reduce

import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info.metrics import purity
from qibo.quantum_info.random_ensembles import (
    random_clifford,
    random_density_matrix,
    random_gaussian_matrix,
    random_hermitian,
    random_pauli,
    random_pauli_hamiltonian,
    random_quantum_channel,
    random_statevector,
    random_stochastic_matrix,
    random_unitary,
)


@pytest.mark.parametrize("seed", [None, 10, np.random.Generator(np.random.MT19937(10))])
def test_random_gaussian_matrix(backend, seed):
    with pytest.raises(TypeError):
        dims = np.array([2])
        random_gaussian_matrix(dims, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        rank = np.array([2])
        random_gaussian_matrix(dims, rank, backend=backend)
    with pytest.raises(ValueError):
        dims = -1
        random_gaussian_matrix(dims, backend=backend)
    with pytest.raises(ValueError):
        dims, rank = 2, 4
        random_gaussian_matrix(dims, rank, backend=backend)
    with pytest.raises(ValueError):
        dims, rank = 2, -1
        random_gaussian_matrix(dims, rank, backend=backend)
    with pytest.raises(ValueError):
        dims, stddev = 2, -1
        random_gaussian_matrix(dims, stddev=stddev, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_gaussian_matrix(dims, seed=0.1, backend=backend)

    # just runs the function with no tests
    random_gaussian_matrix(4, seed=seed, backend=backend)


def test_random_hermitian(backend):
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian(dims, semidefinite="True", backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian(dims, normalize="True", backend=backend)
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_hermitian(dims, backend=backend)
    with pytest.raises(ValueError):
        dims = 0
        random_hermitian(dims, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_hermitian(dims, seed=0.1, backend=backend)

    # test if function returns Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, backend=backend)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = backend.calculate_norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    # test if function returns semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True, backend=backend)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = backend.calculate_norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues >= 0), True)

    # test if function returns normalized Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, normalize=True, backend=backend)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = backend.calculate_norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues <= 1), True)

    # test if function returns normalized and semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True, normalize=True, backend=backend)
    matrix_dagger = np.transpose(np.conj(matrix))
    norm = backend.calculate_norm(matrix - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues >= 0), True)
    backend.assert_allclose(all(eigenvalues <= 1), True)


def test_random_unitary(backend):
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_unitary(dims, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        measure = 1
        random_unitary(dims, measure, backend=backend)
    with pytest.raises(ValueError):
        dims = 0
        random_unitary(dims, backend=backend)
    with pytest.raises(ValueError):
        dims = 2
        random_unitary(dims, measure="gaussian", backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_unitary(dims=2, seed=0.1, backend=backend)

    # tests if operator is unitary (measure == "haar")
    dims = 4
    matrix = random_unitary(dims, backend=backend)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix_inv = np.linalg.inv(matrix)
    norm = backend.calculate_norm(matrix_inv - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)

    # tests if operator is unitary (measure == None)
    dims, measure = 4, None
    matrix = random_unitary(dims, measure, backend=backend)
    matrix_dagger = np.transpose(np.conj(matrix))
    matrix_inv = np.linalg.inv(matrix)
    norm = backend.calculate_norm(matrix_inv - matrix_dagger)
    backend.assert_allclose(norm < PRECISION_TOL, True)


@pytest.mark.parametrize("measure", [None, "haar", "bcsz"])
@pytest.mark.parametrize(
    "representation",
    ["chi", "chi-IZXY", "choi", "kraus", "liouville", "pauli", "pauli-IZXY"],
)
def test_random_quantum_channel(backend, representation, measure):
    with pytest.raises(TypeError):
        test = random_quantum_channel(4, representation=True, backend=backend)
    with pytest.raises(ValueError):
        test = random_quantum_channel(4, representation="Choi", backend=backend)

    # All subroutines are already tested elsewhere,
    # so here we only execute them once for coverage
    random_quantum_channel(4, representation, measure, backend=backend)


@pytest.mark.parametrize("haar", [False, True])
@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_random_statevector(backend, haar, seed):
    with pytest.raises(TypeError):
        dims = "10"
        random_statevector(dims, backend=backend)
    with pytest.raises(ValueError):
        dims = 0
        random_statevector(dims, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_statevector(dims, haar=1, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_statevector(dims, seed=0.1, backend=backend)

    # tests if random statevector is a pure state
    dims = 4
    state = random_statevector(dims, haar=haar, seed=seed, backend=backend)
    backend.assert_allclose(purity(state) <= 1.0 + PRECISION_TOL, True)
    backend.assert_allclose(purity(state) >= 1.0 - PRECISION_TOL, True)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("basis", [None, "pauli-IXYZ", "pauli-IZXY"])
@pytest.mark.parametrize("metric", ["hilbert-schmidt", "ginibre", "bures"])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("dims", [2, 4])
def test_random_density_matrix(backend, dims, pure, metric, basis, normalize):
    with pytest.raises(TypeError):
        test = random_density_matrix(dims=np.array([1]), backend=backend)
    with pytest.raises(ValueError):
        test = random_density_matrix(dims=0, backend=backend)
    with pytest.raises(TypeError):
        test = random_density_matrix(dims=2, rank=np.array([1]), backend=backend)
    with pytest.raises(ValueError):
        test = random_density_matrix(dims=2, rank=3, backend=backend)
    with pytest.raises(ValueError):
        test = random_density_matrix(dims=2, rank=0, backend=backend)
    with pytest.raises(TypeError):
        test = random_density_matrix(dims=2, pure="True", backend=backend)
    with pytest.raises(TypeError):
        test = random_density_matrix(dims=2, metric=1, backend=backend)
    with pytest.raises(ValueError):
        test = random_density_matrix(dims=2, metric="gaussian", backend=backend)
    with pytest.raises(TypeError):
        test = random_density_matrix(dims=2, metric=metric, basis=True)
    with pytest.raises(ValueError):
        test = random_density_matrix(dims=2, metric=metric, basis="Pauli")
    with pytest.raises(TypeError):
        test = random_density_matrix(dims=2, metric=metric, normalize="True")
    with pytest.raises(TypeError):
        random_density_matrix(dims=4, seed=0.1, backend=backend)

    if basis is None and normalize is True:
        with pytest.raises(ValueError):
            test = random_density_matrix(dims=dims, normalize=True)
    else:
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
                np.real(np.trace(state)) <= 1.0 + PRECISION_TOL, True
            )
            backend.assert_allclose(
                np.real(np.trace(state)) >= 1.0 - PRECISION_TOL, True
            )
            backend.assert_allclose(purity(state) <= 1.0 + PRECISION_TOL, True)
            if pure is True:
                backend.assert_allclose(purity(state) >= 1.0 - PRECISION_TOL, True)

            state_dagger = np.transpose(np.conj(state))
            norm = backend.calculate_norm(state - state_dagger)
            backend.assert_allclose(norm < PRECISION_TOL, True)
        else:
            normalization = 1.0 if normalize is False else 1.0 / np.sqrt(dims)
            backend.assert_allclose(
                backend.calculate_norm(state[0] - normalization) <= PRECISION_TOL, True
            )
            assert all(
                backend.calculate_norm(exp_value) <= normalization
                for exp_value in state[1:]
            )


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("return_circuit", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_random_clifford(backend, nqubits, return_circuit, seed):
    with pytest.raises(TypeError):
        test = random_clifford(
            nqubits="1", return_circuit=return_circuit, backend=backend
        )
    with pytest.raises(ValueError):
        test = random_clifford(
            nqubits=-1, return_circuit=return_circuit, backend=backend
        )
    with pytest.raises(TypeError):
        test = random_clifford(nqubits, return_circuit="True", backend=backend)
    with pytest.raises(TypeError):
        test = random_clifford(
            nqubits, return_circuit=return_circuit, seed=0.1, backend=backend
        )

    result_single = matrices.Z @ matrices.H

    result_two = np.kron(matrices.H, matrices.S) @ np.kron(matrices.S, matrices.Z)
    result_two = np.kron(matrices.Z @ matrices.S, matrices.I) @ result_two
    result_two = matrices.CNOT @ matrices.CZ @ result_two

    result = result_single if nqubits == 1 else result_two
    result = backend.cast(result, dtype=result.dtype)

    matrix = random_clifford(
        nqubits, return_circuit=return_circuit, seed=seed, backend=backend
    )

    if return_circuit:
        matrix = matrix.unitary(backend)

    backend.assert_allclose(matrix, result, atol=PRECISION_TOL)


def test_random_pauli_errors(backend):
    with pytest.raises(TypeError):
        q, depth = "1", 1
        random_pauli(q, depth, backend=backend)
    with pytest.raises(ValueError):
        q, depth = -1, 1
        random_pauli(q, depth, backend=backend)
    with pytest.raises(ValueError):
        q = [0, 1, -3]
        depth = 1
        random_pauli(q, depth, backend=backend)
    with pytest.raises(TypeError):
        q, depth = 1, "1"
        random_pauli(q, depth, backend=backend)
    with pytest.raises(ValueError):
        q, depth = 1, 0
        random_pauli(q, depth, backend=backend)
    with pytest.raises(TypeError):
        q, depth, max_qubits = 1, 1, "1"
        random_pauli(q, depth, max_qubits=max_qubits, backend=backend)
    with pytest.raises(ValueError):
        q, depth, max_qubits = 1, 1, 0
        random_pauli(q, depth, max_qubits=max_qubits, backend=backend)
    with pytest.raises(ValueError):
        q, depth, max_qubits = 4, 1, 3
        random_pauli(q, depth, max_qubits=max_qubits, backend=backend)
    with pytest.raises(ValueError):
        q = [0, 1, 3]
        depth = 1
        max_qubits = 2
        random_pauli(q, depth, max_qubits=max_qubits, backend=backend)
    with pytest.raises(TypeError):
        q, depth = 1, 1
        random_pauli(q, depth, return_circuit="True", backend=backend)
    with pytest.raises(TypeError):
        q, depth = 2, 1
        subset = np.array([0, 1])
        random_pauli(q, depth, subset=subset, backend=backend)
    with pytest.raises(TypeError):
        q, depth = 2, 1
        subset = ["I", 0]
        random_pauli(q, depth, subset=subset, backend=backend)
    with pytest.raises(TypeError):
        q, depth = 1, 1
        random_pauli(q, depth, seed=0.1, backend=backend)


def test_pauli_single(backend):
    result = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]])
    result = backend.cast(result, dtype=result.dtype)

    matrix = random_pauli(0, 1, 1, seed=10, backend=backend).unitary(backend=backend)
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
    result_subset = backend.identity_density_matrix(2, normalize=False)

    matrix = random_pauli(
        qubits, depth, max_qubits, subset, return_circuit, seed, backend
    )

    if return_circuit:
        matrix = matrix.unitary(backend=backend)
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

        if subset is None:
            backend.assert_allclose(
                backend.calculate_norm(matrix - result_complete_set) < PRECISION_TOL,
                True,
            )
        else:
            backend.assert_allclose(
                backend.calculate_norm(matrix - result_subset) < PRECISION_TOL, True
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
        backend.assert_allclose(np.abs(eigenvalues[0]) < PRECISION_TOL, True)
        backend.assert_allclose(np.abs(eigenvalues[1] - 1) < PRECISION_TOL, True)
        backend.assert_allclose(
            np.abs(eigenvalues[-1] - max_eigenvalue) < PRECISION_TOL, True
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
    with pytest.raises(TypeError):
        dims = 4
        random_stochastic_matrix(dims, seed=0.1, backend=backend)

    # tests if matrix is row-stochastic
    dims = 4
    matrix = random_stochastic_matrix(dims, backend=backend)
    sum_rows = np.sum(matrix, axis=1)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    # tests if matrix is diagonally dominant
    dims = 4
    matrix = random_stochastic_matrix(
        dims, diagonally_dominant=True, max_iterations=1000, backend=backend
    )
    sum_rows = np.sum(matrix, axis=1)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(2 * np.diag(matrix) - sum_rows > 0), True)

    # tests if matrix is bistochastic
    dims = 4
    matrix = random_stochastic_matrix(dims, bistochastic=True, backend=backend)
    sum_rows = np.sum(matrix, axis=1)
    column_rows = np.sum(matrix, axis=0)

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
    sum_rows = np.sum(matrix, axis=1)
    column_rows = np.sum(matrix, axis=0)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(column_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(column_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(2 * np.diag(matrix) - sum_rows > 0), True)
    backend.assert_allclose(all(2 * np.diag(matrix) - column_rows > 0), True)

    # tests warning for max_iterations
    dims = 4
    random_stochastic_matrix(dims, bistochastic=True, max_iterations=1, backend=backend)
