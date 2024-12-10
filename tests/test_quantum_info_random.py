"""Tests for the quantum_info.random_ensembles module."""

from functools import reduce

import numpy as np
import pytest

from qibo import Circuit, gates, matrices
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
    uniform_sampling_U3,
)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_uniform_sampling_U3(backend, seed):
    with pytest.raises(TypeError):
        uniform_sampling_U3("1", seed=seed, backend=backend)
    with pytest.raises(ValueError):
        uniform_sampling_U3(0, seed=seed, backend=backend)
    with pytest.raises(TypeError):
        uniform_sampling_U3(2, seed="1", backend=backend)

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
                backend.np.conj(state) @ X @ state,
                backend.np.conj(state) @ Y @ state,
                backend.np.conj(state) @ Z @ state,
            ]
        )
    expectation_values = backend.cast(expectation_values)

    expectation_values = backend.np.mean(expectation_values, axis=0)

    backend.assert_allclose(expectation_values[0], expectation_values[1], atol=1e-1)
    backend.assert_allclose(expectation_values[0], expectation_values[2], atol=1e-1)


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
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
    matrix_dagger = backend.np.conj(matrix).T
    norm = float(backend.calculate_matrix_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    # test if function returns semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True, backend=backend)
    matrix_dagger = backend.np.conj(matrix).T
    norm = float(backend.calculate_matrix_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(backend.to_numpy(matrix))
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues >= 0), True)

    # test if function returns normalized Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, normalize=True, backend=backend)
    matrix_dagger = backend.np.conj(matrix).T
    norm = float(backend.calculate_matrix_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(backend.to_numpy(matrix))
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues <= 1), True)

    # test if function returns normalized and semidefinite Hermitian operator
    dims = 4
    matrix = random_hermitian(dims, semidefinite=True, normalize=True, backend=backend)
    matrix_dagger = backend.np.conj(matrix).T
    norm = float(backend.calculate_vector_norm(matrix - matrix_dagger, order=2))
    backend.assert_allclose(norm < PRECISION_TOL, True)

    eigenvalues = np.linalg.eigvalsh(backend.to_numpy(matrix))
    eigenvalues = np.real(eigenvalues)
    backend.assert_allclose(all(eigenvalues >= 0), True)
    backend.assert_allclose(all(eigenvalues <= 1), True)


@pytest.mark.parametrize("measure", [None, "haar"])
def test_random_unitary(backend, measure):
    with pytest.raises(TypeError):
        dims = np.array([1])
        random_unitary(dims, measure=measure, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_unitary(dims, measure=1, backend=backend)
    with pytest.raises(ValueError):
        dims = 0
        random_unitary(dims, measure=measure, backend=backend)
    with pytest.raises(ValueError):
        dims = 2
        random_unitary(dims, measure="gaussian", backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_unitary(dims=2, measure=measure, seed=0.1, backend=backend)

    # tests if operator is unitary (measure == "haar")
    dims = 4
    matrix = random_unitary(dims, measure=measure, backend=backend)
    matrix_dagger = backend.np.conj(matrix).T
    matrix_inv = (
        backend.np.inverse(matrix)
        if backend.platform == "pytorch"
        else np.linalg.inv(matrix)
    )
    norm = float(backend.calculate_matrix_norm(matrix_inv - matrix_dagger, order=2))
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
        test = random_quantum_channel(4, representation=True, backend=backend)
    with pytest.raises(ValueError):
        test = random_quantum_channel(4, representation="Choi", backend=backend)
    with pytest.raises(NotImplementedError):
        test = random_quantum_channel(4, measure="bcsz", order="system")

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


@pytest.mark.parametrize("seed", [None, 10, np.random.default_rng(10)])
def test_random_statevector(backend, seed):
    with pytest.raises(TypeError):
        dims = "10"
        random_statevector(dims, backend=backend)
    with pytest.raises(ValueError):
        dims = 0
        random_statevector(dims, backend=backend)
    with pytest.raises(TypeError):
        dims = 2
        random_statevector(dims, seed=0.1, backend=backend)

    # tests if random statevector is a pure state
    dims = 4
    state = random_statevector(dims, seed=seed, backend=backend)
    backend.assert_allclose(
        abs(purity(state, backend=backend) - 1.0) < PRECISION_TOL, True
    )


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
        norm_function = (
            backend.calculate_matrix_norm
            if basis is None
            else backend.calculate_vector_norm
        )
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
                backend.to_numpy(
                    norm_function(state - backend.np.conj(state).T, order=2)
                )
            )
            backend.assert_allclose(norm < PRECISION_TOL, True)
        else:
            normalization = 1.0 if normalize is False else 1.0 / np.sqrt(dims)
            backend.assert_allclose(state[0], normalization)
            assert all(
                np.abs(backend.to_numpy(exp_value)) <= normalization
                for exp_value in state[1:]
            )


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("return_circuit", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_random_clifford(backend, nqubits, return_circuit, density_matrix, seed):
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

    result_two = np.kron(matrices.H, matrices.S) @ np.kron(matrices.S, matrices.Y)
    result_two = np.kron(matrices.S @ matrices.X, matrices.I) @ result_two
    result_two = matrices.CNOT @ matrices.CZ @ result_two

    result = result_single if nqubits == 1 else result_two
    result = backend.cast(result, dtype=result.dtype)

    matrix = random_clifford(
        nqubits,
        return_circuit=return_circuit,
        density_matrix=density_matrix,
        seed=seed,
        backend=backend,
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
        np.abs(
            backend.to_numpy(backend.calculate_matrix_norm(matrix - result, order=2))
        )
        < PRECISION_TOL,
        True,
    )


@pytest.mark.parametrize("qubits", [2, [0, 1], np.array([0, 1])])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("max_qubits", [None])
@pytest.mark.parametrize("subset", [None, ["I", "X"]])
@pytest.mark.parametrize("return_circuit", [True, False])
@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("seed", [10])
def test_random_pauli(
    backend, qubits, depth, max_qubits, subset, return_circuit, density_matrix, seed
):
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
        qubits, depth, max_qubits, subset, return_circuit, density_matrix, seed, backend
    )

    if return_circuit:
        matrix = matrix.unitary(backend=backend)
        matrix = backend.cast(matrix, dtype=matrix.dtype)
        if subset is None:
            backend.assert_allclose(
                float(
                    backend.calculate_matrix_norm(matrix - result_complete_set, order=2)
                )
                < PRECISION_TOL,
                True,
            )
        else:
            backend.assert_allclose(
                float(backend.calculate_matrix_norm(matrix - result_subset, order=2))
                < PRECISION_TOL,
                True,
            )
    else:
        matrix = backend.np.transpose(matrix, (1, 0, 2, 3))
        matrix = [reduce(backend.np.kron, row) for row in matrix]
        matrix = reduce(backend.np.matmul, matrix)

        if subset is None:
            backend.assert_allclose(
                float(
                    backend.calculate_matrix_norm(matrix - result_complete_set, order=2)
                )
                < PRECISION_TOL,
                True,
            )
        else:
            backend.assert_allclose(
                float(backend.calculate_matrix_norm(matrix - result_subset, order=2))
                < PRECISION_TOL,
                True,
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
    with pytest.raises(TypeError):
        dims = 4
        random_stochastic_matrix(dims, seed=0.1, backend=backend)

    # tests if matrix is row-stochastic
    dims = 4
    matrix = random_stochastic_matrix(dims, backend=backend)
    sum_rows = backend.np.sum(matrix, axis=1)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    # tests if matrix is diagonally dominant
    dims = 4
    matrix = random_stochastic_matrix(
        dims, diagonally_dominant=True, max_iterations=1000, backend=backend
    )

    sum_rows = backend.np.sum(matrix, axis=1)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(2 * backend.np.diag(matrix) - sum_rows > 0), True)

    # tests if matrix is bistochastic
    dims = 4
    matrix = random_stochastic_matrix(dims, bistochastic=True, backend=backend)
    sum_rows = backend.np.sum(matrix, axis=1)
    column_rows = backend.np.sum(matrix, axis=0)

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
    sum_rows = backend.np.sum(matrix, axis=1)
    column_rows = backend.np.sum(matrix, axis=0)

    backend.assert_allclose(all(sum_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(sum_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(column_rows < 1 + PRECISION_TOL), True)
    backend.assert_allclose(all(column_rows > 1 - PRECISION_TOL), True)

    backend.assert_allclose(all(2 * backend.np.diag(matrix) - sum_rows > 0), True)
    backend.assert_allclose(all(2 * backend.np.diag(matrix) - column_rows > 0), True)

    # tests warning for max_iterations
    dims = 4
    random_stochastic_matrix(dims, bistochastic=True, max_iterations=1, backend=backend)
