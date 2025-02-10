import numpy as np
from numpy import ndarray
from numpy.random import permutation
from scipy.linalg import expm

ENGINE = np


def _pauli_basis(
    nqubits: int,
    dim: int,
    basis_single: ndarray,
) -> ndarray:
    input_indices = [range(3 * i, 3 * (i + 1)) for i in range(nqubits)]
    output_indices = (i for indices in zip(*input_indices) for i in indices)
    operands = [basis_single for _ in range(nqubits)]
    inputs = [item for pair in zip(operands, input_indices) for item in pair]
    return ENGINE.einsum(*inputs, output_indices).reshape(4**nqubits, dim, dim)


def _post_sparse_pauli_basis_vectorization(
    basis: ndarray, dim: int
) -> tuple[ndarray, ndarray]:
    indices = ENGINE.nonzero(basis)
    basis = basis[indices].reshape(-1, dim)
    indices = indices[1].reshape(-1, dim)
    return basis, indices


def _vectorize_sparse_pauli_basis_row(
    basis: ndarray, dim: int
) -> tuple[ndarray, ndarray]:
    basis = _vectorization_row(basis, dim)
    return _post_sparse_pauli_basis_vectorization(basis, dim)


def _vectorize_sparse_pauli_basis_column(
    basis: ndarray, dim: int
) -> tuple[ndarray, ndarray]:
    basis = _vectorization_column(basis, dim)
    return _post_sparse_pauli_basis_vectorization(basis, dim)


def _vectorize_sparse_pauli_basis_system(
    basis: ndarray, dim: int
) -> tuple[ndarray, ndarray]:
    basis = _vectorization_system(basis)
    return _post_sparse_pauli_basis_vectorization(basis, dim)


def _vectorization_row(state: ndarray, dim: int) -> ndarray:
    return ENGINE.reshape(state, (-1, dim**2))


def _vectorization_column(state: ndarray, dim: int) -> ndarray:
    indices = list(range(len(state.shape)))
    indices[-2:] = reversed(indices[-2:])
    state = ENGINE.transpose(state, indices)
    return ENGINE.reshape(state, (-1, dim**2))


def _vectorization_system(state: ndarray) -> ndarray:
    nqubits = int(ENGINE.log2(state.shape[-1]))
    new_axis = [
        0,
    ]
    for qubit in range(nqubits):
        new_axis.extend([qubit + nqubits + 1, qubit + 1])
    state = ENGINE.reshape(state, [-1] + [2] * 2 * nqubits)
    state = ENGINE.transpose(state, new_axis)
    return ENGINE.reshape(state, (-1, 2 ** (2 * nqubits)))


def _random_statevector(dims: int):
    state = ENGINE.random.standard_normal(dims)
    state = state + 1.0j * ENGINE.random.standard_normal(dims)
    return state / ENGINE.linalg.norm(state)


def _random_density_matrix_pure(dims: int):
    state = _random_statevector(dims)
    return ENGINE.outer(state, ENGINE.conj(state).T)


def _random_gaussian_matrix(dims: int, rank: int, mean: float, stddev: float):
    dims = (dims, rank)
    matrix = 1.0j * ENGINE.random.normal(loc=mean, scale=stddev, size=dims)
    matrix += ENGINE.random.normal(loc=mean, scale=stddev, size=dims)
    return matrix


def _random_density_matrix_hs_ginibre(dims: int, rank: int, mean: float, stddev: float):
    state = _random_gaussian_matrix(dims, rank, mean, stddev)
    state = ENGINE.matmul(state, ENGINE.transpose(ENGINE.conj(state), (1, 0)))
    return state / ENGINE.trace(state)


def _random_hermitian(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return (matrix + ENGINE.conj(matrix).T) / 2


def _random_hermitian_semidefinite(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return ENGINE.matmul(ENGINE.conj(matrix).T, matrix)


def _random_unitary(dims: int):
    H = _random_hermitian(dims)
    return expm(-1.0j * H / 2)


def _random_unitary_haar(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    Q, R = ENGINE.linalg.qr(matrix)
    D = ENGINE.diag(R)
    D = D / ENGINE.abs(D)
    R = ENGINE.diag(D)
    return ENGINE.matmul(Q, R)


def _random_density_matrix_bures(dims: int, rank: int, mean: float, stddev: float):
    nqubits = int(ENGINE.log2(dims))
    state = ENGINE.eye(dims)
    state += _random_unitary(dims)
    state = ENGINE.matmul(
        state,
        _random_gaussian_matrix(dims, rank, mean, stddev),
    )
    state = ENGINE.matmul(state, ENGINE.transpose(ENGINE.conj(state), (1, 0)))
    return state / ENGINE.trace(state)


def _sample_from_quantum_mallows_distribution(nqubits: int) -> tuple[ndarray, ndarray]:
    mute_index = list(range(nqubits))
    exponents = ENGINE.arange(nqubits, 0, -1, dtype=ENGINE.int64)
    powers = 4**exponents
    powers[powers == 0] = ENGINE.iinfo(ENGINE.int64).max
    r = ENGINE.random.uniform(0, 1, size=nqubits)
    indexes = -1 * (ENGINE.ceil(ENGINE.log2(r + (1 - r) / powers)))
    hadamards = 1 * (indexes < exponents)
    permutations = ENGINE.zeros(nqubits, dtype=int)
    for l, (index, m) in enumerate(zip(indexes, exponents)):
        k = index if index < m else 2 * m - index - 1
        k = int(k)
        permutations[l] = mute_index[k]
        del mute_index[k]
    return hadamards, permutations


def _delta_gamma_matrices(nqubits: int, hadamards: ndarray, permutations: ndarray):
    delta_matrix = ENGINE.eye(nqubits, dtype=int)
    delta_matrix_prime = ENGINE.copy(delta_matrix)

    gamma_matrix_prime = ENGINE.random.randint(0, 2, size=nqubits)
    gamma_matrix_prime = ENGINE.diag(gamma_matrix_prime)

    gamma_matrix = ENGINE.random.randint(0, 2, size=nqubits)
    gamma_matrix = hadamards * gamma_matrix
    gamma_matrix = ENGINE.diag(gamma_matrix)

    tril_indices = ENGINE.tril_indices(nqubits, k=-1)
    delta_matrix_prime[tril_indices] = ENGINE.random.randint(
        0, 2, size=len(tril_indices[0])
    )
    gamma_matrix_prime[tril_indices] = ENGINE.random.randint(
        0, 2, size=len(tril_indices[0])
    )
    triu_indices = ENGINE.triu_indices(nqubits, k=1)
    gamma_matrix_prime[triu_indices] = gamma_matrix_prime[tril_indices]

    # with these you should be able to reconstruct all the conditional branches below
    # thus lifting the need for the loop, however, due to the very convoluted
    # conditional checks, it may become quite complicate to follow
    # I'd rather try to understand first, if we can simplify the branching below
    p_col_gt_row = permutations[triu_indices[1]] > permutations[triu_indices[0]]
    p_col_le_row = p_col_gt_row ^ True
    h_row_eq_1 = hadamards[triu_indices[0]]
    h_col_eq_1 = hadamards[triu_indices[1]]
    h_row_eq_0 = h_row_eq_1 ^ True
    h_col_eq_0 = h_col_eq_1 ^ True

    # This is quite confusing and convoluted, I have the impression that it may be significantly
    # simplified
    # filling off-diagonal elements of gammas and deltas matrices
    for j in range(nqubits):
        for k in range(j + 1, nqubits):
            if hadamards[k] == 1 and hadamards[j] == 1:  # pragma: no cover
                b = ENGINE.random.randint(0, 2)
                gamma_matrix[k, j] = b
                gamma_matrix[j, k] = b
                if permutations[k] > permutations[j]:
                    b = ENGINE.random.randint(0, 2)
                    delta_matrix[k, j] = b

            if hadamards[k] == 0 and hadamards[j] == 1:
                b = ENGINE.random.randint(0, 2)
                delta_matrix[k, j] = b
                if permutations[k] > permutations[j]:
                    b = ENGINE.random.randint(0, 2)
                    gamma_matrix[k, j] = b
                    gamma_matrix[j, k] = b

            if (
                hadamards[k] == 1
                and hadamards[j] == 0
                and permutations[k] < permutations[j]
            ):  # pragma: no cover
                b = ENGINE.random.randint(0, 2)
                gamma_matrix[k, j] = b
                gamma_matrix[j, k] = b

            if (
                hadamards[k] == 0
                and hadamards[j] == 0
                and permutations[k] < permutations[j]
            ):  # pragma: no cover
                b = ENGINE.random.randint(0, 2)
                delta_matrix[k, j] = b
