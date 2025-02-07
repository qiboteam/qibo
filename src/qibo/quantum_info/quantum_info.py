import numpy as np
from numpy import ndarray
from scipy.linalg import expm


def _pauli_basis(
    nqubits: int,
    dim: int,
    basis_single: ndarray,
) -> ndarray:
    """Creates the ``nqubits``-qubit Pauli basis.

    Args:
        nqubits (int): number of qubits.
        basis_single (ndarray): the one qubit pauli basis.

    Returns:
        ndarray: the full nqubits basis.
    """
    input_indices = [range(3 * i, 3 * (i + 1)) for i in range(nqubits)]
    output_indices = (i for indices in zip(*input_indices) for i in indices)
    operands = [basis_single for _ in range(nqubits)]
    inputs = [item for pair in zip(operands, input_indices) for item in pair]
    return np.einsum(*inputs, output_indices).reshape(4**nqubits, dim, dim)


def _vectorize_sparse_pauli_basis(
    basis: ndarray, dim: int, order: str
) -> tuple[ndarray, ndarray]:
    basis = vectorization(basis, order=order)
    indices = np.nonzero(basis)
    basis = basis[indices].reshape(-1, dim)
    indices = indices[1].reshape(-1, dim)
    return basis, indices


def _vectorization_row(state: ndarray, dim: int) -> ndarray:
    return np.reshape(state, (-1, dim**2))


def _vectorization_column(state: ndarray, dim: int) -> ndarray:
    indices = list(range(len(state.shape)))
    indices[-2:] = reversed(indices[-2:])
    state = np.transpose(state, indices)
    return np.reshape(state, (-1, dim**2))


def _vectorization_system(state: ndarray) -> ndarray:
    nqubits = int(np.log2(state.shape[-1]))
    new_axis = [
        0,
    ]
    for qubit in range(nqubits):
        new_axis.extend([qubit + nqubits + 1, qubit + 1])
    state = np.reshape(state, [-1] + [2] * 2 * nqubits)
    state = np.transpose(state, new_axis)
    return np.reshape(state, (-1, 2 ** (2 * nqubits)))


def _random_statevector(dims):
    state = np.random.standard_normal(dims)
    state = state + 1.0j * np.random.standard_normal(dims)
    return state / np.linalg.norm(state)


def _random_density_matrix_pure(dims):
    state = _random_statevector(dims)
    return np.outer(state, np.conj(state).T)


def _random_gaussian_matrix(dims, rank, mean, stddev):
    dims = (dims, rank)
    matrix = 1.0j * np.random.normal(loc=mean, scale=stddev, size=dims)
    matrix += np.random.normal(loc=mean, scale=stddev, size=dims)
    return matrix


def _random_density_matrix_hs_ginibre(dims, rank, mean, stddev):
    state = _random_gaussian_matrix(dims, rank, mean, stddev)
    state = np.matmul(state, np.transpose(np.conj(state), (1, 0)))
    return state / np.trace(state)


def _random_hermitian(dims, rank, mean, stddev):
    matrix = _random_gaussian_matrix(dims, rank, mean, stddev)
    return (matrix + np.conj(matrix).T) / 2


def _random_unitary(dims, rank, mean, stddev):
    H = _random_hermitian(dims, rank, mean, stddev)
    return expm(-1.0j * H / 2)


def _random_density_matrix_bures(dims, rank, mean, stddev):
    nqubits = int(np.log2(dims))
    state = np.eye(dims)
    state += _random_unitary(dims, rank, mean, stddev)
    state = np.matmul(
        state,
        _random_gaussian_matrix(dims, rank, mean, stddev),
    )
    state = np.matmul(state, np.transpose(np.conj(state), (1, 0)))
    return state / np.trace(state)


def _sample_from_quantum_mallows_distribution(nqubits: int):
    mute_index = list(range(nqubits))
    exponents = np.arange(nqubits, 0, -1, dtype=np.int64)
    powers = 4**exponents
    powers[powers == 0] = np.iinfo(np.int64).max
    r = np.random.uniform(0, 1, size=nqubits)
    indexes = -1 * (np.ceil(np.log2(r + (1 - r) / powers)))
    hadamards = 1 * (indexes < exponents)
    permutations = np.zeros(nqubits, dtype=int)
    for l, (index, m) in enumerate(zip(indexes, exponents)):
        k = index if index < m else 2 * m - index - 1
        k = int(k)
        permutations[l] = mute_index[k]
        del mute_index[k]
    return hadamards, permutations


def _delta_gamma_matrices():
    pass
