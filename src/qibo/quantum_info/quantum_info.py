import numpy as np
from numpy import index_exp, ndarray
from numpy.random import permutation
from scipy.linalg import expm

ENGINE = np


def _pauli_basis(
    nqubits: int,
    pauli_0: ndarray,
    pauli_1: ndarray,
    pauli_2: ndarray,
    pauli_3: ndarray,
    normalization: float = 1.0,
) -> ndarray:
    basis_single = ENGINE.vstack((pauli_0, pauli_1, pauli_2, pauli_3)).reshape(4, 2, 2)
    dim = 2**nqubits
    input_indices = [range(3 * i, 3 * (i + 1)) for i in range(nqubits)]
    output_indices = (i for indices in zip(*input_indices) for i in indices)
    operands = [basis_single for _ in range(nqubits)]
    inputs = [item for pair in zip(operands, input_indices) for item in pair]
    return (
        ENGINE.einsum(  # pylint: disable=too-many-function-args
            *inputs, output_indices
        ).reshape(dim**2, dim, dim)
        / normalization
    )


def _post_sparse_pauli_basis_vectorization(
    basis: ndarray, dim: int
) -> tuple[ndarray, ndarray]:
    indices = ENGINE.nonzero(basis)
    basis = basis[indices].reshape(-1, dim)
    indices = indices[1].reshape(-1, dim)
    return basis, indices


_vectorize_pauli_basis = """
def _vectorize_pauli_basis_{order}(
    nqubits: int, pauli_0: ndarray, pauli_1: ndarray, pauli_2: ndarray, pauli_3: ndarray, normalization: float = 1.
) -> ndarray:
    dim = 2**nqubits
    basis = _pauli_basis(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization)
    return _vectorization_{order}(basis, dim)
"""

_vectorize_sparse_pauli_basis = """
def _vectorize_sparse_pauli_basis_{order}(
    nqubits: int, pauli_0: ndarray, pauli_1: ndarray, pauli_2: ndarray, pauli_3: ndarray, normalization: float = 1.
) -> tuple[ndarray, ndarray]:
    dim = 2**nqubits
    basis = _vectorize_pauli_basis_{order}(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization)
    return _post_sparse_pauli_basis_vectorization(basis, dim)
"""

_pauli_to_comp_basis = """
def _pauli_to_comp_basis_sparse_{order}(
        nqubits: int, pauli_0: ndarray, pauli_1: ndarray, pauli_2: ndarray, pauli_3: ndarray, normalization: float = 1.
) -> ndarray:
    unitary = _vectorize_pauli_basis_{order}(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization).T
    nonzero = ENGINE.nonzero(unitary)
    return unitary[nonzero].reshape(unitary.shape[0], -1), nonzero[1]
"""

_super_op_from_haar_measure = """
def _super_op_from_haar_measure_{order}(dims: int) -> ndarray:
    super_op = _random_unitary_haar(dims)
    super_op = _vectorization_{order}(super_op, dims)
    return ENGINE.outer(super_op, ENGINE.conj(super_op))
"""

_super_op_from_hermitian_measure = """
def _super_op_from_hermitian_measure_{order}(dims: int) -> ndarray:
    super_op = _random_unitary(dims)
    super_op = _vectorization_{order}(super_op, dims)
    return ENGINE.outer(super_op, ENGINE.conj(super_op))
"""

_to_choi = """
def _to_choi_{order}(channel: ndarray) -> ndarray:
    channel = _vectorization_{order}(channel, channel.shape[-1])
    return ENGINE.outer(channel, ENGINE.conj(channel))
"""

_to_liouville = """
def _to_liouville_{order}(channel: ndarray) -> ndarray:
    channel = _to_choi_{order}(channel)
    return _reshuffling(channel, {ax1}, {ax2})
"""

exec(_to_liouville.format(order="row", ax1=1, ax2=2))
exec(_to_liouville.format(order="column", ax1=0, ax2=3))


_to_pauli_liouville = """
def _to_pauli_liouville_{order}(
        channel: ndarray, pauli_0: ndarray, pauli_1: ndarray, pauli_2: ndarray, pauli_3: ndarray, normalization: float = 1.
) -> ndarray:
    nqubits = int(np.log2(channel.shape[0]))
    channel = _to_liouville_{order}(channel)
    unitary = _vectorize_pauli_basis_{order}(
        nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization
    )
    return ENGINE.conj(unitary) @ channel @ unitary.T
"""


for order in ("row", "column", "system"):
    for func in (
        _vectorize_pauli_basis,
        _vectorize_sparse_pauli_basis,
        _pauli_to_comp_basis,
        _super_op_from_haar_measure,
        _super_op_from_hermitian_measure,
        _to_choi,
        _to_pauli_liouville,
    ):
        exec(func.format(order=order))


def _vectorization_row(state: ndarray, dim: int) -> ndarray:
    return ENGINE.reshape(state, (-1, dim**2))


def _vectorization_column(state: ndarray, dim: int) -> ndarray:
    indices = list(range(len(state.shape)))
    indices[-2:] = reversed(indices[-2:])
    state = ENGINE.transpose(state, indices)
    return ENGINE.reshape(state, (-1, dim**2))


# WARNING: dim is not used, but it is useful to uniform
# the call with the other functions
def _vectorization_system(state: ndarray, dim: int = 0) -> ndarray:
    nqubits = int(ENGINE.log2(state.shape[-1]))
    new_axis = [
        0,
    ]
    for qubit in range(nqubits):
        new_axis.extend([qubit + nqubits + 1, qubit + 1])
    state = ENGINE.reshape(state, [-1] + [2] * 2 * nqubits)
    state = ENGINE.transpose(state, new_axis)
    return ENGINE.reshape(state, (-1, 2 ** (2 * nqubits)))


def _unvectorization(state: ndarray, dim: int) -> ndarray:
    nqubits = int(ENGINE.log2(dim))
    axes_old = list(ENGINE.arange(1, 2 * nqubits + 1))
    state = ENGINE.reshape(state, (state.shape[0],) + (2,) * 2 * nqubits)
    state = ENGINE.transpose(
        state,
        [
            0,
        ]
        + axes_old[1::2]
        + axes_old[0::2],
    )
    return ENGINE.reshape(state, (state.shape[0],) + (2**nqubits,) * 2)


def _reshuffling(super_op: ndarray, ax1: int, ax2: int) -> ndarray:
    dim = int(ENGINE.sqrt(super_op.shape[0]))
    super_op = ENGINE.reshape(super_op, (dim,) * 4)
    axes = ENGINE.arange(len(super_op.shape))
    axes[[ax1, ax2]] = axes[[ax2, ax1]]
    super_op = ENGINE.transpose(super_op, axes)
    return ENGINE.reshape(super_op, [dim**2, dim**2])


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
    state = ENGINE.eye(dims, dtype=complex)
    state += _random_unitary(dims)
    state = ENGINE.matmul(
        state,
        _random_gaussian_matrix(dims, rank, mean, stddev),
    )
    state = ENGINE.matmul(state, ENGINE.transpose(ENGINE.conj(state), (1, 0)))
    return state / ENGINE.trace(state)


def _sample_from_quantum_mallows_distribution(nqubits: int) -> tuple[ndarray, ndarray]:
    exponents = ENGINE.arange(nqubits, 0, -1, dtype=ENGINE.int64)
    powers = 4**exponents
    powers[powers == 0] = ENGINE.iinfo(ENGINE.int64).max
    r = ENGINE.random.uniform(0, 1, size=nqubits)
    indexes = (-1) * ENGINE.ceil(ENGINE.log2(r + (1 - r) / powers)).astype(ENGINE.int64)
    idx_le_exp = indexes < exponents
    hadamards = idx_le_exp.astype(ENGINE.int64)
    idx_gt_exp = idx_le_exp ^ True
    indexes[idx_gt_exp] = 2 * exponents[idx_gt_exp] - indexes[idx_gt_exp] - 1
    mute_index = list(range(nqubits))
    permutations = ENGINE.zeros(nqubits, dtype=int)
    for l, index in enumerate(indexes):
        permutations[l] = mute_index[index]
        del mute_index[index]
    return hadamards, permutations


def _gamma_delta_matrices(
    nqubits: int, hadamards: ndarray, permutations: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
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

    return gamma_matrix, gamma_matrix_prime, delta_matrix, delta_matrix_prime


def _super_op_from_bcsz_measure_preamble(
    dims: int, rank: int
) -> tuple[ndarray, ndarray]:
    nqubits = int(np.log2(dims))
    super_op = _random_gaussian_matrix(
        dims**2,
        rank=rank,
        mean=0,
        stddev=1,
    )
    super_op = super_op @ ENGINE.conj(super_op).T
    # partial trace implemented with einsum
    super_op_reduced = ENGINE.einsum("ijik->jk", ENGINE.reshape(super_op, (dims,) * 4))
    eigenvalues, eigenvectors = ENGINE.linalg.eigh(super_op_reduced)
    eigenvalues = ENGINE.sqrt(1.0 / eigenvalues)
    eigenvectors = eigenvectors.T
    operator = ENGINE.einsum("ij,ik->ijk", eigenvectors, ENGINE.conj(eigenvectors))
    operator = ENGINE.sum(
        eigenvalues.reshape(len(eigenvalues), 1, 1) * operator, axis=0
    )
    return operator, super_op


def _super_op_from_bcsz_measure_row(dims: int, rank: int) -> ndarray:
    operator, super_op = _super_op_from_bcsz_measure_preamble(dims, rank)
    operator = ENGINE.kron(ENGINE.eye(dims, dtype=complex), operator)
    return operator @ super_op @ operator


def _super_op_from_bcsz_measure_column(dims: int, rank: int) -> ndarray:
    operator, super_op = _super_op_from_bcsz_measure_preamble(dims, rank)
    operator = ENGINE.kron(operator, ENGINE.eye(dims, dtype=complex))
    return operator @ super_op @ operator


def _to_liouville_row(channel: ndarray) -> ndarray:
    channel = _to_choi_row(channel)  # pylint: disable=undefined-variable
    return _reshuffling(channel, 1, 2)


def _to_liouville_column(channel: ndarray) -> ndarray:
    channel = _to_choi_column(channel)  # pylint: disable=undefined-variable
    return _reshuffling(channel, 0, 3)
