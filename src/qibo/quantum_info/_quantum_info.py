import math
from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

ENGINE = np


def _pauli_basis(
    nqubits: int,
    pauli_0: ArrayLike,
    pauli_1: ArrayLike,
    pauli_2: ArrayLike,
    pauli_3: ArrayLike,
    normalization: Union[float, int] = 1.0,
) -> ArrayLike:
    basis_single = ENGINE.vstack((pauli_0, pauli_1, pauli_2, pauli_3)).reshape(4, 2, 2)
    dim = 2**nqubits
    input_indices = [range(3 * i, 3 * (i + 1)) for i in range(nqubits)]
    output_indices = [i for indices in zip(*input_indices) for i in indices]
    operands = [basis_single for _ in range(nqubits)]
    inputs = [item for pair in zip(operands, input_indices) for item in pair]
    return (
        ENGINE.einsum(  # pylint: disable=too-many-function-args
            *inputs, output_indices
        ).reshape(dim**2, dim, dim)
        / normalization
    )


def _post_sparse_pauli_basis_vectorization(
    basis: ArrayLike, dim: int
) -> Tuple[ArrayLike, ArrayLike]:
    indices = ENGINE.nonzero(basis)
    basis = basis[indices].reshape(-1, dim)
    indices = indices[1].reshape(-1, dim)
    return basis, indices


_vectorize_pauli_basis = """
def _vectorize_pauli_basis_{order}(
    nqubits: int, pauli_0: ArrayLike, pauli_1: ArrayLike, pauli_2: ArrayLike, pauli_3: ArrayLike, normalization: Union[float, int] = 1.0
) -> ArrayLike:
    dim = 2**nqubits
    basis = _pauli_basis(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization)
    return _vectorization_{order}(basis, dim)
"""

_vectorize_sparse_pauli_basis = """
def _vectorize_sparse_pauli_basis_{order}(
    nqubits: int, pauli_0: ArrayLike, pauli_1: ArrayLike, pauli_2: ArrayLike, pauli_3: ArrayLike, normalization: Union[float, int] = 1.0
) -> Tuple[ArrayLike, ArrayLike]:
    dim = 2**nqubits
    basis = _vectorize_pauli_basis_{order}(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization)
    return _post_sparse_pauli_basis_vectorization(basis, dim)
"""

_pauli_to_comp_basis = """
def _pauli_to_comp_basis_sparse_{order}(
        nqubits: int, pauli_0: ArrayLike, pauli_1: ArrayLike, pauli_2: ArrayLike, pauli_3: ArrayLike, normalization: Union[float, int] = 1.0
) -> ArrayLike:
    unitary = _vectorize_pauli_basis_{order}(nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization).T
    nonzero = ENGINE.nonzero(unitary)
    return unitary[nonzero].reshape(unitary.shape[0], -1), nonzero[1]
"""

_super_op_from_haar_measure = """
def _super_op_from_haar_measure_{order}(dims: int) -> ArrayLike:
    super_op = _random_unitary_haar(dims)
    super_op = _vectorization_{order}(super_op, dims).ravel()
    return ENGINE.outer(super_op, ENGINE.conj(super_op))
"""

_super_op_from_hermitian_measure = """
def _super_op_from_hermitian_measure_{order}(dims: int) -> ArrayLike:
    super_op = _random_unitary(dims)
    super_op = _vectorization_{order}(super_op, dims).ravel()
    return ENGINE.outer(super_op, ENGINE.conj(super_op))
"""

_to_choi = """
def _to_choi_{order}(channel: ArrayLike) -> ArrayLike:
    channel = _vectorization_{order}(channel, channel.shape[-1]).ravel()
    return ENGINE.outer(channel, ENGINE.conj(channel))
"""

_to_liouville = """
def _to_liouville_{order}(channel: ArrayLike) -> ArrayLike:
    channel = _to_choi_{order}(channel)
    return _reshuffling(channel, {ax1}, {ax2})
"""

exec(_to_liouville.format(order="row", ax1=1, ax2=2))
exec(_to_liouville.format(order="column", ax1=0, ax2=3))


_to_pauli_liouville = """
def _to_pauli_liouville_{order}(
        channel: ArrayLike, pauli_0: ArrayLike, pauli_1: ArrayLike, pauli_2: ArrayLike, pauli_3: ArrayLike, normalization: Union[float, int] = 1.0
) -> ArrayLike:
    nqubits = int(np.log2(channel.shape[0]))
    channel = _to_liouville_{order}(channel)
    unitary = _vectorize_pauli_basis_{order}(
        nqubits, pauli_0, pauli_1, pauli_2, pauli_3, normalization
    )
    return ENGINE.conj(unitary) @ channel @ unitary.T
"""

_choi_to_kraus = """
def _choi_to_kraus_{order}(choi_super_op: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    U, coefficients, V = ENGINE.singular_value_decomposition(choi_super_op)
    U = U.T
    coefficients = ENGINE.sqrt(coefficients)
    V = ENGINE.conj(V)
    dim = int(np.sqrt(U.shape[-1]))
    coefficients = coefficients.reshape(U.shape[0], 1, 1)
    kraus_left = coefficients * _unvectorization_{order}(U, dim)
    kraus_right = coefficients * _unvectorization_{order}(V, dim)
    kraus_ops = ENGINE.vstack((kraus_left[None,:], kraus_right[None,:]))
    return kraus_ops, coefficients
"""

_choi_to_kraus_cp = """
def _choi_to_kraus_cp_{order}(
    eigenvalues: ArrayLike, eigenvectors: ArrayLike, precision: float
) -> Tuple[ArrayLike, ArrayLike]:
    eigv_gt_tol = ENGINE.abs(eigenvalues) > precision
    coefficients = ENGINE.sqrt(eigenvalues[eigv_gt_tol])
    eigenvectors = eigenvectors[eigv_gt_tol]
    dim = int(np.sqrt(eigenvectors.shape[-1]))
    kraus_ops = coefficients.reshape(-1, 1, 1) * _unvectorization_{order}(
        eigenvectors, dim
    )
    return kraus_ops, coefficients
"""

_kraus_to_choi = """
def _kraus_to_choi_{order}(kraus_ops: ArrayLike) -> ArrayLike:
    kraus_ops = _vectorization_{order}(kraus_ops, kraus_ops.shape[-1])
    return kraus_ops.T @ ENGINE.conj(kraus_ops)
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
        _choi_to_kraus,
        _choi_to_kraus_cp,
        _kraus_to_choi,
    ):
        exec(func.format(order=order))


def _vectorization_row(state: ArrayLike, dim: int) -> ArrayLike:
    return ENGINE.reshape(state, (-1, dim**2))


def _vectorization_column(state: ArrayLike, dim: int) -> ArrayLike:
    indices = list(range(state.ndim))
    indices[-2:] = indices[-2:][::-1]
    state = ENGINE.transpose(state, indices)
    return ENGINE.reshape(state, (-1, dim**2))


# WARNING: dim is not used, but it is useful to uniform
# the call with the other functions
def _vectorization_system(state: ArrayLike, dim: int = 0) -> ArrayLike:
    nqubits = int(math.log2(state.shape[-1]))
    new_axis = [
        0,
    ]
    for qubit in range(nqubits):
        new_axis.extend([qubit + nqubits + 1, qubit + 1])
    state = ENGINE.reshape(state, [-1] + [2] * 2 * nqubits)
    state = ENGINE.transpose(state, new_axis)
    return ENGINE.reshape(state, (-1, 2 ** (2 * nqubits)))


def _unvectorization_row(state: ArrayLike, dim: int) -> ArrayLike:
    return ENGINE.reshape(state, (state.shape[0], dim, dim))


def _unvectorization_column(state: ArrayLike, dim: int) -> ArrayLike:
    # this should be equivalent and more broadly supported (e.g. by numba)
    # return ENGINE.reshape(state.T, (dim, dim, state.shape[0])).T
    return ENGINE.reshape(state, (state.shape[0], dim, dim), order="F")


def _unvectorization_system(state: ArrayLike, dim: int) -> ArrayLike:
    nqubits = int(math.log2(dim))
    axes_old = list(range(1, 2 * nqubits + 1))
    state = ENGINE.reshape(state, (state.shape[0],) + (2,) * 2 * nqubits)
    state = ENGINE.transpose(state, [0] + axes_old[1::2] + axes_old[0::2])
    return ENGINE.reshape(state, (state.shape[0],) + (2**nqubits,) * 2)


def _reshuffling(super_op: ArrayLike, ax1: int, ax2: int) -> ArrayLike:
    dim = int(math.sqrt(super_op.shape[0]))
    super_op = ENGINE.reshape(super_op, (dim,) * 4)
    axes = list(range(len(super_op.shape)))
    tmp = axes[ax1]
    axes[ax1] = axes[ax2]
    axes[ax2] = tmp
    super_op = ENGINE.transpose(super_op, axes)
    return ENGINE.reshape(super_op, [dim**2, dim**2])


def _random_statevector_real(dims: int):
    state = ENGINE.random_normal(0, 1, dims)
    return state / ENGINE.vector_norm(state)


def _random_statevector(dims: int):
    state = ENGINE.random_normal(0, 1, dims)
    state = state + 1.0j * ENGINE.random_normal(0, 1, size=dims)
    return state / ENGINE.vector_norm(state)


def _random_density_matrix_pure(dims: int):
    state = _random_statevector(dims)
    return ENGINE.outer(state, ENGINE.conj(state).T)


def _random_gaussian_matrix(dims: int, rank: int, mean: float, stddev: float):
    dims = (dims, rank)
    matrix = 1.0j * ENGINE.random_normal(mean, stddev, size=dims)
    matrix += ENGINE.random_normal(mean, stddev, size=dims)
    return matrix


def _random_density_matrix_hs_ginibre(dims: int, rank: int, mean: float, stddev: float):
    state = _random_gaussian_matrix(dims, rank, mean, stddev)
    state = state @ ENGINE.conj(state).T
    return state / ENGINE.trace(state)


def _random_hermitian(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return (matrix + ENGINE.conj(matrix).T) / 2


def _random_hermitian_semidefinite(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    return ENGINE.conj(matrix).T @ matrix


def _random_unitary(dims: int):
    H = _random_hermitian(dims)
    return ENGINE.expm(-1.0j * H / 2)


def _random_unitary_haar(dims: int):
    matrix = _random_gaussian_matrix(dims, dims, 0.0, 1.0)
    Q, R = ENGINE.qr(matrix)
    D = ENGINE.diag(R)
    D = D / ENGINE.abs(D)
    R = ENGINE.diag(D)
    return Q @ R


def _random_density_matrix_bures(dims: int, rank: int, mean: float, stddev: float):
    state = ENGINE.identity(dims, dtype=ENGINE.complex128)
    state += _random_unitary(dims)
    state = state @ _random_gaussian_matrix(dims, rank, mean, stddev)
    state = state @ ENGINE.conj(state).T
    return state / ENGINE.trace(state)


def _fill_tril(mat, symmetric):
    """Add symmetric random ints to off-diagonals"""
    dim = mat.shape[0]
    # Optimized for low dimensions
    if dim == 1:
        return

    if dim <= 4:
        mat[1, 0] = ENGINE.random_integers(2, dtype=ENGINE.uint8)
        if symmetric:
            mat[0, 1] = mat[1, 0]
        if dim > 2:
            mat[2, 0] = ENGINE.random_integers(2, dtype=np.uint8)
            mat[2, 1] = ENGINE.random_integers(2, dtype=np.uint8)
            if symmetric:
                mat[0, 2] = mat[2, 0]
                mat[1, 2] = mat[2, 1]
        if dim > 3:
            mat[3, 0] = ENGINE.random_integers(2, dtype=np.uint8)
            mat[3, 1] = ENGINE.random_integers(2, dtype=np.uint8)
            mat[3, 2] = ENGINE.random_integers(2, dtype=np.uint8)
            if symmetric:
                mat[0, 3] = mat[3, 0]
                mat[1, 3] = mat[3, 1]
                mat[2, 3] = mat[3, 2]
        return

    # Use numpy indices for larger dimensions
    rows, cols = ENGINE.tril_indices(dim, -1)
    vals = ENGINE.random_integers(2, size=rows.size, dtype=ENGINE.uint8)
    mat[(rows, cols)] = vals
    if symmetric:
        mat[(cols, rows)] = vals


def _inverse_tril(mat, block_inverse_threshold):
    """Invert a lower-triangular matrix with unit diagonal."""
    # Optimized inversion function for low dimensions
    dim = mat.shape[0]

    if dim <= 2:
        return mat

    if dim <= 5:
        inv = ENGINE.cast(mat, dtype=mat.dtype, copy=True)
        inv[2, 0] = mat[2, 0] ^ (mat[1, 0] & mat[2, 1])
        if dim > 3:
            inv[3, 1] = mat[3, 1] ^ (mat[2, 1] & mat[3, 2])
            inv[3, 0] = mat[3, 0] ^ (mat[3, 2] & mat[2, 0]) ^ (mat[1, 0] & inv[3, 1])
        if dim > 4:
            inv[4, 2] = (mat[4, 2] ^ (mat[3, 2] & mat[4, 3])) & 1
            inv[4, 1] = mat[4, 1] ^ (mat[4, 3] & mat[3, 1]) ^ (mat[2, 1] & inv[4, 2])
            inv[4, 0] = (
                mat[4, 0]
                ^ (mat[1, 0] & inv[4, 1])
                ^ (mat[2, 0] & inv[4, 2])
                ^ (mat[3, 0] & mat[4, 3])
            )
        return inv % 2

    # For higher dimensions we use Numpy's inverse function
    # however this function tends to fail and result in a non-symplectic
    # final matrix if n is too large.
    if dim <= block_inverse_threshold:
        return ENGINE.inv(mat) % 2

    # For very large matrices  we divide the matrix into 4 blocks of
    # roughly equal size and use the analytic formula for the inverse
    # of a block lower-triangular matrix:
    # inv([[A, 0],[C, D]]) = [[inv(A), 0], [inv(D).C.inv(A), inv(D)]]
    # call the inverse function recursively to compute inv(A) and invD

    dim1 = dim // 2
    mat_a = _inverse_tril(mat[0:dim1, 0:dim1], block_inverse_threshold)
    mat_d = _inverse_tril(mat[dim1:dim, dim1:dim], block_inverse_threshold)
    mat_c = (mat_d @ mat[dim1:dim, 0:dim1]) @ mat_a
    inv = ENGINE.vstack(
        [
            ENGINE.hstack([mat_a, ENGINE.zeros((dim1, dim - dim1), dtype=int)]),
            ENGINE.hstack([mat_c, mat_d]),
        ]
    )
    return inv % 2


def _sample_from_quantum_mallows_distribution(
    nqubits: int,
) -> Tuple[ArrayLike, ArrayLike]:
    """Using the quantum Mallows distribution, samples a binary array
    representing a layer of Hadamard gates as well as an array with permutated
    qubit indexes. For more details, see Reference [1].

    Args:
        nqubits (int): number of qubits.

    Returns:
        (:class:`numpy.typing.ArrayLike`, :class:`numpy.typing.ArrayLike`): tuple of binary
        :class:`numpy.typing.ArrayLike` and :class:`numpy.typing.ArrayLike` of indexes.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
            structure of the Clifford group*.
            `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_.
    """
    exponents = ENGINE.arange(nqubits, 0, -1, dtype=ENGINE.int64)
    powers = 4**exponents
    powers[powers == 0] = int(np.iinfo(ENGINE.int64).max)
    r = ENGINE.random_uniform(0, 1, size=nqubits)
    indexes = (-1) * ENGINE.ceil(ENGINE.log2(r + (1 - r) / powers))
    indexes = ENGINE.cast(indexes, dtype=ENGINE.int64)
    idx_le_exp = indexes < exponents
    hadamards = ENGINE.cast(idx_le_exp, dtype=ENGINE.int64)
    idx_gt_exp = idx_le_exp ^ True
    indexes[idx_gt_exp] = 2 * exponents[idx_gt_exp] - indexes[idx_gt_exp] - 1
    permutations = ENGINE.empty(nqubits, dtype=ENGINE.int64)
    mask = ENGINE.ones(nqubits, dtype=bool)
    for l, index in enumerate(indexes):
        available = ENGINE.flatnonzero(mask)
        permutations[l] = available[index]
        mask[permutations[l]] = False
    return hadamards, permutations


def _super_op_from_bcsz_measure_preamble(
    dims: int, rank: int
) -> Tuple[ArrayLike, ArrayLike]:
    super_op = _random_gaussian_matrix(
        dims**2,
        rank=rank,
        mean=0.0,
        stddev=1.0,
    )
    super_op = super_op @ ENGINE.conj(super_op).T
    # partial trace implemented with einsum
    super_op_reduced = ENGINE.einsum("ijik->jk", ENGINE.reshape(super_op, (dims,) * 4))
    eigenvalues, eigenvectors = ENGINE.eigh(super_op_reduced)
    eigenvalues = ENGINE.sqrt(1.0 / eigenvalues)
    eigenvectors = eigenvectors.T
    operator = ENGINE.einsum("ij,ik->ijk", eigenvectors, ENGINE.conj(eigenvectors))
    operator = ENGINE.sum(
        eigenvalues.reshape(len(eigenvalues), 1, 1) * operator, axis=0
    )
    return operator, super_op


def _super_op_from_bcsz_measure_row(dims: int, rank: int) -> ArrayLike:
    operator, super_op = _super_op_from_bcsz_measure_preamble(dims, rank)
    operator = ENGINE.kron(ENGINE.identity(dims, dtype=operator.dtype), operator)
    return operator @ super_op @ operator


def _super_op_from_bcsz_measure_column(dims: int, rank: int) -> ArrayLike:
    operator, super_op = _super_op_from_bcsz_measure_preamble(dims, rank)
    operator = ENGINE.kron(operator, ENGINE.identity(dims, dtype=operator.dtype))
    return operator @ super_op @ operator


def _kraus_to_stinespring(
    kraus_ops: ArrayLike, initial_state_env: ArrayLike, dim_env: int
) -> ArrayLike:
    alphas = ENGINE.zeros((dim_env, dim_env, dim_env), dtype=kraus_ops.dtype)
    alphas[range(dim_env), range(dim_env)] = initial_state_env
    # batched kron product
    return ENGINE.einsum("aij,akl->ikjl", kraus_ops, alphas).reshape(
        2 * (kraus_ops.shape[1] * alphas.shape[1],)
    )


def _stinespring_to_kraus(
    stinespring: ArrayLike, initial_state_env: ArrayLike, dim: int, dim_env: int
) -> ArrayLike:
    stinespring = ENGINE.reshape(stinespring, (dim, dim_env, dim, dim_env))
    stinespring = ENGINE.swapaxes(stinespring, 1, 2)
    alphas = ENGINE.identity(dim_env, dtype=stinespring.dtype)
    stinespring = (alphas @ stinespring).reshape(dim, dim_env, dim + dim_env)
    stinespring = ENGINE.vstack(
        (stinespring[:, :, :dim_env], stinespring[:, :, dim_env:])
    )
    return (stinespring @ initial_state_env).reshape(dim, dim_env, dim_env)
