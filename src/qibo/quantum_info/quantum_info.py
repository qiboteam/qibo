import numpy as np
from numpy import ndarray


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


def vectorization(state: ndarray, order: str) -> ndarray:
    if (
        (len(state.shape) > 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise TypeError(
            f"Object must have dims either (k,), (k, k), (N, 1, k) or (N, k, k), but have dims {state.shape}."
        )
    if order not in ["row", "column", "system"]:
        raise ValueError(
            f"order must be either 'row' or 'column' or 'system', but it is {order}."
        )

    dim = state.shape[-1]

    if len(state.shape) == 1:
        state = np.outer(state, np.conj(state))
    elif len(state.shape) == 3 and state.shape[1] == 1:
        state = np.einsum("aij,akl->aijkl", state, np.conj(state)).reshape(
            state.shape[0], dim, dim
        )

    if order == "row":
        state = _vectorization_row(state, dim)
    elif order == "column":
        state = _vectorization_column(state, dim)
    else:
        state = _vectorization_system(state)

    return np.squeeze(
        state, axis=tuple(i for i, ax in enumerate(state.shape) if ax == 1)
    )
