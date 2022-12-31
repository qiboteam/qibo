from functools import reduce
from itertools import product

import numpy as np

from qibo import matrices
from qibo.config import raise_error


def vectorization(state):
    """Returns state :math:`\\rho` in its Liouville
    representation :math:`\\ket{\\rho}`.

    Args:
        state: state vector or density matrix.

    Returns:
        Liouville representation of ``state``.
    """

    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"Object must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    def block_split(matrix, nrows: int, ncols: int):
        """Block-vectorization of a square :math:`N \times N`
        matrix into 4 :math:`\frac{N}{2} \times \frac{N}{2}`
        matrices, where :math:`N = 2^{n}` and :math:`n` is the
        number of qubits.

        Args:
            matrix: :math:`N \times N` matrix.
            nrows (int): number of rows of the block matrix.
            ncols (int): number of columns of the block matrix

        Returns:
            Block-vectorization of ``matrix``.
        """
        dim, _ = matrix.shape
        return (
            matrix.reshape(int(dim / nrows), nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols)[[0, 2, 1, 3]]
        )

    d = len(state)
    n = int(d / 2)
    nqubits = int(np.log2(d))

    if len(state.shape) == 1:
        state = np.outer(state, state.conj())

    if n == 1:
        state = state.reshape((1, -1), order="F")[0]
    else:
        state = block_split(state, n, n)
        for _ in range(nqubits - 2, 0, -1):
            n = int(n / 2)
            state = np.array([block_split(matrix, n, n) for matrix in state])
            state = state.reshape((np.prod(state.shape[:-2]), *(state.shape[-2:])))
        state = np.array(
            [matrix.reshape((1, -1), order="F") for matrix in state]
        ).flatten()

    return state


def pauli_basis(nqubits: int, normalize: bool = False, vectorize: bool = False):
    """Creates the ``nqubits``-qubit Pauli basis.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, normalized basis is returned.
            Defaults to False.
        vectorize (bool, optional): If ``False``, returns a nested array with
            all Pauli matrices. If ``True``, retuns an array where every
            row is a vectorized Pauli matrix. Defaults to ``False``.

    Returns:
        list: list with all Pauli matrices forming the basis.
    """

    if nqubits <= 0:
        raise_error(ValueError, "nqubits must be a positive int.")

    if not isinstance(normalize, bool):
        raise_error(
            TypeError,
            f"normalize must be type bool, but it is type {type(normalize)} instead.",
        )

    if not isinstance(vectorize, bool):
        raise_error(
            TypeError,
            f"vectorize must be type bool, but it is type {type(vectorize)} instead.",
        )

    basis = [matrices.I, matrices.X, matrices.Y, matrices.Z]

    if vectorize:
        basis = [matrix.reshape((1, -1), order="F")[0] for matrix in basis]

    if nqubits >= 2:
        basis = list(product(basis, repeat=nqubits))
        if vectorize:
            basis = [reduce(np.outer, matrix).ravel() for matrix in basis]
        else:
            basis = [reduce(np.kron, matrix) for matrix in basis]

    basis = np.array(basis)

    if normalize:
        basis /= np.sqrt(2**nqubits)

    return basis


def comp_basis_to_pauli(nqubits: int, normalize: bool = False):
    """Unitary matrix :math:`U` that converts operators from the Liouville
    representation in the computational basis to the Pauli-Liouville
    representation.

    The unitary :math:`U` is given by

    ..math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, {|k)}{(P_{k}|} \\,\\, ,

    where :math:`{|A)}` is the system-vectorization of :math:`A`,
    :math:`{|k)}` is the vectorization of the computational basis element
    :math:`\\ketbra{k}`, and :math:`{|P_{k})}` is the vectorization of the
    :math:`k`-th Pauli matrix.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
        Pauli basis. Defaults to False.

    Returns:
        Unitary matrix :math:`U`.

    """

    unitary = pauli_basis(nqubits, normalize, vectorize=True)
    unitary = np.conj(unitary)

    return unitary


def pauli_to_comp_basis(nqubits: int, normalize: bool = False):
    """Unitary matrix :math:`U` that converts operators from the
    Pauli-Liouville representation to the Liouville representation
    in the computational basis.

    The unitary :math:`U` is given by

    ..math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, {|P_{k})}{(b_{k}|} \\, .

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
        Pauli basis. Defaults to False.

    Returns:
        Unitary matrix :math:`U`.
    """

    matrix = np.transpose(np.conj(comp_basis_to_pauli(nqubits, normalize)))

    return matrix
