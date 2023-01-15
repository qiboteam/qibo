from functools import reduce
from itertools import product

import numpy as np

from qibo import matrices
from qibo.config import raise_error


def vectorization(state, order: str = "row"):
    """Returns state :math:`\\rho` in its Liouville
    representation :math:`\\ket{\\rho}`.

    Args:
        state: state vector or density matrix.
        order (str, optional): If ``"row"``, vectorization is performed
            row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, a block-vectorization is
            performed. Default is ``"row"``.

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

    if not isinstance(order, str):
        raise_error(
            TypeError, f"order must be type str, but it is type {type(order)} instead."
        )
    else:
        if (order != "row") and (order != "column") and (order != "system"):
            raise_error(
                ValueError,
                f"order must be either 'row' or 'column' or 'system', but it is {order}.",
            )

    if len(state.shape) == 1:
        state = np.outer(state, np.conj(state))

    if order == "row":
        state = np.reshape(state, (1, -1), order="C")[0]
    elif order == "column":
        state = np.reshape(state, (1, -1), order="F")[0]
    else:
        d = len(state)
        nqubits = int(np.log2(d))

        new_axis = []
        for x in range(nqubits):
            new_axis += [x + nqubits, x]
        state = np.reshape(
            np.transpose(np.reshape(state, [2] * 2 * nqubits), axes=new_axis), -1
        )

    return state


def unvectorization(state, order: str = "row"):
    """Returns state :math:`\\rho` from its Liouville
    representation :math:`\\ket{\\rho}`.

    Args:
        state: :func:`vectorization` of a quantum state.
        order (str, optional): If ``"row"``, unvectorization is performed
            row-wise. If ``"column"``, unvectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Default is ``"row"``.

    Returns:
        Density matrix of ``state``.
    """

    if len(state.shape) != 1:
        raise_error(
            TypeError,
            f"Object must have dims (k,), but have dims {state.shape}.",
        )

    if not isinstance(order, str):
        raise_error(
            TypeError, f"order must be type str, but it is type {type(order)} instead."
        )
    else:
        if (order != "row") and (order != "column") and (order != "system"):
            raise_error(
                ValueError,
                f"order must be either 'row' or 'column' or 'system', but it is {order}.",
            )

    d = int(np.sqrt(len(state)))

    if (order == "row") or (order == "column"):
        order = "C" if order == "row" else "F"
        state = np.reshape(state, (d, d), order=order)
    else:
        nqubits = int(np.log2(d))
        axes_old = list(np.arange(0, 2 * nqubits))
        state = np.reshape(
            np.transpose(
                np.reshape(state, [2] * 2 * nqubits),
                axes=axes_old[1::2] + axes_old[0::2],
            ),
            [2**nqubits] * 2,
        )

    return state


def pauli_basis(
    nqubits: int, normalize: bool = False, vectorize: bool = False, order: str = None
):
    """Creates the ``nqubits``-qubit Pauli basis.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, normalized basis is returned.
            Defaults to False.
        vectorize (bool, optional): If ``False``, returns a nested array with
            all Pauli matrices. If ``True``, retuns an array where every
            row is a vectorized Pauli matrix. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. If ``vectorization=False``, then ``order=None`` is
            forced. Default is ``None``.

    Returns:
        ndarray: all Pauli matrices forming the basis.
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

    if vectorize and order is None:
        raise_error(ValueError, "when vectorize=True, order must be specified.")

    basis = [matrices.I, matrices.X, matrices.Y, matrices.Z]

    if nqubits >= 2:
        basis = list(product(basis, repeat=nqubits))
        if vectorize:
            basis = [
                vectorization(reduce(np.kron, matrices), order=order)
                for matrices in basis
            ]
        else:
            basis = [reduce(np.kron, matrices) for matrices in basis]
    else:
        if vectorize:
            basis = [vectorization(matrix, order=order) for matrix in basis]

    basis = np.array(basis)

    if normalize:
        basis /= np.sqrt(2**nqubits)

    return basis


def comp_basis_to_pauli(nqubits: int, normalize: bool = False, order: str = "row"):
    """Unitary matrix :math:`U` that converts operators from the Liouville
    representation in the computational basis to the Pauli-Liouville
    representation.

    The unitary :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, \\ketbra{k}{P_{k}} \\,\\, ,

    where :math:`\\ket{P_{k}}` is the system-vectorization of the :math:`k`-th
    Pauli operator :math:`P_{k}`, and :math:`\\ket{k}` is the computational
    basis element.

    When converting a state :math:`\\ket{\\rho}` to its Pauli-Liouville
    representation :math:`\\ket{\\rho'}`, one should use ``order="system"``
    in :func:`vectorization`.

    Example:
        .. code-block:: python

            from qibo.quantum_info import random_density_matrix, vectorization, comp_basis_to_pauli
            nqubits = 2
            d = 2**nqubits
            rho = random_density_matrix(d)
            U_c2p = comp_basis_to_pauli(nqubits)
            rho_liouville = vectorization(rho, order="system")
            rho_pauli_liouville = U_c2p @ rho_liouville

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Default is ``"row"``.

    Returns:
        Unitary matrix :math:`U`.

    """

    unitary = pauli_basis(nqubits, normalize, vectorize=True, order=order)
    unitary = np.conj(unitary)

    return unitary


def pauli_to_comp_basis(nqubits: int, normalize: bool = False, order: str = "row"):
    """Unitary matrix :math:`U` that converts operators from the
    Pauli-Liouville representation to the Liouville representation
    in the computational basis.

    The unitary :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, \\ketbra{P_{k}}{b_{k}} \\, .

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
            Pauli basis. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Default is ``"row"``.

    Returns:
        Unitary matrix :math:`U`.
    """

    unitary = pauli_basis(nqubits, normalize, vectorize=True, order=order)
    unitary = np.transpose(unitary)

    return unitary
