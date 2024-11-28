from itertools import product
from typing import Optional

import numpy as np

from qibo import matrices
from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.quantum_info.superoperator_transformations import vectorization


def pauli_basis(
    nqubits: int,
    normalize: bool = False,
    vectorize: bool = False,
    sparse: bool = False,
    order: Optional[str] = None,
    pauli_order: str = "IXYZ",
    backend=None,
):
    """Creates the ``nqubits``-qubit Pauli basis.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, normalized basis is returned.
            Defaults to False.
        vectorize (bool, optional): If ``False``, returns a nested array with
            all Pauli matrices. If ``True``, retuns an array where every
            row is a vectorized Pauli matrix. Defaults to ``False``.
        sparse (bool, optional): If ``True``, retuns Pauli basis in a sparse
            representation. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. If ``vectorization=False``, then ``order=None`` is
            forced. Defaults to ``None``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to ``"IXYZ"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray or tuple: all Pauli matrices forming the basis. If ``sparse=True``
            and ``vectorize=True``, tuple is composed of an array of non-zero
            elements and an array with their row-wise indexes.
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

    if not isinstance(sparse, bool):
        raise_error(
            TypeError,
            f"sparse must be type bool, but it is type {type(sparse)} instead.",
        )

    if not isinstance(pauli_order, str):
        raise_error(
            TypeError,
            f"pauli_order must be type str, but it is type {type(pauli_order)} instead.",
        )

    if set(pauli_order) != {"I", "X", "Y", "Z"}:
        raise_error(
            ValueError,
            f"pauli_order has to contain 4 symbols: I, X, Y, Z. Got {pauli_order} instead.",
        )

    if vectorize and order is None:
        raise_error(ValueError, "when vectorize=True, order must be specified.")

    if sparse and not vectorize:
        raise_error(
            NotImplementedError,
            "sparse representation is not implemented for unvectorized Pauli basis.",
        )

    backend = _check_backend(backend)

    pauli_labels = {"I": matrices.I, "X": matrices.X, "Y": matrices.Y, "Z": matrices.Z}
    dim = 2**nqubits
    basis_single = backend.cast([pauli_labels[label] for label in pauli_order])
    einsum = np.einsum if backend.platform == "tensorflow" else backend.np.einsum

    if nqubits > 1:
        input_indices = [range(3 * i, 3 * (i + 1)) for i in range(nqubits)]
        output_indices = (i for indices in zip(*input_indices) for i in indices)
        operands = [basis_single for _ in range(nqubits)]
        inputs = [item for pair in zip(operands, input_indices) for item in pair]
        basis_full = einsum(*inputs, output_indices).reshape(4**nqubits, dim, dim)
    else:
        basis_full = basis_single

    if vectorize and sparse:
        if backend.platform == "pytorch":
            nonzero = lambda x: backend.np.nonzero(x, as_tuple=True)
        else:
            nonzero = backend.np.nonzero
        basis = vectorization(basis_full, order=order, backend=backend)
        indices = nonzero(backend.np.abs(basis))  # abs needed because of ``tensorflow``
        basis = basis[indices].reshape(-1, dim)
        indices = indices[1].reshape(-1, dim)

    elif vectorize and not sparse:
        basis = vectorization(basis_full, order=order, backend=backend)
    else:
        basis = basis_full

    if normalize:
        basis = basis / np.sqrt(2**nqubits)

    if vectorize and sparse:
        return basis, indices

    return basis


def comp_basis_to_pauli(
    nqubits: int,
    normalize: bool = False,
    sparse: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
):
    """Unitary matrix :math:`U` that converts operators from the Liouville
    representation in the computational basis to the Pauli-Liouville
    representation.

    The unitary :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, |k)(P_{k}| \\,\\, ,

    where :math:`|P_{k})` is the vectorization of the :math:`k`-th
    Pauli operator :math:`P_{k}`, and :math:`|k)` is the vectorization
    of the :math:`k`-th computational basis element.
    For a definition of vectorization, see :func:`qibo.quantum_info.vectorization`.

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
        sparse (bool, optional): If ``True``, returns unitary matrix in
            sparse representation. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to ``"IXYZ"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray or tuple: Unitary matrix :math:`U`. If ``sparse=True``,
            tuple is composed of array of non-zero elements and an
            array with their row-wise indexes.

    """
    backend = _check_backend(backend)

    if sparse:
        elements, indexes = pauli_basis(
            nqubits,
            normalize,
            vectorize=True,
            sparse=sparse,
            order=order,
            pauli_order=pauli_order,
            backend=backend,
        )
        elements = backend.np.conj(elements)

        return elements, indexes

    unitary = pauli_basis(
        nqubits,
        normalize,
        vectorize=True,
        sparse=sparse,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    unitary = backend.np.conj(unitary)

    return unitary


def pauli_to_comp_basis(
    nqubits: int,
    normalize: bool = False,
    sparse: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend=None,
):
    """Unitary matrix :math:`U` that converts operators from the
    Pauli-Liouville representation to the Liouville representation
    in the computational basis.

    The unitary :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, |P_{k})(b_{k}| \\, ,

    where :math:`|P_{k})` is the vectorization of the :math:`k`-th
    Pauli operator :math:`P_{k}`, and :math:`|k)` is the vectorization
    of the :math:`k`-th computational basis element.
    For a definition of vectorization, see :func:`qibo.quantum_info.vectorization`.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
            Pauli basis. Defaults to ``False``.
        sparse (bool, optional): If ``True``, returns unitary matrix in
            sparse representation. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to ``"IXYZ"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be
            used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray or tuple: Unitary matrix :math:`U`. If ``sparse=True``,
            tuple is composed of array of non-zero elements and an
            array with their row-wise indexes.
    """
    backend = _check_backend(backend)

    unitary = pauli_basis(
        nqubits,
        normalize,
        vectorize=True,
        sparse=False,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    unitary = unitary.T

    if sparse:
        elements, indexes = [], []
        for row in unitary:
            index_list = backend.np.flatnonzero(row)
            indexes.append(index_list)
            elements.append(row[index_list])

        elements = backend.cast(elements)
        indexes = backend.cast(indexes)

        return elements, indexes

    return unitary
