from numpy.typing import ArrayLike

from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info.utils import _get_single_paulis, _pauli_basis_normalization


def pauli_basis(
    nqubits: int,
    normalize: bool = False,
    vectorize: bool = False,
    sparse: bool = False,
    order: str | None = None,
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Creates the ``nqubits``-qubit Pauli basis.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, normalized basis is returned.
            Defaults to False.
        vectorize (bool, optional): If ``False``, returns a nested array with
            all Pauli matrices. If ``True``, returns an array where every
            row is a vectorized Pauli matrix. Defaults to ``False``.
        sparse (bool, optional): If ``True``, returns Pauli basis in a sparse
            representation. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Required when ``vectorize=True`` and ignored when
            ``vectorize=False``. Defaults to ``None``.
        pauli_order (str, optional): corresponds to the order of 4 single-qubit
            Pauli elements. Defaults to ``"IXYZ"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike or tuple: all Pauli matrices forming the basis. If ``sparse=True``
            and ``vectorize=True``, tuple is composed of an array of non-zero
            elements and an array with their row-wise indexes.
    """

    if vectorize and order not in ("row", "column", "system"):
        raise_error(
            ValueError,
            "when vectorize=True, order must be 'row', 'column' or 'system'. "
            f"Got {order} instead.",
        )

    if sparse and not vectorize:
        raise_error(
            NotImplementedError,
            "sparse representation is not implemented for unvectorized Pauli basis.",
        )

    backend = _check_backend(backend)
    fname = f"_pauli_basis_{order}"
    normalization = _pauli_basis_normalization(nqubits) if normalize else 1.0

    if vectorize and sparse:
        func = getattr(backend.qinfo, f"_vectorize_sparse{fname}")
    elif vectorize:
        func = getattr(backend.qinfo, f"_vectorize{fname}")
    else:
        func = backend.qinfo._pauli_basis

    return func(
        nqubits, *_get_single_paulis(pauli_order, backend), normalization=normalization
    )


def comp_basis_to_pauli(
    nqubits: int,
    normalize: bool = False,
    sparse: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Matrix :math:`U` that converts operators from the Liouville
    representation in the computational basis to the Pauli-Liouville
    representation.

    The matrix :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, |b_{k})(P_{k}| \\, ,

    where :math:`|P_{k})` is the vectorization of the :math:`k`-th
    Pauli operator :math:`P_{k}`, and :math:`|b_{k})` is the vectorization
    of the :math:`k`-th computational basis element.
    :math:`U` is unitary only if the Pauli basis is normalized, i.e. ``normalize=True``.
    For a definition of vectorization, see :func:`qibo.quantum_info.vectorization`.

    Example:
        .. code-block:: python

            from qibo.quantum_info import random_density_matrix, vectorization, comp_basis_to_pauli
            nqubits = 2
            d = 2**nqubits
            rho = random_density_matrix(d)
            U_c2p = comp_basis_to_pauli(nqubits, normalize=True, order="system")
            rho_liouville = vectorization(rho, order="system")
            rho_pauli_liouville = U_c2p @ rho_liouville

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
            normalized Pauli basis. Defaults to ``False``.
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
        ArrayLike or tuple: Matrix :math:`U`. If ``sparse=True``,
            tuple is composed of array of non-zero elements and an
            array with their row-wise indexes.
    """
    backend = _check_backend(backend)

    unitary = pauli_basis(
        nqubits,
        normalize,
        vectorize=True,
        sparse=sparse,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    if sparse:
        elements, indexes = unitary

        return backend.conj(elements), indexes

    return backend.conj(unitary)


def pauli_to_comp_basis(
    nqubits: int,
    normalize: bool = False,
    sparse: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
    """Matrix :math:`U` that converts operators from the
    Pauli-Liouville representation to the Liouville representation
    in the computational basis.

    The matrix :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, |P_{k})(b_{k}| \\, ,

    where :math:`|P_{k})` is the vectorization of the :math:`k`-th
    Pauli operator :math:`P_{k}`, and :math:`|b_{k})` is the vectorization
    of the :math:`k`-th computational basis element.
    :math:`U` is unitary only if the Pauli basis is normalized, i.e. ``normalize=True``.
    For a definition of vectorization, see :func:`qibo.quantum_info.vectorization`.

    Args:
        nqubits (int): number of qubits.
        normalize (bool, optional): If ``True``, converts to the
            normalized Pauli basis. Defaults to ``False``.
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
        ArrayLike or tuple: Matrix :math:`U`. If ``sparse=True``,
            tuple is composed of array of non-zero elements and an
            array with their row-wise indexes.
    """
    backend = _check_backend(backend)

    if sparse:
        if order not in ("row", "column", "system"):
            raise_error(
                ValueError,
                f"order must be 'row', 'column' or 'system'. Got {order} instead.",
            )

        normalization = _pauli_basis_normalization(nqubits) if normalize else 1.0
        func = getattr(backend.qinfo, f"_pauli_to_comp_basis_sparse_{order}")
        elements, indexes = func(
            nqubits,
            *_get_single_paulis(pauli_order, backend),
            normalization=normalization,
        )
        # the sparse template returns `indexes` flattened; reshape both to (d^2, d)
        shape = (4**nqubits, 2**nqubits)

        return backend.reshape(elements, shape), backend.reshape(indexes, shape)

    return pauli_basis(
        nqubits,
        normalize,
        vectorize=True,
        sparse=False,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    ).T
