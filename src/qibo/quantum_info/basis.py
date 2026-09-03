from typing import Optional

from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.quantum_info.utils import _get_single_paulis, _pauli_basis_normalization


def pauli_basis(
    nqubits: int,
    normalize: bool = False,
    vectorize: bool = False,
    sparse: bool = False,
    order: Optional[str] = None,
    pauli_order: str = "IXYZ",
    backend: Optional[Backend] = None,
):
    """Create the :math:`n`-qubit Pauli basis.

    For :math:`d = 2^{n}`, the returned Pauli basis is represented by the following array:

    .. math::
        \\mathcal{P} = \\mathcal{N} \\, \\left[ P_{0}, \\, P_{1}, \\,
            \\cdots, P_{d^{2} - 1} \\right] \\, ,

    where :math:`P_{k}` is the representation of the :math:`k`-th element of the Pauli basis,
    and :math:`\\mathcal{N}` is a normalization factor that equals to :math:`1/\\sqrt{d}` if
    ``normalize=True``, and :math:`1` otherwise.
    If ``vectorize=False``, each :math:`P_{k}` is the :math:`d \\times d` matrix representing
    the :math:`k`-th Pauli element. If ``vectorize=True``, then the Paulis are vectorized
    according to ``order`` (see :func:`qibo.quantum_info.vectorization`).

    Args:
        nqubits (int): number of qubits :math:`n`.
        normalize (bool, optional): if ``True``, :math:`\\mathcal{N} = 1/\\sqrt{d}`,
            and the normalized Pauli basis is returned. Defaults to ``False``.
        vectorize (bool, optional): if ``False``, returns a nested array with
            all Pauli matrices. If ``True``, retuns an array where every
            row is a vectorized Pauli matrix according to vectorization ``order``.
            Defaults to ``False``.
        sparse (bool, optional): if ``True``, retuns Pauli basis in a sparse
            representation. Defaults to ``False``.
        order (str, optional): if ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. If ``vectorization=False``, then ``order=None`` is
            forced. Defaults to ``None``.
        pauli_order (str, optional): corresponds to the order of :math:`4` single-qubit
            Pauli elements. Defaults to ``"IXYZ"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray or tuple: All Pauli matrices forming the basis. If ``sparse=True``
        and ``vectorize=True``, tuple is composed of an array of non-zero
        elements and an array with their row-wise indexes.
    """

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
    backend=None,
):
    """Unitary matrix :math:`U` that converts operators to the Pauli-Liouville representation.

    For :math:`d = 2^{n}`, the unitary matrix :math:`U` is given by

    .. math::
        U = \\sum_{k = 0}^{d^{2} - 1} \\, |k)(P_{k}| \\,\\, ,

    where :math:`|P_{k})` is the vectorization of the :math:`k`-th
    Pauli operator :math:`P_{k}`, and :math:`|k)` is the :math:`k`-th
    computational basis element in :math:`\\mathbb{C}^{d^{2}}`.
    For a definition of vectorization, see :func:`qibo.quantum_info.vectorization`.

    Example:
        .. code-block:: python

            # Imports below are equivalent to the following:
            # from qibo.quantum_info.basis import comp_basis_to_pauli
            # from qibo.quantum_info.random_ensembles import random_density_matrix
            # from qibo.quantum_info.superoperator_transformations import vectorization
            from qibo.quantum_info import comp_basis_to_pauli, random_density_matrix, vectorization

            nqubits = 2
            dims = 2**nqubits

            U_c2p = comp_basis_to_pauli(nqubits, order="system")

            rho = random_density_matrix(dims, pure=False)

            rho_liouville = vectorization(rho, order="system")
            rho_pauli_liouville = U_c2p @ rho_liouville

    Args:
        nqubits (int): number of qubits :math:`n`.
        normalize (bool, optional): If ``True``, returns unitary matrix that converts
            to the normalized Pauli basis. Defaults to ``False``.
        sparse (bool, optional): If ``True``, returns unitary matrix in
            sparse representation. Defaults to ``False``.
        order (str, optional): If ``"row"``, vectorization of Pauli basis is
            performed row-wise. If ``"column"``, vectorization is performed
            column-wise. If ``"system"``, system-wise vectorization is
            performed. Defaults to ``"row"``.
        pauli_order (str, optional): corresponds to the order of :math:`4` single-qubit
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
        elements = backend.conj(elements)

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

    unitary = backend.conj(unitary)

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

    if sparse:
        normalization = _pauli_basis_normalization(nqubits) if normalize else 1.0
        func = getattr(backend.qinfo, f"_pauli_to_comp_basis_sparse_{order}")
        return func(
            nqubits,
            *_get_single_paulis(pauli_order, backend),
            normalization=normalization,
        )

    return pauli_basis(
        nqubits,
        normalize,
        vectorize=True,
        sparse=False,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    ).T
