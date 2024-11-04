"""Module with common linear algebra operations for quantum information."""

import math
from typing import List, Tuple, Union

from qibo.backends import _check_backend
from qibo.config import raise_error


def commutator(operator_1, operator_2):
    """Returns the commutator of ``operator_1`` and ``operator_2``.

    The commutator of two matrices :math:`A` and :math:`B` is given by

    .. math::
        [A, B] = A \\, B - B \\, A \\,.

    Args:
        operator_1 (ndarray): First operator.
        operator_2 (ndarray): Second operator.

    Returns:
        ndarray: Commutator of ``operator_1`` and ``operator_2``.
    """
    if (
        (len(operator_1.shape) >= 3)
        or (len(operator_1) == 0)
        or (len(operator_1.shape) == 2 and operator_1.shape[0] != operator_1.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_1`` must have shape (k,k), but have shape {operator_1.shape}.",
        )

    if (
        (len(operator_2.shape) >= 3)
        or (len(operator_2) == 0)
        or (len(operator_2.shape) == 2 and operator_2.shape[0] != operator_2.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_2`` must have shape (k,k), but have shape {operator_2.shape}.",
        )

    if operator_1.shape != operator_2.shape:
        raise_error(
            TypeError,
            "``operator_1`` and ``operator_2`` must have the same shape, "
            + f"but {operator_1.shape} != {operator_2.shape}",
        )

    return operator_1 @ operator_2 - operator_2 @ operator_1


def anticommutator(operator_1, operator_2):
    """Returns the anticommutator of ``operator_1`` and ``operator_2``.

    The anticommutator of two matrices :math:`A` and :math:`B` is given by

    .. math::
        \\{A, B\\} = A \\, B + B \\, A \\,.

    Args:
        operator_1 (ndarray): First operator.
        operator_2 (ndarray): Second operator.

    Returns:
        ndarray: Anticommutator of ``operator_1`` and ``operator_2``.
    """
    if (
        (len(operator_1.shape) >= 3)
        or (len(operator_1) == 0)
        or (len(operator_1.shape) == 2 and operator_1.shape[0] != operator_1.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_1`` must have shape (k,k), but have shape {operator_1.shape}.",
        )

    if (
        (len(operator_2.shape) >= 3)
        or (len(operator_2) == 0)
        or (len(operator_2.shape) == 2 and operator_2.shape[0] != operator_2.shape[1])
    ):
        raise_error(
            TypeError,
            f"``operator_2`` must have shape (k,k), but have shape {operator_2.shape}.",
        )

    if operator_1.shape != operator_2.shape:
        raise_error(
            TypeError,
            "``operator_1`` and ``operator_2`` must have the same shape, "
            + f"but {operator_1.shape} != {operator_2.shape}",
        )

    return operator_1 @ operator_2 + operator_2 @ operator_1


def partial_trace(
    state, traced_qubits: Union[List[int], Tuple[int, ...]], backend=None
):
    """Returns the density matrix resulting from tracing out ``traced_qubits`` from ``state``.

    Total number of qubits is inferred by the shape of ``state``.

    Args:
        state (ndarray): density matrix or statevector.
        traced_qubits (Union[List[int], Tuple[int]]): indices of qubits to be traced out.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: Density matrix of the remaining qubit(s).
    """
    if (
        (len(state.shape) >= 3)
        or (len(state) == 0)
        or (len(state.shape) == 2 and state.shape[0] != state.shape[1])
    ):
        raise_error(
            TypeError,
            f"``state`` must have dims either (k,) or (k,k), but have dims {state.shape}.",
        )

    backend = _check_backend(backend)

    state = backend.cast(state, dtype=state.dtype)
    nqubits = math.log2(state.shape[0])

    if not nqubits.is_integer():
        raise_error(
            ValueError,
            "dimension(s) of ``state`` must be a power of 2, "
            + f"but it is {2**nqubits}.",
        )

    nqubits = int(nqubits)

    statevector = bool(len(state.shape) == 1)

    factor = 1 if statevector else 2
    state = backend.np.reshape(state, factor * nqubits * (2,))

    if statevector:
        axes = 2 * [list(traced_qubits)]
        rho = backend.np.tensordot(state, backend.np.conj(state), axes)
        shape = 2 * (2 ** (nqubits - len(traced_qubits)),)

        return backend.np.reshape(rho, shape)

    order = tuple(sorted(traced_qubits))
    order += tuple(set(list(range(nqubits))) ^ set(traced_qubits))
    order += tuple(k + nqubits for k in order)
    shape = 2 * (2 ** len(traced_qubits), 2 ** (nqubits - len(traced_qubits)))

    state = backend.np.transpose(state, order)
    state = backend.np.reshape(state, shape)

    return backend.np.einsum("abac->bc", state)


def partial_transpose(
    operator, partition: Union[List[int], Tuple[int, ...]], backend=None
):
    """Return matrix after the partial transposition of ``partition`` qubits in ``operator``.

    Given a :math:`n`-qubit operator :math:`O \\in \\mathcal{H}_{A} \\otimes \\mathcal{H}_{B}`,
    the partial transpose with respect to ``partition`` :math:`B` is given by

    .. math::
        \\begin{align}
        O^{T_{B}} &= \\sum_{jklm} \\, O_{lm}^{jk} \\, \\ketbra{j}{k} \\otimes
            \\left(\\ketbra{l}{m}\\right)^{T} \\\\
        &= \\sum_{jklm} \\, O_{lm}^{jk} \\, \\ketbra{j}{k} \\otimes \\ketbra{m}{l} \\\\
        &= \\sum_{jklm} \\, O_{ml}^{jk} \\, \\ketbra{j}{k} \\otimes \\ketbra{l}{m} \\, ,
        \\end{align}

    where the superscript :math:`T` indicates the transposition operation,
    and :math:`T_{B}` indicates transposition on ``partition`` :math:`B`.
    The total number of qubits is inferred by the shape of ``operator``.

    Args:
        operator (ndarray): :math:`1`- or :math:`2`-dimensional operator, or an array of
            :math:`1`- or :math:`2`-dimensional operators,
        partition (Union[List[int], Tuple[int, ...]]): indices of qubits to be transposed.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            it uses the current backend. Defaults to ``None``.

    Returns:
        ndarray: Partially transposed operator(s) :math:`\\O^{T_{B}}`.
    """
    backend = _check_backend(backend)

    shape = operator.shape
    nstates = shape[0]
    dims = shape[-1]
    nqubits = math.log2(dims)

    if not nqubits.is_integer():
        raise_error(
            ValueError,
            f"dimensions of ``state`` (or states in a batch) must be a power of 2.",
        )

    if (len(shape) > 3) or (nstates == 0) or (len(shape) == 2 and nstates != dims):
        raise_error(
            TypeError,
            "``operator`` must have dims either (k,), (k, k), (N, 1, k) or (N, k, k), "
            + f"but has dims {shape}.",
        )

    nqubits = int(nqubits)

    if len(shape) == 1:
        operator = backend.np.outer(operator, backend.np.conj(operator.T))
    elif len(shape) == 3 and shape[1] == 1:
        operator = backend.np.einsum(
            "aij,akl->aijkl", operator, backend.np.conj(operator)
        ).reshape(nstates, dims, dims)

    new_shape = list(range(2 * nqubits + 1))
    for ind in partition:
        ind += 1
        new_shape[ind] = ind + nqubits
        new_shape[ind + nqubits] = ind
    new_shape = tuple(new_shape)

    reshaped = backend.np.reshape(operator, [-1] + [2] * (2 * nqubits))
    reshaped = backend.np.transpose(reshaped, new_shape)

    final_shape = (dims, dims)
    if len(operator.shape) == 3:
        final_shape = (nstates,) + final_shape

    return backend.np.reshape(reshaped, final_shape)


def matrix_exponentiation(
    phase: Union[float, complex],
    matrix,
    eigenvectors=None,
    eigenvalues=None,
    backend=None,
):
    """Calculates the exponential of a matrix.

    Given a ``matrix`` :math:`H` and a ``phase`` :math:`\\theta`,
    it returns the exponential of the form

    .. math::
        \\exp\\left(-i \\, \\theta \\, H \\right) \\, .

    If the ``eigenvectors`` and ``eigenvalues`` are given, the matrix diagonalization
    is used for the exponentiation.

    Args:
        phase (float or complex): phase that multiplies the matrix.
        matrix (ndarray): matrix to be exponentiated.
        eigenvectors (ndarray, optional): _if not ``None``, eigenvectors are used
            to calculate ``matrix`` exponentiation as part of diagonalization.
            Must be used together with ``eigenvalues``. Defaults to ``None``.
        eigenvalues (ndarray, optional): if not ``None``, eigenvalues are used
            to calculate ``matrix`` exponentiation as part of diagonalization.
            Must be used together with ``eigenvectors``. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: matrix exponential of :math:`-i \\, \\theta \\, H`.
    """
    backend = _check_backend(backend)

    return backend.calculate_matrix_exp(phase, matrix, eigenvectors, eigenvalues)


def matrix_power(
    matrix, power: Union[float, int], precision_singularity: float = 1e-14, backend=None
):
    """Given a ``matrix`` :math:`A` and power :math:`\\alpha`, calculate :math:`A^{\\alpha}`.

    Args:
        matrix (ndarray): matrix whose power to calculate.
        power (float or int): power to raise ``matrix`` to.
        precision_singularity (float, optional): If determinant of ``matrix`` is smaller than
            ``precision_singularity``, then matrix is considered to be singular.
            Used when ``power`` is negative.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray: matrix power :math:`A^{\\alpha}`.
    """
    backend = _check_backend(backend)

    return backend.calculate_matrix_power(matrix, power, precision_singularity)


def singular_value_decomposition(matrix, backend=None):
    """Calculate the Singular Value Decomposition (SVD) of ``matrix``.

    Given an :math:`M \\times N` complex matrix :math:`A`, its SVD is given by

    .. math:
        A = U \\, S \\, V^{\\dagger} \\, ,

    where :math:`U` and :math:`V` are, respectively, an :math:`M \\times M`
    and an :math:`N \\times N` complex unitary matrices, and :math:`S` is an
    :math:`M \\times N` diagonal matrix with the singular values of :math:`A`.

    Args:
        matrix (ndarray): matrix whose SVD to calculate.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ndarray, ndarray, ndarray: Singular value decomposition of :math:`A`, i.e.
        :math:`U`, :math:`S`, and :math:`V^{\\dagger}`, in that order.
    """
    backend = _check_backend(backend)

    return backend.calculate_singular_value_decomposition(matrix)


def schmidt_decomposition(
    state, partition: Union[List[int], Tuple[int, ...]], backend=None
):
    """Return the Schmidt decomposition of a :math:`n`-qubit bipartite pure quantum ``state``.

    Given a bipartite pure state :math:`\\ket{\\psi}\\in\\mathcal{H}_{A}\\otimes\\mathcal{H}_{B}`,
    its Schmidt decomposition is given by

    .. math::
        \\ket{\\psi} = \\sum_{k = 1}^{\\min\\{a, \\, b\\}} \\, c_{k} \\,
            \\ket{\\phi_{k}} \\otimes \\ket{\\nu_{k}} \\, ,

    with :math:`a` and :math:`b` being the respective cardinalities of :math:`\\mathcal{H}_{A}`
    and :math:`\\mathcal{H}_{B}`, and :math:`\\{\\phi_{k}\\}_{k\\in[\\min\\{a, \\, b\\}]}
    \\subset \\mathcal{H}_{A}` and :math:`\\{\\nu_{k}\\}_{k\\in[\\min\\{a, \\, b\\}]}
    \\subset \\mathcal{H}_{B}` being orthonormal sets. The coefficients
    :math:`\\{c_{k}\\}_{k\\in[\\min\\{a, \\, b\\}]}` are real, non-negative, and unique
    up to re-ordering.

    The decomposition is calculated using :func:`qibo.quantum_info.singular_value_decomposition`,
    resulting in

    .. math::
        \\ketbra{\\psi}{\\psi} = U \\, S \\, V^{\\dagger} \\, ,

    where :math:`U` is an :math:`a \\times a` unitary matrix, :math:`V` is an :math:`b \\times b`
    unitary matrix, and :math:`S` is an :math:`a \\times b` positive semidefinite diagonal matrix
    that contains the singular values of :math:`\\ketbra{\\psi}{\\psi}`.

    Args:
        state (ndarray): stevector or density matrix.
        partition (Union[List[int], Tuple[int, ...]]): indices of qubits in one of the two
            partitions. The other partition is inferred as the remaining qubits.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray, ndarray, ndarray: Respectively, the matrices :math:`U`, :math:`S`,
        and :math:`V^{\\dagger}`.
    """
    backend = _check_backend(backend)

    nqubits = math.log2(state.shape[-1])
    if not nqubits.is_integer():
        raise_error(ValueError, f"dimensions of ``state`` must be a power of 2.")

    nqubits = int(nqubits)
    partition_2 = partition.__class__(set(list(range(nqubits))) ^ set(partition))

    tensor = backend.np.reshape(state, [2] * nqubits)
    tensor = backend.np.transpose(tensor, partition + partition_2)
    tensor = backend.np.reshape(tensor, (2 ** len(partition), -1))

    return singular_value_decomposition(tensor, backend=backend)
