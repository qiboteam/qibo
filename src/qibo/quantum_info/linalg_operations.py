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


def partial_trace(state, traced_qubits: Union[List[int], Tuple[int]], backend=None):
    """Returns the density matrix resulting from tracing out ``traced_qubits`` from ``state``.

    Total number of qubits is inferred by the shape of ``state``.

    Args:
        state (ndarray): density matrix or statevector.
        traced_qubits (Union[List[int], Tuple[int]]): indices of qubits to be traced out.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

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
    order += tuple(i for i in range(nqubits) if i not in traced_qubits)
    order += tuple(i + nqubits for i in order)
    shape = 2 * (2 ** len(traced_qubits), 2 ** (nqubits - len(traced_qubits)))

    state = backend.np.transpose(state, order)
    state = backend.np.reshape(state, shape)

    return backend.np.einsum("abac->bc", state)


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
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: matrix exponential of :math:`-i \\, \\theta \\, H`.
    """
    backend = _check_backend(backend)

    return backend.calculate_matrix_exp(phase, matrix, eigenvectors, eigenvalues)


def matrix_power(matrix, power: Union[float, int], backend=None):
    """Given a ``matrix`` :math:`A` and power :math:`\\alpha`, calculate :math:`A^{\\alpha}`.

    Args:
        matrix (ndarray): matrix whose power to calculate.
        power (float or int): power to raise ``matrix`` to.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            :class:`qibo.backends.GlobalBackend`. Defaults to ``None``.

    Returns:
        ndarray: matrix power :math:`A^{\\alpha}`.
    """
    backend = _check_backend(backend)

    return backend.calculate_matrix_power(matrix, power)
