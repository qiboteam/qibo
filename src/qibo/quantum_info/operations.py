"""Module with the most common linear algebra operations for quantum information."""

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
