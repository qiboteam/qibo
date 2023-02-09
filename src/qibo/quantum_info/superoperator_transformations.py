import numpy as np

from qibo.config import raise_error
from qibo.gates.abstract import Gate
from qibo.gates.gates import Unitary
from qibo.gates.special import FusedGate


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
        ndarray: Liouville representation of ``state``.
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
        ndarray: Density matrix of ``state``.
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


def liouville_to_choi(super_op, order: str = "row"):
    """Convert Liouville representation of quantum channel
    to its Choi representation.

    Args:
        super_op: Liouville representation of quanutm channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Default is ``"row"``.
    Returns:
        ndarray: Choi representation of quantum channel.
    """

    return _reshuffling(super_op, order=order)


def choi_to_liouville(choi_super_op, order: str = "row"):
    """Convert Choi representation of quantum channel
    to its Liouville representation.

    Args:
        choi_super_op: Choi representation of quanutm channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Default is ``"row"``.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """

    return _reshuffling(choi_super_op, order=order)


def choi_to_kraus(
    choi_super_op,
    precision_tol: float = None,
    order: str = "row",
    validate_CP: bool = True,
):
    """Convert Choi representation of a quantum channel into Kraus operators.

    Args:
        choi_super_op: Choi representation of a quantum channel.
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda < \\text{precision_tol}` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Default is ``"row"``.
        validate_CP (bool, optional): If ``True``, checks if ``choi_super_op``
            is a completely positive map. If ``False``, it assumes that
            ``choi_super_op`` is completely positive (and Hermitian).
            Defaults to ``True``.

    Returns:
        (ndarray, ndarray) or (ndarray, ndarray, ndarray): Kraus operators of
            quantum channel and their respective coefficients. If map is non-CP,
            then function returns left- and right-generalized Kraus operators
            as well as the square root of their corresponding singular values.
    """

    if precision_tol is not None and not isinstance(precision_tol, float):
        raise_error(
            TypeError,
            f"precision_tol must be type float, but it is type {type(precision_tol)}",
        )

    if precision_tol is not None and precision_tol < 0:
        raise_error(
            ValueError,
            f"precision_tol must be a non-negative float, but it is {precision_tol}.",
        )

    if precision_tol is None:  # pragma: no cover
        from qibo.config import PRECISION_TOL

        precision_tol = PRECISION_TOL

    if not isinstance(validate_CP, bool):
        raise_error(
            TypeError,
            f"validate_CP must be type bool, but it is type {type(validate_CP)}.",
        )

    if validate_CP:
        norm = np.linalg.norm(choi_super_op - np.transpose(np.conj(choi_super_op)))
        if norm > PRECISION_TOL:
            non_CP = True
        else:
            # using eigh because, in this case, choi_super_op is
            # *already confirmed* to be Hermitian
            eigenvalues, eigenvectors = np.linalg.eigh(choi_super_op)
            eigenvectors = np.transpose(eigenvectors)

            non_CP = True if any(eigenvalues < -PRECISION_TOL) else False
    else:
        non_CP = False
        # using eigh because, in this case, choi_super_op is
        # *assumed* to be Hermitian
        eigenvalues, eigenvectors = np.linalg.eigh(choi_super_op)
        eigenvectors = np.transpose(eigenvectors)

    if non_CP:
        from warnings import warn

        warn("Input choi_super_op is a non-completely positive map.")

        # using singular value decomposition because choi_super_op is non-CP
        U, S, V = np.linalg.svd(choi_super_op)
        U = np.transpose(U)
        S = np.sqrt(S)
        V = np.conj(V)

        kraus_left, kraus_right = list(), list()
        for eigenvector_left, eigenvector_right in zip(U, V):
            kraus_left.append(unvectorization(eigenvector_left, order=order))
            kraus_right.append(unvectorization(eigenvector_right, order=order))

        kraus_left = np.array(kraus_left)
        kraus_right = np.array(kraus_right)

        return kraus_left, kraus_right, S
    else:
        # when choi_super_op is CP
        kraus_ops, coefficients = list(), list()
        for eig, kraus in zip(eigenvalues, eigenvectors):
            if np.abs(eig) > precision_tol:
                kraus_ops.append(unvectorization(kraus, order=order))
                coefficients.append(np.sqrt(eig))

        kraus_ops = np.array(kraus_ops)
        coefficients = np.array(coefficients)

        return kraus_ops, coefficients


def kraus_to_choi(kraus_ops, order: str = "row"):
    """Convert Kraus representation of quantum channel
    to its Choi representation.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k`  acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Default is ``"row"``.

    Returns:
        ndarray: Choi representation of the Kraus channel.
    """

    from qibo.backends import NumpyBackend

    backend = NumpyBackend()

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)
    nqubits = 1 + max(target_qubits)
    d = 2**nqubits

    super_op = np.zeros((d**2, d**2), dtype="complex")
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.asmatrix(backend)
        kraus_op = vectorization(kraus_op, order=order)
        super_op += np.outer(kraus_op, np.conj(kraus_op))
        del kraus_op

    return super_op


def kraus_to_liouville(kraus_ops, order: str = "row"):
    """Convert from Kraus representation to Liouville representation.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Default is ``"row"``.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """
    super_op = kraus_to_choi(kraus_ops, order=order)
    super_op = choi_to_liouville(super_op, order=order)

    return super_op


def liouville_to_kraus(super_op, precision_tol: float = None, order: str = "row"):
    """Convert Liouville representation of a quantum channel to
    its Kraus representation. It uses the Choi representation as
    an intermediate step.

    Args:
        super_op (ndarray): Liouville representation of quantum channel.
        precision_tol (float, optional): Precision tolerance for eigenvalues
            found in the spectral decomposition problem. Any eigenvalue
            :math:`\\lambda < \\text{precision_tol}` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to None.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Default is ``"row"``.

    Returns:
        (ndarray, ndarray): Kraus operators of quantum channel and their
            respective coefficients.
    """
    choi_super_op = liouville_to_choi(super_op, order=order)
    kraus_ops, coefficients = choi_to_kraus(choi_super_op, precision_tol, order=order)

    return kraus_ops, coefficients


def _reshuffling(super_op, order: str = "row"):
    """Reshuffling operation used to convert Lioville representation
    of quantum channels to their Choi representation (and vice-versa).

    Args:
        super_op (ndarray): Liouville (Choi) representation of a
            quantum channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. If ``"system"``, operator is converted to
            a representation based on row vectorization, reshuffled,
            and then converted back to its representation with
            respect to system-wise vectorization. Default is ``"row"``.

    Returns:
        ndarray: Choi (Liouville) representation of the quantum channel.
    """

    if not isinstance(order, str):
        raise_error(TypeError, f"order must be type str, but it is type {type(order)}.")

    orders = ["row", "column", "system"]
    if order not in orders:
        raise_error(
            ValueError,
            f"order must be either 'row' or 'column' or 'system', but it is {order}.",
        )
    del orders

    if order == "system":
        raise_error(
            NotImplementedError, "reshuffling not implemented for system vectorization."
        )

    d = int(np.sqrt(super_op.shape[0]))

    super_op = np.reshape(super_op, [d] * 4)

    axes = [1, 2] if order == "row" else [0, 3]
    super_op = np.swapaxes(super_op, *axes)

    super_op = np.reshape(super_op, [d**2, d**2])

    return super_op


def _set_gate_and_target_qubits(kraus_ops):  # pragma: no cover
    """Returns Kraus operators as a set of gates acting on
    their respective ``target qubits``.

    Args:
        kraus_ops (list): List of Kraus operators as pairs ``(qubits, Ak)``
            where ``qubits`` refers the qubit ids that :math:`A_k` acts on
            and :math:`A_k` is the corresponding matrix as a ``np.ndarray``.

    Returns:
        (tuple, tuple): gates and their respective target qubits.
    """
    if isinstance(kraus_ops[0], Gate):
        gates = tuple(kraus_ops)
        target_qubits = tuple(
            sorted({q for gate in kraus_ops for q in gate.target_qubits})
        )
    else:
        gates, qubitset = [], set()
        for qubits, matrix in kraus_ops:
            rank = 2 ** len(qubits)
            shape = tuple(matrix.shape)
            if shape != (rank, rank):
                raise_error(
                    ValueError,
                    "Invalid Krauss operator shape {} for "
                    "acting on {} qubits."
                    "".format(shape, len(qubits)),
                )
            qubitset.update(qubits)
            gates.append(Unitary(matrix, *list(qubits)))
        gates = tuple(gates)
        target_qubits = tuple(sorted(qubitset))

    return gates, target_qubits
