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


def liouville_to_choi(super_op):
    """Convert Liouville representation of quantum channel
    to its Choi representation.

    Args:
        super_op: Liouville representation of quanutm channel.

    Returns:
        ndarray: Choi representation of quantum channel.
    """

    return _reshuffling(super_op)


def choi_to_liouville(choi_super_op):
    """Convert Choi representation of quantum channel
    to its Liouville representation.

    Args:
        choi_super_op: Choi representation of quanutm channel.

    Returns:
        ndarray: Liouville representation of quantum channel.
    """

    return _reshuffling(choi_super_op)

def choi_to_kraus(choi_super_op, precision_tol: float = None):
    """Convert Choi representation of a quantum channel into Kraus operators.

    Args:
        choi_super_op: Choi representation of a quantum channel.
        precision_tol (float, optional): Precision tolerance for eigenvalues 
            found in the spectral decomposition problem. Any eigenvalue 
            :math:`\\lambda < \\text{precision_tol}` is set to 0 (zero).
            If ``None``, ``precision_tol`` defaults to 
            ``qibo.config.PRECISION_TOL=1e-8``. Defaults to ``None``.

    Returns:
        ndarray: Kraus operators of quantum channel.
        ndarray: coefficients of Kraus operators.
    """

    if precision_tol is None: # pragma: no cover
        from qibo.config import PRECISION_TOL

        precision_tol = PRECISION_TOL

    # using eigh because Choi representation is, 
    # in theory,    always Hermitian
    eigenvalues, eigenvectors = np.linalg.eigh(choi_super_op)
    eigenvectors = np.transpose(eigenvectors)


    kraus_ops, coefficients = list(), list()
    for eig, kraus in zip(eigenvalues, eigenvectors):
        if np.abs(eig) > precision_tol:
            kraus_ops.append(unvectorization(kraus))
            coefficients.append(np.sqrt(eig))

    kraus_ops = np.array(kraus_ops)
    coefficients = np.array(coefficients)

    return kraus_ops, coefficients


def kraus_to_choi(kraus_ops):
    """Convert Kraus representation of quantum channel
    to its Choi representation.

    Args:
        ops (list): List of Kraus operators as pairs ``(qubits, Ak)`` where
            ``qubits`` refers the qubit ids that :math:`A_k` acts on and 
            :math:`A_k` is the corresponding matrix as a ``np.ndarray``.

    Returns:
        ndarray: Choi representation of the Kraus channel.
    """

    from qibo.backends import NumpyBackend

    backend = NumpyBackend()

    gates, target_qubits = _set_gate_and_target_qubits(kraus_ops)
    nqubits = 1 + max(target_qubits)
    d = 2 ** nqubits

    super_op = np.zeros((d**2, d**2), dtype="complex")
    for gate in gates:
        kraus_op = FusedGate(*range(nqubits))
        kraus_op.append(gate)
        kraus_op = kraus_op.asmatrix(backend)
        kraus_op = vectorization(kraus_op)
        super_op += np.outer(kraus_op, np.conj(kraus_op))
        del kraus_op

    return super_op


def kraus_to_liouville(kraus_ops, coefficients=None):

    super_op = kraus_to_choi(kraus_ops, coefficients)
    super_op = choi_to_liouville(super_op)

    return super_op


def liouville_to_kraus(super_op, precision_tol: float = None):

    choi_super_op = liouville_to_choi(super_op)
    kraus_ops, coefficients = choi_to_kraus(choi_super_op, precision_tol)

    return kraus_ops, coefficients


def _reshuffling(super_operator):

    d = int(np.sqrt(super_operator.shape[0]))

    super_operator = np.reshape(super_operator, [d] * 4)
    super_operator = np.swapaxes(super_operator, 0, 3)
    super_operator = np.reshape(super_operator, [d**2, d**2])

    return super_operator


def _set_gate_and_target_qubits(kraus_ops):

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