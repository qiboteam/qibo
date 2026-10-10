import math
from functools import reduce

from numpy.typing import ArrayLike

from qibo.backends import Backend, _check_backend
from qibo.config import PRECISION_TOL, raise_error
from qibo.gates.abstract import Gate
from qibo.gates.channels import Channel
from qibo.gates.gates import Unitary
from qibo.quantum_info.utils import _pauli_basis_normalization


def _check_pauli_transform_method(method: str | None) -> str:
    """Validate ``method`` for Pauli-basis conversions."""
    if method is not None and method not in ("dense", "fht", "standard"):
        raise_error(
            ValueError,
            f"``method`` must be either None, 'fht', 'standard', or 'dense', but it is {method}.",
        )

    if method is None or method == "fht":
        return "fht"

    return "standard"


def _check_pauli_superoperator_shape(super_op: ArrayLike, name: str) -> tuple[int, int]:
    """Validate the shape of a Pauli or Liouville superoperator."""
    shape = tuple(super_op.shape)

    # exponent of the dimension, which has to be a power of 4 with n >= 1
    exponent = shape[0].bit_length() - 1 if len(shape) == 2 else 0
    if (
        len(shape) != 2
        or shape[0] != shape[1]
        or 2**exponent != shape[0]
        or exponent == 0
        or exponent % 2 != 0
    ):
        raise_error(
            ValueError,
            f"{name} must be of shape (4^n, 4^n), but it is {super_op.shape}",
        )

    nqubits = exponent // 2

    return 2**nqubits, nqubits


def _fast_walsh_hadamard_transform(
    array: ArrayLike, axis: int = -1, backend: Backend | None = None
) -> ArrayLike:
    """Apply an unnormalized Fast Walsh-Hadamard transform along ``axis``."""
    backend = _check_backend(backend)

    axis = axis % len(array.shape)
    array = backend.cast(array, dtype=array.dtype)
    array = backend.swapaxes(array, axis, -1)

    dim = array.shape[-1]
    if dim & (dim - 1):  # pragma: no cover
        raise_error(
            ValueError, "Walsh-Hadamard transform dimension must be a power of 2."
        )

    block = 1
    while block < dim:
        shape = array.shape[:-1] + (dim // (2 * block), 2, block)
        array = backend.reshape(array, shape)
        even = array[..., 0, :]
        odd = array[..., 1, :]
        array = backend.concatenate(
            (
                backend.expand_dims(even + odd, -2),
                backend.expand_dims(even - odd, -2),
            ),
            axis=-2,
        )
        array = backend.reshape(array, array.shape[:-3] + (dim,))
        block *= 2

    return backend.swapaxes(array, axis, -1)


def _slice_axis(array: ArrayLike, axis: int, index: int) -> ArrayLike:
    """Return ``array`` sliced at ``index`` along ``axis``."""
    slices = [slice(None)] * len(array.shape)
    slices[axis] = index

    return array[tuple(slices)]


def _reorder_axis(
    array: ArrayLike,
    axis: int,
    permutation: list[int] | tuple[int, ...] | ArrayLike,
    backend: Backend | None = None,
) -> ArrayLike:
    """Reorder one axis using only scalar slicing and concatenation."""
    backend = _check_backend(backend)

    axis = axis % len(array.shape)
    reordered = []
    for index in permutation:
        reordered.append(backend.expand_dims(_slice_axis(array, axis, index), axis))

    return backend.concatenate(reordered, axis=axis)


def _xor_pair_axis(
    array: ArrayLike, axis: int, backend: Backend | None = None
) -> ArrayLike:
    r"""Apply ``(r, q) -> (r \oplus q, q)`` to two adjacent binary axes."""
    backend = _check_backend(backend)

    axis = axis % len(array.shape)
    next_axis = axis + 1

    row_0_col_0 = _slice_axis(_slice_axis(array, axis, 0), next_axis - 1, 0)
    row_0_col_1 = _slice_axis(_slice_axis(array, axis, 0), next_axis - 1, 1)
    row_1_col_0 = _slice_axis(_slice_axis(array, axis, 1), next_axis - 1, 0)
    row_1_col_1 = _slice_axis(_slice_axis(array, axis, 1), next_axis - 1, 1)

    row_0 = backend.concatenate(
        (
            backend.expand_dims(row_0_col_0, next_axis - 1),
            backend.expand_dims(row_1_col_1, next_axis - 1),
        ),
        axis=next_axis - 1,
    )
    row_1 = backend.concatenate(
        (
            backend.expand_dims(row_1_col_0, next_axis - 1),
            backend.expand_dims(row_0_col_1, next_axis - 1),
        ),
        axis=next_axis - 1,
    )

    return backend.concatenate(
        (backend.expand_dims(row_0, axis), backend.expand_dims(row_1, axis)),
        axis=axis,
    )


def _xor_transform(array: ArrayLike, backend: Backend | None = None) -> ArrayLike:
    """Apply the self-inverse XOR permutation along the last two axes."""
    backend = _check_backend(backend)

    dim = array.shape[-1]
    nqubits = int(math.log2(dim))
    batch_shape = array.shape[:-2]

    array = backend.reshape(array, batch_shape + (2,) * (2 * nqubits))
    offset = len(batch_shape)
    axes = tuple(range(offset)) + tuple(
        offset + index for pair in range(nqubits) for index in (pair, nqubits + pair)
    )
    array = backend.transpose(array, axes)

    for qubit in range(nqubits):
        array = _xor_pair_axis(array, offset + 2 * qubit, backend=backend)

    inverse_axes = [0] * len(axes)
    for index, axis in enumerate(axes):
        inverse_axes[axis] = index
    array = backend.transpose(array, tuple(inverse_axes))

    return backend.reshape(array, batch_shape + (dim, dim))


def _phase_matrix(
    dim: int, sign: int = -1, backend: Backend | None = None
) -> ArrayLike:
    """Return ``(sign * i) ** |r & s|`` for all pairs ``(r, s)``."""
    backend = _check_backend(backend)

    phase = backend.cast([[1.0, 1.0], [1.0, sign * 1.0j]], dtype=backend.complex128)
    phase = reduce(backend.kron, [phase] * int(math.log2(dim)))

    return phase


def _check_pauli_order(pauli_order: str) -> None:
    """Validate the single-qubit Pauli order."""
    if len(pauli_order) != 4 or set(pauli_order) != {
        "I",
        "X",
        "Y",
        "Z",
    }:
        raise_error(
            ValueError,
            f"pauli_order has to contain 4 symbols: I, X, Y, Z. Got {pauli_order} instead.",
        )


def _symplectic_coefficients_to_pauli_order(
    coefficients: ArrayLike,
    nqubits: int,
    dim: int,
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike:
    """Vectorize coefficients ``alpha[r, s]`` according to ``pauli_order``."""
    backend = _check_backend(backend)
    _check_pauli_order(pauli_order)

    batch_shape = coefficients.shape[:-2]
    coefficients = backend.reshape(coefficients, batch_shape + (2,) * (2 * nqubits))
    offset = len(batch_shape)
    axes = tuple(range(offset)) + tuple(
        offset + index for pair in range(nqubits) for index in (pair, nqubits + pair)
    )
    coefficients = backend.transpose(coefficients, axes)
    coefficients = backend.reshape(coefficients, batch_shape + (4,) * nqubits)

    canonical_order = "IZXY"
    permutation = tuple(canonical_order.index(pauli) for pauli in pauli_order)
    for qubit in range(nqubits):
        coefficients = _reorder_axis(
            coefficients, offset + qubit, permutation, backend=backend
        )

    return backend.reshape(coefficients, batch_shape + (dim**2,))


def _pauli_order_to_symplectic_coefficients(
    vectors: ArrayLike,
    nqubits: int,
    dim: int,
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike:
    """Convert Pauli-ordered vectors to coefficients ``alpha[r, s]``."""
    backend = _check_backend(backend)
    _check_pauli_order(pauli_order)

    batch_shape = vectors.shape[:-1]
    coefficients = backend.reshape(vectors, batch_shape + (4,) * nqubits)

    offset = len(batch_shape)
    canonical_order = "IZXY"
    permutation = tuple(pauli_order.index(pauli) for pauli in canonical_order)
    for qubit in range(nqubits):
        coefficients = _reorder_axis(
            coefficients, offset + qubit, permutation, backend=backend
        )

    coefficients = backend.reshape(coefficients, batch_shape + (2,) * (2 * nqubits))
    axes = (
        tuple(range(offset))
        + tuple(offset + 2 * qubit for qubit in range(nqubits))
        + tuple(offset + 2 * qubit + 1 for qubit in range(nqubits))
    )
    coefficients = backend.transpose(coefficients, axes)

    return backend.reshape(coefficients, batch_shape + (dim, dim))


def _operator_to_pauli_coefficients_fht(
    operators: ArrayLike, dim: int, backend: Backend | None = None
) -> ArrayLike:
    """Return Pauli decomposition coefficients for a batch of operators."""
    backend = _check_backend(backend)

    coefficients = _xor_transform(operators, backend=backend)
    coefficients = _fast_walsh_hadamard_transform(
        coefficients, axis=-1, backend=backend
    )
    coefficients = coefficients * _phase_matrix(dim, sign=-1, backend=backend) / dim

    return coefficients


def _pauli_coefficients_to_operator_fht(
    coefficients: ArrayLike, dim: int, backend: Backend | None = None
) -> ArrayLike:
    """Reconstruct a batch of operators from Pauli decomposition coefficients."""
    backend = _check_backend(backend)

    operators = coefficients * _phase_matrix(dim, sign=1, backend=backend) * dim
    operators = (
        _fast_walsh_hadamard_transform(operators, axis=-1, backend=backend) / dim
    )
    operators = _xor_transform(operators, backend=backend)

    return operators


def _operator_to_pauli_vectors_fht(
    operators: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike:
    """Convert a batch of operators to vectorized Pauli-basis coordinates."""
    backend = _check_backend(backend)

    coefficients = _operator_to_pauli_coefficients_fht(operators, dim, backend=backend)
    normalization = _pauli_basis_normalization(nqubits) if normalize else 1.0
    coefficients = _symplectic_coefficients_to_pauli_order(
        coefficients,
        nqubits=nqubits,
        dim=dim,
        pauli_order=pauli_order,
        backend=backend,
    )

    return coefficients * dim / normalization


def _pauli_vectors_to_operator_fht(
    vectors: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike:
    """Convert vectorized Pauli-basis coordinates to computational operators."""
    backend = _check_backend(backend)

    normalization = _pauli_basis_normalization(nqubits) if normalize else 1.0
    coefficients = _pauli_order_to_symplectic_coefficients(
        vectors / normalization,
        nqubits=nqubits,
        dim=dim,
        pauli_order=pauli_order,
        backend=backend,
    )

    return _pauli_coefficients_to_operator_fht(coefficients, dim, backend=backend)


def _to_pauli_liouville_fht(
    channel: ArrayLike,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike:
    """Converts ``channel`` to Pauli-Liouville representation using FHT."""
    from qibo.quantum_info.superoperator_transformations import to_liouville

    backend = _check_backend(backend)

    super_op = to_liouville(channel, order=order, backend=backend)
    dim, nqubits = _check_pauli_superoperator_shape(super_op, "super_op")

    return _liouville_to_pauli_fht(
        super_op,
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )


def _liouville_to_pauli_fht(
    super_op: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike:
    """Converts Liouville representation to Pauli-Liouville using FHT."""
    from qibo.quantum_info.superoperator_transformations import unvectorization

    backend = _check_backend(backend)

    columns = unvectorization(backend.transpose(super_op), order=order, backend=backend)
    columns = _operator_to_pauli_vectors_fht(
        columns,
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    super_op = backend.transpose(columns)

    rows = _operator_to_pauli_vectors_fht(
        unvectorization(backend.conj(super_op), order=order, backend=backend),
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    return backend.conj(rows)


def _pauli_to_liouville_fht(
    pauli_op: ArrayLike,
    nqubits: int,
    dim: int,
    normalize: bool = False,
    order: str = "row",
    pauli_order: str = "IXYZ",
    backend: Backend | None = None,
) -> ArrayLike:
    """Converts Pauli-Liouville representation to Liouville using FHT."""
    from qibo.quantum_info.superoperator_transformations import vectorization

    backend = _check_backend(backend)

    columns = _pauli_vectors_to_operator_fht(
        backend.transpose(pauli_op),
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    columns = vectorization(columns, order=order, backend=backend)
    super_op = backend.transpose(columns)

    rows = _pauli_vectors_to_operator_fht(
        backend.conj(super_op),
        nqubits=nqubits,
        dim=dim,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    rows = vectorization(rows, order=order, backend=backend)

    return backend.conj(rows)


def _reshuffling(
    super_op: ArrayLike, order: str = "row", backend: Backend | None = None
) -> ArrayLike:
    """Reshuffling operation used to convert Lioville representation
    of quantum channels to their Choi representation (and vice-versa).

    For an operator :math:`A` with dimensions :math:`d^{2} \\times d^{2}`,
    the reshuffling operation consists of reshaping :math:`A` as a
    4-dimensional tensor, swapping two axes, and reshaping back to a
    :math:`d^{2} \\times d^{2}` matrix.

    If ``order="row"``, then:

    .. math::
        A_{\\alpha\\beta, \\, \\gamma\\delta} \\mapsto A_{\\alpha, \\, \\beta, \\,
            \\gamma, \\, \\delta} \\mapsto A_{\\alpha, \\, \\gamma, \\, \\beta, \\, \\delta}
            \\mapsto A_{\\alpha\\gamma, \\, \\beta\\delta}

    If ``order="column"``, then:

    .. math::
        A_{\\alpha\\beta, \\, \\gamma\\delta} \\mapsto A_{\\alpha, \\, \\beta, \\,
            \\gamma, \\, \\delta} \\mapsto A_{\\delta, \\, \\beta, \\, \\gamma, \\, \\alpha}
            \\mapsto A_{\\delta\\beta, \\, \\gamma\\alpha}

    Args:
        super_op (ArrayLike): Liouville (Choi) representation of a
            quantum channel.
        order (str, optional): If ``"row"``, reshuffling is performed
            with respect to row-wise vectorization. If ``"column"``,
            reshuffling is performed with respect to column-wise
            vectorization. Defaults to ``"row"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        ArrayLike: Choi (Liouville) representation of the quantum channel.
    """
    if order not in ("row", "column"):
        raise_error(
            ValueError,
            f"Unsupported {order} order, please pick one in ('row', 'column').",
        )
    backend = _check_backend(backend)

    dim = math.sqrt(super_op.shape[0])

    if (
        super_op.shape[0] != super_op.shape[1]
        or dim % 1 != 0
        or math.log2(int(dim)) % 1 != 0
    ):
        raise_error(
            ValueError,
            f"`super_op` must be of shape (4^n, 4^n), but it is {super_op.shape}",
        )

    axes = [1, 2] if order == "row" else [0, 3]
    return backend.qinfo._reshuffling(super_op, *axes)


def _set_gate_and_target_qubits(
    kraus_ops: list | Channel, backend: Backend | None = None
) -> tuple[tuple, tuple]:
    """Returns Kraus operators as a set of gates acting on
    their respective ``target qubits``.

    Args:
        kraus_ops (list or :class:`qibo.gates.channels.Channel`): List of Kraus
            operators as pairs ``(qubits, Ak)`` where ``qubits`` refers the
            qubit ids that :math:`A_k` acts on and :math:`A_k` is the
            corresponding matrix as a ``ArrayLike``. If a channel is given,
            its operators are rescaled by the square root of their coefficients,
            and, if the coefficients sum to less than one, the remaining
            probability is assigned to the identity.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend
            to be used in the execution. If ``None``, it uses
            the current backend. Defaults to ``None``.

    Returns:
        (tuple, tuple): gates and their respective target qubits.
    """
    backend = _check_backend(backend)

    if isinstance(kraus_ops, Channel):
        channel = kraus_ops
        coefficients = list(channel.coefficients)
        remainder = 1 - sum(coefficients)

        kraus_ops = [
            (gate.qubits, backend.sqrt(coefficient) * gate.matrix(backend))
            for coefficient, gate in zip(coefficients, channel.gates)
        ]
        if remainder > PRECISION_TOL:
            kraus_ops.append(
                (
                    channel.target_qubits,
                    backend.sqrt(remainder)
                    * backend.identity(
                        2 ** len(channel.target_qubits), dtype=backend.complex128
                    ),
                )
            )

    if len(kraus_ops) == 0:
        raise_error(ValueError, "``kraus_ops`` must contain at least one operator.")

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
                    f"Invalid Kraus operator shape {shape} for "
                    + f"acting on {len(qubits)} qubits.",
                )
            qubitset.update(qubits)
            gates.append(Unitary(matrix, *list(qubits)))
        gates = tuple(gates)
        target_qubits = tuple(sorted(qubitset))

    return gates, target_qubits
