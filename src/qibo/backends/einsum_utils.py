"""
Gates use ``einsum`` to apply gates to state vectors. The einsum string that
specifies the contraction indices is created using the following methods and
used by :meth:`qibo.backends.numpy.NumpyEngine.apply_gate`.
"""

from functools import cache

from qibo.config import EINSUM_CHARS, raise_error


def prepare_strings(qubits, nqubits):
    if nqubits + len(qubits) > len(EINSUM_CHARS):  # pragma: no cover
        raise_error(NotImplementedError, "Not enough einsum characters.")

    inp = list(EINSUM_CHARS[:nqubits])
    out = inp[:]
    trans = list(EINSUM_CHARS[nqubits : nqubits + len(qubits)])
    for i, q in enumerate(qubits):
        trans.append(inp[q])
        out[q] = trans[i]

    inp = "".join(inp)
    out = "".join(out)
    trans = "".join(trans)
    rest = EINSUM_CHARS[nqubits + len(qubits) :]
    return inp, out, trans, rest


def apply_gate_string(qubits, nqubits):
    inp, out, trans, _ = prepare_strings(qubits, nqubits)
    return f"{inp},{trans}->{out}"


def apply_gate_density_matrix_string(qubits, nqubits):
    inp, out, trans, rest = prepare_strings(qubits, nqubits)
    if nqubits > len(rest):  # pragma: no cover
        raise_error(NotImplementedError, "Not enough einsum characters.")

    trest = rest[:nqubits]
    left = f"{inp}{trest},{trans}->{out}{trest}"
    right = f"{trest}{inp},{trans}->{trest}{out}"
    return left, right


def apply_gate_density_matrix_controlled_string(qubits, nqubits):
    inp, out, trans, rest = prepare_strings(qubits, nqubits)
    if nqubits > len(rest):  # pragma: no cover
        raise_error(NotImplementedError, "Not enough einsum characters.")

    trest, c = rest[:nqubits], rest[nqubits]
    left = f"{c}{inp}{trest},{trans}->{c}{out}{trest}"
    right = f"{c}{trest}{inp},{trans}->{c}{trest}{out}"
    return left, right


def control_order(gate, nqubits):
    loop_start = 0
    order = list(gate.control_qubits)
    targets = list(gate.target_qubits)
    for control in gate.control_qubits:
        for i in range(loop_start, control):
            order.append(i)  # noqa: PERF402
        loop_start = control + 1
        for i, t in enumerate(gate.target_qubits):
            if t > control:
                targets[i] -= 1
    for i in range(loop_start, nqubits):
        order.append(i)  # noqa: PERF402
    return order, targets


def control_order_density_matrix(gate, nqubits):
    ncontrol = len(gate.control_qubits)
    order, targets = control_order(gate, nqubits)
    additional_order = [x + len(order) for x in order]
    order_dm = (
        order[:ncontrol]
        + list(additional_order[:ncontrol])
        + order[ncontrol:]
        + list(additional_order[ncontrol:])
    )
    return order_dm, targets


def reverse_order(order):
    rorder = len(order) * [0]
    for i, r in enumerate(order):
        rorder[r] = i
    return rorder


@cache
def permutations(qubits: tuple[int, ...], nqubits: int) -> tuple[list[int], list[int]]:
    """Compute the permutations of the indices needed for gate application
    through unfolding and matrix product. The first permutation moves the
    contracted axes, i.e. the interested qubits, to the front, the second
    premutation reverts the first one.

    Args:
        qubits (Tuple[int, ...]): the interested qubits.
        nqubits (int): the total number of qubits of the system.
    Returns:
        (Tuple[List[int], List[int]]) the two permuations of the indices.
    """
    fwd_perm = list(qubits) + [q for q in range(nqubits) if q not in qubits]
    inv_perm = zip(list(range(nqubits)), fwd_perm)
    inv_perm, _ = list(zip(*sorted(inv_perm, key=lambda x: x[1])))
    return fwd_perm, inv_perm


@cache
def permutations_density_matrix(
    controls: tuple[int, ...], targets: tuple[int, ...], nqubits: int
) -> tuple[list[int], list[int]]:
    """Compute the permutations of the indices needed for (controlled) gate
    application on density matrices through unfolding and matrix product.

    The row axes are ordered as ``(controls, targets, rest)`` and the column axes
    as ``(controls, rest, targets)``, so that after reshaping the target rows can
    be multiplied from the left and the target columns from the right with plain
    two-dimensional matrix products.

    Args:
        controls (Tuple[int, ...]): the control qubits (may be empty).
        targets (Tuple[int, ...]): the target qubits.
        nqubits (int): the total number of qubits of the system.

    Returns:
        (Tuple[List[int], List[int]]) the forward permutation and its inverse.
    """
    rest = [q for q in range(nqubits) if q not in controls and q not in targets]
    fwd_perm = list(controls) + list(targets) + rest
    fwd_perm += [nqubits + q for q in list(controls) + rest + list(targets)]
    return fwd_perm, reverse_order(fwd_perm)
