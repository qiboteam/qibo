"""
Gates use ``einsum`` to apply gates to state vectors. The einsum string that
specifies the contraction indices is created using the following methods and
used by :meth:`qibo.backends.numpy.NumpyEngine.apply_gate`.
"""
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
            order.append(i)
        loop_start = control + 1
        for i, t in enumerate(gate.target_qubits):
            if t > control:
                targets[i] -= 1
    for i in range(loop_start, nqubits):
        order.append(i)
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
