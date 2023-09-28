from typing import Dict, Tuple


def binary2spin(
    linear: Dict[int, float],
    quadratic: Dict[Tuple[int, int], float],
    offset: float = 0,
):
    """Convert binary model to spin model

        Please remember to put a negative sign to h and J after using this
        function , if you are going to form a mamiltonian from the spin
        model. Hamiltonians usually have a leading negative sign in them,
        but QUBOs don't.

    Args:
        linear (dict): linear term of the binary model
        quadratic (dict): quadratic term of the binary model
        offset (float): offset of the binary model

    Returns:
        h (dict): bias of the spin model
        J (dict): interaction of the spin model
        offset (float): offset of the spin model
    """

    h = {x: 0.5 * w for x, w in linear.items()}

    J = []
    for (x, y), w in quadratic.items():
        J.append(((x, y), 0.25 * w))
        h[x] += 0.25 * w
        h[y] += 0.25 * w
    J = dict(J)

    offset += 0.5 * sum(linear.values())
    offset += 0.25 * sum(quadratic.values())

    return h, J, offset


def spin2binary(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    offset: float = 0,
):
    """Convert spin model to binary model

        Please remember to put a negative sign to h and J before using this
        function if you extract them from a hamiltonian.  Hamiltonians
        usually have a leading negative sign in them, but QUBOs don't.

    Args:
        h (dict): bias of the spin model
        J (dict): interaction of the spin model
        offset (float): offset of the spin model

    Returns:
        linear (dict): linear term of the binary model
        quadratic (dict): quadratic term of the binary model
        offset (float): offset of the binary model
    """

    linear = {s: 2.0 * bias for s, bias in h.items()}

    quadratic = []
    for (s, t), bias in J.items():
        quadratic.append(((s, t), 4.0 * bias))
        linear[s] -= 2.0 * bias
        linear[t] -= 2.0 * bias
    quadratic = dict(quadratic)

    offset -= sum(linear.values())
    offset += sum(quadratic.values())

    return linear, quadratic, offset


def spin2QiboHamiltonian(
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    dense: bool = True,
):
    """Convert spin model to qibo Hamiltonian

        Mixer is not included.

        Please remember to put a negative sign to h and J if you get h and J
        from binary2spin.  Hamiltonians usually have a leading negative sign
        in them, but QUBOs don't.

    .. math::
        H = - \\sum_{i=0}^N \\sum _{j=0}^N J_{ij} Z_i Z_j - \\sum_{i=0}^N h_i Z_i.

    Args:
        h (dict): bias of the spin model
        J (dict): interaction of the spin model
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.

    Returns:
        linear (dict): linear term of the binary model
        quadratic (dict): quadratic term of the binary model
        offset (float): offset of the binary model
    """

    from qibo import hamiltonians
    from qibo.symbols import Z

    symbolic_ham = sum(Z(k, commutative=True) * v for k, v in h.items())
    symbolic_ham += sum(
        Z(k0, commutative=True) * Z(k1, commutative=True) * v
        for (k0, k1), v in J.items()
    )
    symbolic_ham = -symbolic_ham

    ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)

    if dense:
        return ham.dense
    else:
        return ham
