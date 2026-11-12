from itertools import pairwise

from qibo import gates


def _pauli_rotation(pauli: str, angle: float) -> list[gates.Gate]:
    """Creates the gates of the rotation :math:`e^{-i \\theta P / 2}` of a Pauli string :math:`P`.

    The Pauli matrices of the string are rotated to :math:`Z` with Clifford gates, then a ladder
    of CNOT gates writes the parity of the qubits in the last of them, which has a
    :class:`qibo.gates.RZ` gate. The identity is a :class:`qibo.gates.GlobalPhase` gate.

    Args:
        pauli (str): Pauli string, with one letter from ``"IXYZ"`` for each qubit. The :math:`j`-th
            letter is the Pauli matrix applied to the :math:`j`-th qubit.
        angle (float): angle :math:`\\theta` of the rotation.

    Returns:
        list: gates of the rotation.
    """
    targets = [qubit for qubit, letter in enumerate(pauli) if letter != "I"]

    # the identity is a global phase
    if len(targets) == 0:
        return [gates.GlobalPhase(-angle / 2)]

    # basis change that maps the Pauli string to Z in each of the targets, and a ladder
    # of CNOTs that writes its parity in the last target
    before, after = [], []
    for qubit in targets:
        if pauli[qubit] == "X":
            before.append(gates.H(qubit))
            after.append(gates.H(qubit))
        elif pauli[qubit] == "Y":
            before.extend([gates.SDG(qubit), gates.H(qubit)])
            after.extend([gates.H(qubit), gates.S(qubit)])
    pairs = list(pairwise(targets))

    rotation = []
    rotation.extend(before)
    rotation.extend(gates.CNOT(*pair) for pair in pairs)
    rotation.append(gates.RZ(targets[-1], theta=angle))
    rotation.extend(gates.CNOT(*pair) for pair in reversed(pairs))
    rotation.extend(after)

    return rotation
