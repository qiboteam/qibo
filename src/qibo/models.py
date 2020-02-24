from qibo.config import Circuit


def QFTCircuit(nqubits: int, with_swaps: bool = True) -> Circuit:
    """Creates a circuit that implements the Quantum Fourier Transform.

    Args:
        nqubits: Number of qubits in the circuit.
        with_swaps: Use SWAP gates at the end of the circuit so that the final
            qubit ordering agrees with the initial state.

    Returns:
        A qibo.models.Circuit that implements the Quantum Fourier Transform.
    """
    from qibo.config import gates

    circuit = Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1))
        m = 2
        for i2 in range(i1 + 1, nqubits):
            theta = 1.0 / 2 ** (m - 1)
            circuit.add(gates.CRZ(i2, i1, theta))
            m += 1

    if with_swaps:
        for i in range(nqubits // 2):
            circuit.add(gates.SWAP(i, nqubits - i - 1))

    return circuit


def QFTGates(state, with_swaps: bool = True):
    """Implements the Quantum Fourier Transform using gates.

    Same as `QFTCircuit` but uses simple gates to transform the state instead
    of constructing a `models.Circuit` container.

    Args:
        state: Initial state vector on which QFT should act on.
            Should have shape nqubits * (2,).
        with_swaps: Same as in `QFTCircuit`.

    Returns:
        Transformed state vector.
    """
    import tensorflow as tf
    from qibo.config import gates

    nqubits = len(tuple(state.shape))
    for i1 in range(nqubits):
        state = gates.H(i1)(state)
        m = 2
        for i2 in range(i1 + 1, nqubits):
            theta = 1.0 / 2 ** (m - 1)
            state = gates.CRZ(i2, i1, theta)(state)
            m += 1

    if with_swaps:
        for i in range(nqubits // 2):
            state = gates.SWAP(i, nqubits - i - 1)(state)

    return tf.reshape(state, (2**nqubits,))