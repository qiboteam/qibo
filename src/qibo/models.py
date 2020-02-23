from qibo.config import Circuit
from qibo.config import gates


def QFTCircuit(nqubits: int, with_swaps: bool = True) -> Circuit:
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