import numpy as np
from qibo import models, gates


def SupremacyLikeCircuit(nqubits, backend, nlayers):
    one_qubit_gates = ["RX", "RY", "RZ"]
    circuit = models.Circuit(nqubits)
    d = 1
    for l in range(nlayers):
        for i in range(nqubits):
            gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
            circuit.add(gate(i, np.pi / 2.0).with_backend(backend))
        for i in range(nqubits):
            circuit.add(gates.CZPow(i, (i + d) % nqubits, np.pi / 6.0).with_backend(backend))
        d += 1
        if d > nqubits - 1:
            d = 1
    for i in range(nqubits):
        gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
        circuit.add(gate(i, np.pi / 2.0).with_backend(backend))
        circuit.add(gates.M(i).with_backend(backend))
    return circuit


def PrepareGHZ(nqubits):
    circuit = models.Circuit(nqubits)
    circuit.add(gates.H(0).with_backend(backend))
    for i in range(nqubits - 1):
        circuit.add(gates.CNOT(i, i + 1).with_backend(backend))
    return circuit


def QFT(nqubits, backend):
    circuit = models.Circuit(nqubits)
    for i1 in range(nqubits):
        circuit.add(gates.H(i1).with_backend(backend))
        m = 2
        for i2 in range(i1 + 1, nqubits):
            theta = np.pi / 2 ** (m - 1)
            circuit.add(gates.CZPow(i2, i1, theta).with_backend(backend))
            m += 1

    for i in range(nqubits // 2):
        circuit.add(gates.SWAP(i, nqubits - i - 1).with_backend(backend))

    return circuit


def Hadamards(nqubits, backend):
    circuit = models.Circuit(nqubits)
    for i in range(nqubits):
        circuit.add(gates.H(i).with_backend(backend))
    return circuit


circuits = {"supremacy": SupremacyLikeCircuit,
            "qft": QFT,
            "ghz": PrepareGHZ,
            "hadamards": Hadamards}
