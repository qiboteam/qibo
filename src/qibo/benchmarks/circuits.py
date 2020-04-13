from qibo import models, gates


def SupremacyLikeCircuit(nqubits, nlayers):
    one_qubit_gates = ["RX", "RY", "RZ"]
    circuit = models.Circuit(nqubits)
    d = 1
    for l in range(nlayers):
        for i in range(nqubits):
            gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
            circuit.add(gate(i, 0.5))
        for i in range(nqubits):
            circuit.add(gates.CRZ(i, (i + d) % nqubits, 1.0/6.0))
        d += 1
        if d > nqubits - 1:
            d = 1
    for i in range(nqubits):
        gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
        circuit.add(gate(i, 0.5))
        circuit.add(gates.M(i))
    return circuit


circuits = {"supremacy": SupremacyLikeCircuit,
            "qft": models.QFT}
