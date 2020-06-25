import numpy as np
from qibo import models
from qibo.tensorflow import gates as native_gates
from qibo.tensorflow import cgates as custom_gates
from typing import Dict, Optional


def get_gates(backend: str):
    if backend == "Custom":
        return custom_gates
    if backend in {"DefaultEinsum", "MatmulEinsum"}:
        return native_gates
    raise ValueError("Unknown backend {}.".format(backend))


def SupremacyLikeCircuit(nqubits: int, backend: str, nlayers: int) -> models.Circuit:
    gates = get_gates(backend)
    one_qubit_gates = ["RX", "RY", "RZ"]
    circuit = models.Circuit(nqubits)
    d = 1
    for l in range(nlayers):
        for i in range(nqubits):
            gate = getattr(gates, one_qubit_gates[int(np.random.randint(0, len(one_qubit_gates)))])
            circuit.add(gate(i, np.pi / 2.0).with_backend(backend))
        for i in range(nqubits):
            circuit.add(gates.CZPow(i, (i + d) % nqubits, np.pi / 6.0).with_backend(backend))
        d += 1
        if d > nqubits - 1:
            d = 1
    for i in range(nqubits):
        gate = getattr(gates, one_qubit_gates[int(np.random.randint(0, len(one_qubit_gates)))])
        circuit.add(gate(i, np.pi / 2.0).with_backend(backend))
        circuit.add(gates.M(i).with_backend(backend))
    return circuit


def PrepareGHZ(nqubits: int, backend: str) -> models.Circuit:
    gates = get_gates(backend)
    circuit = models.Circuit(nqubits)
    circuit.add(gates.H(0).with_backend(backend))
    for i in range(nqubits - 1):
        circuit.add(gates.CNOT(i, i + 1).with_backend(backend))
    return circuit


def QFT(nqubits: int, backend: str) -> models.Circuit:
    gates = get_gates(backend)
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


def VariationalCircuit(nqubits: int, backend: str, nlayers: int = 1,
                       theta_values: Optional[np.ndarray] = None
                       ) -> models.Circuit:
    gates = get_gates(backend)
    if theta_values is None:
        theta = iter(2 * np.pi * np.random.random(nlayers * 2 * nqubits))
    else:
        theta = iter(theta_values)

    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        circuit.add((gates.RY(i, next(theta)).with_backend(backend)
                     for i in range(nqubits)))
        circuit.add((gates.CZ(i, i + 1).with_backend(backend)
                     for i in range(0, nqubits - 1, 2)))
        circuit.add((gates.RY(i, next(theta)).with_backend(backend)
                     for i in range(nqubits)))
        circuit.add((gates.CZ(i, i + 1).with_backend(backend)
                     for i in range(1, nqubits - 2, 2)))
        circuit.add(gates.CZ(0, nqubits - 1).with_backend(backend))
    return circuit


def OptimizedVariationalCircuit(nqubits: int, backend: str, nlayers: int = 1,
                                theta_values: Optional[np.ndarray] = None
                                ) -> models.Circuit:
    gates = get_gates(backend)
    if theta_values is None:
        theta = iter(2 * np.pi * np.random.random(nlayers * 2 * nqubits))
    else:
        theta = iter(theta_values)

    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        thetas1 = {i: next(theta) for i in range(nqubits)}
        thetas2 = {i: next(theta) for i in range(nqubits)}
        circuit.add(gates.VariationalLayer(pairs, gates.RY, gates.CZ,
                                           thetas1, thetas2).with_backend(backend))
        circuit.add((gates.CZ(i, i + 1).with_backend(backend)
                     for i in range(1, nqubits - 2, 2)))
        circuit.add(gates.CZ(0, nqubits - 1).with_backend(backend))
    return circuit


def OneQubitGate(nqubits: int, backend: str, gate_type: str = "H",
                 params: Dict[str, float] = {}, nlayers: int = 1
                 ) -> models.Circuit:
    gates = get_gates(backend)
    circuit = models.Circuit(nqubits)
    gate = lambda q: getattr(gates, gate_type)(q, **params)

    for _ in range(nlayers):
        for i in range(nqubits):
            circuit.add(gate(i).with_backend(backend))
    return circuit


def TwoQubitGate(nqubits: int, backend: str, gate_type: str = "H",
                 params: Dict[str, float] = {}, nlayers: int = 1
                 ) -> models.Circuit:
    gates = get_gates(backend)
    circuit = models.Circuit(nqubits)
    gate = lambda q: getattr(gates, gate_type)(q, **params)

    for _ in range(nlayers):
        for i in range(0, nqubits - 1, 2):
            circuit.add(gate(i).with_backend(backend))
        for i in range(1, nqubits - 1, 2):
            circuit.add(gate(i).with_backend(backend))
    return circuit


def ToffoliGate(nqubits: int, backend: str, nlayers: int = 1) -> models.Circuit:
    gates = get_gates(backend)
    circuit = models.Circuit(nqubits)
    for _ in range(nlayers):
        for i in range(0, nqubits - 2, 3):
            circuit.add(gates.TOFFOLI(i, i + 1, i + 2).with_backend(backend))
        for i in range(1, nqubits - 2, 3):
            circuit.add(gates.TOFFOLI(i, i + 1, i + 2).with_backend(backend))
        for i in range(2, nqubits - 2, 3):
            circuit.add(gates.TOFFOLI(i, i + 1, i + 2).with_backend(backend))
    return circuit


circuits = {"supremacy": SupremacyLikeCircuit,
            "qft": QFT,
            "ghz": PrepareGHZ,
            "variational": VariationalCircuit,
            "opt-variational": OptimizedVariationalCircuit,
            "one-qubit-gate": OneQubitGate,
            "two-qubit-gate": TwoQubitGate,
            "toffoli-gate": ToffoliGate}
