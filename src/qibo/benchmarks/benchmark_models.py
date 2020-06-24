import numpy as np
from qibo import models, gates
from typing import Dict, Optional


def SupremacyLikeCircuit(nqubits: int, nlayers: int):
    one_qubit_gates = ["RX", "RY", "RZ"]
    d = 1
    for l in range(nlayers):
        for i in range(nqubits):
            gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
            yield gate(i, np.pi / 2.0)
        for i in range(nqubits):
            yield gates.CZPow(i, (i + d) % nqubits, np.pi / 6.0)
        d += 1
        if d > nqubits - 1:
            d = 1
    for i in range(nqubits):
        gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
        yield gate(i, np.pi / 2.0)
        yield gates.M(i)


def PrepareGHZ(nqubits: int):
    yield gates.H(0)
    for i in range(nqubits - 1):
        yield gates.CNOT(i, i + 1)


def VariationalCircuit(nqubits: int, nlayers: int = 1,
                       theta_values: Optional[np.ndarray] = None):
    if theta_values is None:
        theta = iter(2 * np.pi * np.random.random(nlayers * 2 * nqubits))
    else:
        theta = iter(theta_values)

    for l in range(nlayers):
        for i in range(nqubits):
            yield gates.RY(i, next(theta))
        for i in range(0, nqubits - 1, 2):
            yield gates.CZ(i, i + 1)
        for i in range(nqubits):
            yield gates.RY(i, next(theta))
        for i in range(1, nqubits - 2, 2):
          yield gates.CZ(i, i + 1)
        yield gates.CZ(0, nqubits - 1)


def OptimizedVariationalCircuit(nqubits: int, nlayers: int = 1,
                                theta_values: Optional[np.ndarray] = None):
    if theta_values is None:
        theta = iter(2 * np.pi * np.random.random(nlayers * 2 * nqubits))
    else:
        theta = iter(theta_values)

    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    for l in range(nlayers):
        thetas1 = {i: next(theta) for i in range(nqubits)}
        thetas2 = {i: next(theta) for i in range(nqubits)}
        yield gates.VariationalLayer(pairs, gates.RY, gates.CZ, thetas1, thetas2)
        for i in range(1, nqubits - 2, 2):
            yield gates.CZ(i, i + 1)
        yield gates.CZ(0, nqubits - 1)


def OneQubitGate(nqubits: int, gate_type: str = "H",
                 params: Dict[str, float] = {}, nlayers: int = 1):
    gate = lambda q: getattr(gates, gate_type)(q, **params)
    for _ in range(nlayers):
        for i in range(nqubits):
            yield gate(i)


def TwoQubitGate(nqubits: int, gate_type: str = "H",
                 params: Dict[str, float] = {}, nlayers: int = 1):
    gate = lambda q: getattr(gates, gate_type)(q, q + 1, **params)
    for _ in range(nlayers):
        for i in range(0, nqubits - 1, 2):
            yield gate(i)
        for i in range(1, nqubits - 1, 2):
            yield gate(i)


def ToffoliGate(nqubits: int, nlayers: int = 1):
    for _ in range(nlayers):
        for i in range(0, nqubits - 2, 3):
            yield gates.TOFFOLI(i, i + 1, i + 2)
        for i in range(1, nqubits - 2, 3):
            yield gates.TOFFOLI(i, i + 1, i + 2)
        for i in range(2, nqubits - 2, 3):
            yield gates.TOFFOLI(i, i + 1, i + 2)


_CIRCUITS = {"supremacy": SupremacyLikeCircuit,
             "ghz": PrepareGHZ,
             "variational": VariationalCircuit,
             "opt-variational": OptimizedVariationalCircuit,
             "one-qubit-gate": OneQubitGate,
             "two-qubit-gate": TwoQubitGate,
             "toffoli-gate": ToffoliGate}


def CircuitFactory(nqubits: int,
                   circuit_type: str,
                   accelerators: Dict[str, int] = None,
                   memory_device: str = "/CPU:0",
                   **kwargs):
    if circuit_type == "qft":
        circuit = models.QFT(nqubits, accelerators=accelerators,
                             memory_device=memory_device)
    else:
        if circuit_type not in _CIRCUITS:
            raise TypeError("Unknown benchmark circuit type {}."
                            "".format(circuit_type))
        circuit = models.Circuit(nqubits, accelerators=accelerators,
                                 memory_device=memory_device)
        circuit.add(_CIRCUITS[circuit_type](nqubits, **kwargs))
    return circuit
