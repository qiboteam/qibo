# -*- coding: utf-8 -*-
import numpy as np
from qibo import models, gates


def VariationalCircuit(nqubits, nlayers=1, theta_values=None):
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


def OneQubitGate(nqubits, gate_type="H", params={}, nlayers=1):
    gate = lambda q: getattr(gates, gate_type)(q, **params)
    for _ in range(nlayers):
        for i in range(nqubits):
            yield gate(i)


def TwoQubitGate(nqubits, gate_type="H", params={}, nlayers=1):
    gate = lambda q: getattr(gates, gate_type)(q, q + 1, **params)
    for _ in range(nlayers):
        for i in range(0, nqubits - 1, 2):
            yield gate(i)
        for i in range(1, nqubits - 1, 2):
            yield gate(i)


def ToffoliGate(nqubits, nlayers=1):
    for _ in range(nlayers):
        for i in range(0, nqubits - 2, 3):
            yield gates.TOFFOLI(i, i + 1, i + 2)
        for i in range(1, nqubits - 2, 3):
            yield gates.TOFFOLI(i, i + 1, i + 2)
        for i in range(2, nqubits - 2, 3):
            yield gates.TOFFOLI(i, i + 1, i + 2)


def PrepareGHZ(nqubits):
    yield gates.H(0)
    for i in range(nqubits - 1):
        yield gates.CNOT(i, i + 1)


_CIRCUITS = {
    "variational": VariationalCircuit,
    "one-qubit-gate": OneQubitGate,
    "two-qubit-gate": TwoQubitGate,
    "toffoli-gate": ToffoliGate,
    "ghz": PrepareGHZ,
}


def CircuitFactory(nqubits, circuit_name, accelerators=None, **kwargs):
    if circuit_name == "qft":
        circuit = models.QFT(nqubits, accelerators=accelerators)
    else:
        if circuit_name not in _CIRCUITS:
            raise KeyError("Unknown benchmark circuit type {}." "".format(circuit_name))
        circuit = models.Circuit(nqubits, accelerators=accelerators)
        circuit.add(_CIRCUITS.get(circuit_name)(nqubits, **kwargs))
    return circuit
