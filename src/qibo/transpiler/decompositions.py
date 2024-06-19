import numpy as np

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.transpiler.unitary_decompositions import (
    two_qubit_decomposition,
    u3_decomposition,
)

backend = NumpyBackend()


class GateDecompositions:
    """Abstract data structure that holds decompositions of gates."""

    def __init__(self):
        self.decompositions = {}

    def add(self, gate, decomposition):
        """Register a decomposition for a gate."""
        self.decompositions[gate] = decomposition

    def count_2q(self, gate):
        """Count the number of two-qubit gates in the decomposition of the given gate."""
        if gate.parameters:
            decomposition = self.decompositions[gate.__class__](gate)
        else:
            decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) > 1))

    def count_1q(self, gate):
        """Count the number of single qubit gates in the decomposition of the given gate."""
        if gate.parameters:
            decomposition = self.decompositions[gate.__class__](gate)
        else:
            decomposition = self.decompositions[gate.__class__]
        return len(tuple(g for g in decomposition if len(g.qubits) == 1))

    def __call__(self, gate):
        """Decompose a gate."""
        decomposition = self.decompositions[gate.__class__]
        if callable(decomposition):
            decomposition = decomposition(gate)
        return [
            g.on_qubits({i: q for i, q in enumerate(gate.qubits)})
            for g in decomposition
        ]


def _u3_to_gpi2(t, p, l):
    """Decompose a U3 gate into GPI2 gates, the decomposition is optimized to use the minimum number of gates..

    Args:
        t (float): theta parameter of U3 gate.
        p (float): phi parameter of U3 gate.
        l (float): lambda parameter of U3 gate.

    Returns:
        decomposition (list): list of native gates that decompose the U3 gate.
    """
    decomposition = []
    if l != 0.0:
        decomposition.append(gates.RZ(0, l))
    decomposition.append(gates.GPI2(0, 0))
    if t != -np.pi:
        decomposition.append(gates.RZ(0, t + np.pi))
    decomposition.append(gates.GPI2(0, 0))
    if p != -np.pi:
        decomposition.append(gates.RZ(0, p + np.pi))
    return decomposition


# Decompose single qubit gates using GPI2 (more efficient on hardware)
gpi2_dec = GateDecompositions()
gpi2_dec.add(gates.H, [gates.Z(0), gates.GPI2(0, np.pi / 2)])
gpi2_dec.add(gates.X, [gates.GPI2(0, np.pi / 2), gates.GPI2(0, np.pi / 2), gates.Z(0)])
gpi2_dec.add(gates.Y, [gates.Z(0), gates.GPI2(0, 0), gates.GPI2(0, 0)])
gpi2_dec.add(gates.Z, [gates.Z(0)])
gpi2_dec.add(gates.S, [gates.RZ(0, np.pi / 2)])
gpi2_dec.add(gates.SDG, [gates.RZ(0, -np.pi / 2)])
gpi2_dec.add(gates.T, [gates.RZ(0, np.pi / 4)])
gpi2_dec.add(gates.TDG, [gates.RZ(0, -np.pi / 4)])
gpi2_dec.add(gates.SX, [gates.GPI2(0, 0)])
gpi2_dec.add(
    gates.RX,
    lambda gate: [
        gates.Z(0),
        gates.GPI2(0, np.pi / 2),
        gates.RZ(0, gate.parameters[0] + np.pi),
        gates.GPI2(0, np.pi / 2),
    ],
)
gpi2_dec.add(
    gates.RY,
    lambda gate: [
        gates.GPI2(0, 0),
        gates.RZ(0, gate.parameters[0] + np.pi),
        gates.GPI2(0, 0),
        gates.Z(0),
    ],
)
gpi2_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
gpi2_dec.add(gates.GPI2, lambda gate: [gates.GPI2(0, gate.parameters[0])])
gpi2_dec.add(gates.U1, lambda gate: [gates.RZ(0, gate.parameters[0])])
gpi2_dec.add(
    gates.U2,
    lambda gate: _u3_to_gpi2(np.pi / 2, gate.parameters[0], gate.parameters[1]),
)
gpi2_dec.add(gates.U3, lambda gate: _u3_to_gpi2(*gate.parameters))
gpi2_dec.add(
    gates.Unitary, lambda gate: _u3_to_gpi2(*u3_decomposition(gate.parameters[0]))
)
gpi2_dec.add(
    gates.FusedGate, lambda gate: _u3_to_gpi2(*u3_decomposition(gate.matrix(backend)))
)

# Decompose single qubit gates using U3
u3_dec = GateDecompositions()
u3_dec.add(gates.H, [gates.U3(0, -np.pi / 2, np.pi, 0)])
u3_dec.add(gates.X, [gates.U3(0, np.pi, 0, np.pi)])
u3_dec.add(gates.Y, [gates.U3(0, np.pi, 0, 0)])
u3_dec.add(gates.Z, [gates.Z(0)])
u3_dec.add(gates.S, [gates.RZ(0, np.pi / 2)])
u3_dec.add(gates.SDG, [gates.RZ(0, -np.pi / 2)])
u3_dec.add(gates.T, [gates.RZ(0, np.pi / 4)])
u3_dec.add(gates.TDG, [gates.RZ(0, -np.pi / 4)])
u3_dec.add(gates.SX, [gates.U3(0, np.pi / 2, -np.pi / 2, np.pi / 2)])
u3_dec.add(
    gates.RX, lambda gate: [gates.U3(0, gate.parameters[0], -np.pi / 2, np.pi / 2)]
)
u3_dec.add(gates.RY, lambda gate: [gates.U3(0, gate.parameters[0], 0, 0)])
u3_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
u3_dec.add(
    gates.GPI2, lambda gate: [gates.U3(0, *u3_decomposition(gate.matrix(backend)))]
)
u3_dec.add(gates.U1, lambda gate: [gates.RZ(0, gate.parameters[0])])
u3_dec.add(
    gates.U2,
    lambda gate: [gates.U3(0, np.pi / 2, gate.parameters[0], gate.parameters[1])],
)
u3_dec.add(
    gates.U3,
    lambda gate: [
        gates.U3(0, gate.parameters[0], gate.parameters[1], gate.parameters[2])
    ],
)
u3_dec.add(
    gates.Unitary,
    lambda gate: [gates.U3(0, *u3_decomposition(gate.parameters[0]))],
)
u3_dec.add(
    gates.FusedGate,
    lambda gate: [gates.U3(0, *u3_decomposition(gate.matrix(backend)))],
)

# register the iSWAP decompositions
iswap_dec = GateDecompositions()
iswap_dec.add(
    gates.CNOT,
    [
        gates.U3(0, 3 * np.pi / 2, np.pi, 0),
        gates.U3(1, np.pi / 2, np.pi, np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, np.pi, np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, np.pi),
        gates.U3(1, np.pi / 2, np.pi, -np.pi / 2),
    ],
)
iswap_dec.add(
    gates.CZ,
    [
        gates.U3(0, -np.pi / 2, np.pi, 0),
        gates.U3(1, -np.pi / 2, np.pi, 0),
        gates.U3(1, np.pi / 2, np.pi, np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, np.pi, np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, np.pi),
        gates.U3(1, np.pi / 2, np.pi, -np.pi / 2),
        gates.U3(1, -np.pi / 2, np.pi, 0),
    ],
)
iswap_dec.add(
    gates.SWAP,
    [
        gates.iSWAP(0, 1),
        gates.U3(1, np.pi / 2, -np.pi / 2, np.pi / 2),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, -np.pi / 2, np.pi / 2),
        gates.iSWAP(0, 1),
        gates.U3(1, np.pi / 2, -np.pi / 2, np.pi / 2),
    ],
)
iswap_dec.add(gates.iSWAP, [gates.iSWAP(0, 1)])

# register CZ decompositions
cz_dec = GateDecompositions()
cz_dec.add(gates.CNOT, [gates.H(1), gates.CZ(0, 1), gates.H(1)])
cz_dec.add(gates.CZ, [gates.CZ(0, 1)])
cz_dec.add(
    gates.SWAP,
    [
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.H(0),
        gates.CZ(1, 0),
        gates.H(0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.iSWAP,
    [
        gates.U3(0, np.pi / 2.0, 0, -np.pi / 2.0),
        gates.U3(1, np.pi / 2.0, 0, -np.pi / 2.0),
        gates.CZ(0, 1),
        gates.H(0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(0),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.CRX,
    lambda gate: [
        gates.RX(1, gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.RX(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
    ],
)
cz_dec.add(
    gates.CRY,
    lambda gate: [
        gates.RY(1, gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.RY(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
    ],
)
cz_dec.add(
    gates.CRZ,
    lambda gate: [
        gates.RZ(1, gate.parameters[0] / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.CU1,
    lambda gate: [
        gates.RZ(0, gate.parameters[0] / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, -gate.parameters[0] / 2.0),
        gates.CZ(0, 1),
        gates.H(1),
        gates.RZ(1, gate.parameters[0] / 2.0),
    ],
)
cz_dec.add(
    gates.CU2,
    lambda gate: [
        gates.RZ(1, (gate.parameters[1] - gate.parameters[0]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, -np.pi / 4, 0, -(gate.parameters[1] + gate.parameters[0]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, np.pi / 4, gate.parameters[0], 0),
    ],
)
cz_dec.add(
    gates.CU3,
    lambda gate: [
        gates.RZ(1, (gate.parameters[2] - gate.parameters[1]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(
            1,
            -gate.parameters[0] / 2.0,
            0,
            -(gate.parameters[2] + gate.parameters[1]) / 2.0,
        ),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, gate.parameters[0] / 2.0, gate.parameters[1], 0),
    ],
)
cz_dec.add(
    gates.FSWAP,
    [
        gates.U3(0, np.pi / 2, -np.pi / 2, np.pi),
        gates.U3(1, np.pi / 2, np.pi / 2, np.pi / 2),
        gates.CZ(0, 1),
        gates.U3(0, np.pi / 2, 0, -np.pi / 2),
        gates.U3(1, np.pi / 2, 0, np.pi / 2),
        gates.CZ(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, np.pi),
        gates.U3(1, np.pi / 2, 0, np.pi),
    ],
)
cz_dec.add(
    gates.RXX,
    lambda gate: [
        gates.H(0),
        gates.CZ(0, 1),
        gates.RX(1, gate.parameters[0]),
        gates.CZ(0, 1),
        gates.H(0),
    ],
)
cz_dec.add(
    gates.RYY,
    lambda gate: [
        gates.RX(0, np.pi / 2),
        gates.U3(1, np.pi / 2, np.pi / 2, np.pi),
        gates.CZ(0, 1),
        gates.RX(1, gate.parameters[0]),
        gates.CZ(0, 1),
        gates.RX(0, -np.pi / 2),
        gates.U3(1, np.pi / 2, 0, np.pi / 2),
    ],
)
cz_dec.add(
    gates.RZZ,
    lambda gate: [
        gates.H(1),
        gates.CZ(0, 1),
        gates.RX(1, gate.parameters[0]),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.TOFFOLI,
    [
        gates.CZ(1, 2),
        gates.RX(2, -np.pi / 4),
        gates.CZ(0, 2),
        gates.RX(2, np.pi / 4),
        gates.CZ(1, 2),
        gates.RX(2, -np.pi / 4),
        gates.CZ(0, 2),
        gates.RX(2, np.pi / 4),
        gates.RZ(1, np.pi / 4),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RZ(0, np.pi / 4),
        gates.RX(1, -np.pi / 4),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.Unitary,
    lambda gate: two_qubit_decomposition(0, 1, gate.parameters[0], backend=backend),
)
cz_dec.add(
    gates.fSim,
    lambda gate: two_qubit_decomposition(0, 1, gate.matrix(backend), backend=backend),
)
cz_dec.add(
    gates.GeneralizedfSim,
    lambda gate: two_qubit_decomposition(0, 1, gate.matrix(backend), backend=backend),
)

# register other optimized gate decompositions
opt_dec = GateDecompositions()
opt_dec.add(
    gates.SWAP,
    [
        gates.H(0),
        gates.SDG(0),
        gates.SDG(1),
        gates.iSWAP(0, 1),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)


# standard gate decompositions used by :meth:`qibo.gates.gates.Gate.decompose`
standard_decompositions = GateDecompositions()
standard_decompositions.add(gates.SX, [gates.RX(0, np.pi / 2, trainable=False)])
standard_decompositions.add(gates.SXDG, [gates.RX(0, -np.pi / 2, trainable=False)])
standard_decompositions.add(
    gates.U3,
    lambda gate: [
        gates.RZ(0, gate.parameters[2]),
        gates.SX(0),
        gates.RZ(0, gate.parameters[0] + np.pi),
        gates.SX(0),
        gates.RZ(0, gate.parameters[1] + np.pi),
    ],
)
standard_decompositions.add(gates.CY, [gates.SDG(1), gates.CNOT(0, 1), gates.S(1)])
standard_decompositions.add(gates.CZ, [gates.H(1), gates.CNOT(0, 1), gates.H(1)])
standard_decompositions.add(
    gates.CSX, [gates.H(1), gates.CU1(0, 1, np.pi / 2), gates.H(1)]
)
standard_decompositions.add(
    gates.CSXDG, [gates.H(1), gates.CU1(0, 1, -np.pi / 2), gates.H(1)]
)
standard_decompositions.add(
    gates.RZX,
    lambda gate: [
        gates.H(1),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
        gates.H(1),
    ],
)
standard_decompositions.add(
    gates.RXXYY,
    lambda gate: [
        gates.RZ(1, -np.pi / 2),
        gates.S(0),
        gates.SX(1),
        gates.RZ(1, np.pi / 2),
        gates.CNOT(1, 0),
        gates.RY(0, -gate.parameters[0] / 2),
        gates.RY(1, -gate.parameters[0] / 2),
        gates.CNOT(1, 0),
        gates.SDG(0),
        gates.RZ(1, -np.pi / 2),
        gates.SX(1).dagger(),
        gates.RZ(1, np.pi / 2),
    ],
)
standard_decompositions.add(
    gates.RBS,
    lambda gate: [
        gates.H(0),
        gates.CNOT(0, 1),
        gates.H(1),
        gates.RY(0, gate.parameters[0]),
        gates.RY(1, -gate.parameters[0]),
        gates.H(1),
        gates.CNOT(0, 1),
        gates.H(0),
    ],
)
standard_decompositions.add(
    gates.GIVENS, lambda gate: gates.RBS(0, 1, -gate.parameters[0]).decompose()
)
standard_decompositions.add(
    gates.FSWAP, [gates.X(1)] + gates.GIVENS(0, 1, np.pi / 2).decompose() + [gates.X(0)]
)
standard_decompositions.add(
    gates.ECR, [gates.S(0), gates.SX(1), gates.CNOT(0, 1), gates.X(0)]
)
standard_decompositions.add(gates.CCZ, [gates.H(2), gates.TOFFOLI(0, 1, 2), gates.H(2)])
standard_decompositions.add(
    gates.TOFFOLI,
    [
        gates.H(2),
        gates.CNOT(1, 2),
        gates.TDG(2),
        gates.CNOT(0, 2),
        gates.T(2),
        gates.CNOT(1, 2),
        gates.T(1),
        gates.TDG(2),
        gates.CNOT(0, 2),
        gates.CNOT(0, 1),
        gates.T(2),
        gates.T(0),
        gates.TDG(1),
        gates.H(2),
        gates.CNOT(0, 1),
    ],
)
