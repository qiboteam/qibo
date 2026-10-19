import math

from qibo import gates
from qibo.transpiler.decompositions.abstract import GateDecompositions
from qibo.transpiler.decompositions.unitary_decompositions import (
    two_qubit_decomposition,
)

# register CZ decompositions
cz_dec = GateDecompositions()
cz_dec.add(gates.CNOT, [gates.H(1), gates.CZ(0, 1), gates.H(1)])
cz_dec.add(gates.CZ, [gates.CZ(0, 1)])
cz_dec.add(gates.CY, [gates.SDG(1), gates.H(1), gates.CZ(0, 1), gates.H(1), gates.S(1)])
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
        gates.SX(0),
        gates.SX(1),
        gates.S(0),
        gates.S(1),
        gates.X(0),
        gates.X(1),
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
        gates.U3(1, -math.pi / 4, 0, -(gate.parameters[1] + gate.parameters[0]) / 2.0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.H(1),
        gates.U3(1, math.pi / 4, gate.parameters[0], 0),
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
        gates.X(0),
        gates.S(0),
        gates.X(0),
        gates.Z(0),
        gates.SX(0),
        gates.SDG(1),
        gates.H(1),
        gates.X(1),
        gates.SDG(1),
        gates.X(1),
        gates.CZ(0, 1),
        gates.SX(0),
        gates.S(0),
        gates.X(0),
        gates.SDG(1),
        gates.SXDG(1),
        gates.SDG(1),
        gates.SXDG(1),
        gates.CZ(0, 1),
        gates.U3(0, math.pi / 2, math.pi / 2, math.pi),
        gates.U3(1, math.pi / 2, 0, math.pi),
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
        gates.RX(0, math.pi / 2),
        gates.U3(1, math.pi / 2, math.pi / 2, math.pi),
        gates.CZ(0, 1),
        gates.RX(1, gate.parameters[0]),
        gates.CZ(0, 1),
        gates.RX(0, -math.pi / 2),
        gates.SDG(1),
        gates.SXDG(1),
        gates.SDG(1),
        gates.SXDG(1),
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
        gates.RX(2, -math.pi / 4),
        gates.CZ(0, 2),
        gates.RX(2, math.pi / 4),
        gates.CZ(1, 2),
        gates.RX(2, -math.pi / 4),
        gates.CZ(0, 2),
        gates.RX(2, math.pi / 4),
        gates.RZ(1, math.pi / 4),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RZ(0, math.pi / 4),
        gates.RX(1, -math.pi / 4),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)
cz_dec.add(
    gates.CCZ, [gates.H(2)] + cz_dec.decompositions[gates.TOFFOLI] + [gates.H(2)]
)
cz_dec.add(
    gates.Unitary,
    lambda gate, backend: two_qubit_decomposition(
        0, 1, gate.parameters[0], backend=backend
    ),
)
cz_dec.add(
    gates.fSim,
    lambda gate, backend: two_qubit_decomposition(
        0, 1, gate.matrix(backend), backend=backend
    ),
)
cz_dec.add(
    gates.GeneralizedfSim,
    lambda gate, backend: two_qubit_decomposition(
        0, 1, gate.matrix(backend), backend=backend
    ),
)
cz_dec.add(
    gates.RBS,
    lambda gate: [
        gates.H(0),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RY(0, gate.parameters[0]),
        gates.RY(1, -gate.parameters[0]),
        gates.CZ(0, 1),
        gates.H(1),
        gates.H(0),
    ],
)
cz_dec.add(
    gates.DEUTSCH,
    lambda gate: [
        gates.CZ(1, 2),
        gates.RX(2, -gate.parameters[0] / 2),
        gates.CZ(0, 2),
        gates.RX(2, gate.parameters[0] / 2),
        gates.CZ(1, 2),
        gates.RX(2, -gate.parameters[0] / 2),
        gates.CZ(0, 2),
        gates.RX(2, gate.parameters[0] / 2),
        gates.RZ(1, math.pi / 4),
        gates.H(1),
        gates.CZ(0, 1),
        gates.RZ(0, math.pi / 4),
        gates.RX(1, -math.pi / 4),
        gates.CZ(0, 1),
        gates.H(1),
    ],
)


# Angles of the single-qubit gates of the decomposition of CZ into SYC gates, which are
# the only ones that are not multiples of pi / 12.
_SYC_ANGLE_1 = math.acos(2 - math.sqrt(3))
_SYC_ANGLE_2 = math.acos((2 - math.sqrt(3)) ** 2)


# register the decompositions of the CZ gate into the entangling native gates other than
# iSWAP and CNOT, with the CZ gate acting on the qubits 0 and 1. The other gates are
# decomposed into CZ gates, which are replaced by these decompositions. Each one has the
# lowest possible number of native gates, which is one if the gate is equivalent to CZ
# up to single-qubit gates, and two otherwise. SWAP is not here because it cannot create
# entanglement, so it cannot be the only entangling native gate.
native_cz_dec = {
    gates.CH: [gates.RY(1, math.pi / 4), gates.CH(0, 1), gates.RY(1, -math.pi / 4)],
    gates.CSX: [gates.H(1), gates.CSX(0, 1), gates.CSX(0, 1), gates.H(1)],
    gates.CSXDG: [gates.H(1), gates.CSXDG(0, 1), gates.CSXDG(0, 1), gates.H(1)],
    gates.CY: [gates.H(1), gates.S(1), gates.CY(0, 1), gates.SDG(1), gates.H(1)],
    gates.ECR: [
        gates.U3(0, -math.pi, -math.pi, -math.pi / 2),
        gates.U3(1, math.pi / 2, -math.pi, 0),
        gates.ECR(0, 1),
        gates.U3(1, -math.pi / 2, -math.pi / 2, -math.pi),
    ],
    gates.FSWAP: [
        gates.U3(0, -math.pi / 2, -math.pi / 2, -math.pi),
        gates.U3(1, 0, -math.pi / 2, -math.pi / 2),
        gates.FSWAP(0, 1),
        gates.U3(0, 0, math.pi, math.pi / 2),
        gates.U3(1, math.pi / 2, math.pi / 2, 0),
        gates.FSWAP(0, 1),
        gates.U3(0, math.pi / 2, math.pi / 2, 0),
        gates.U3(1, 0, -math.pi, 0),
    ],
    gates.SiSWAP: [
        gates.U3(0, -math.pi / 2, math.pi / 2, math.pi / 2),
        gates.U3(1, math.pi / 2, math.pi / 2, -math.pi / 2),
        gates.SiSWAP(0, 1),
        gates.U3(0, -math.pi, math.pi / 2, -math.pi / 2),
        gates.U3(1, 0, 0, -math.pi),
        gates.SiSWAP(0, 1),
        gates.U3(0, math.pi / 2, 0, math.pi / 2),
        gates.U3(1, math.pi / 2, -math.pi, -math.pi / 2),
    ],
    gates.SiSWAPDG: [
        gates.U3(0, math.pi / 2, math.pi / 2, -math.pi),
        gates.U3(1, -math.pi / 2, math.pi / 2, 0),
        gates.SiSWAPDG(0, 1),
        gates.U3(0, 0, -math.pi / 2, -math.pi / 2),
        gates.U3(1, -math.pi, -math.pi / 2, math.pi / 2),
        gates.SiSWAPDG(0, 1),
        gates.U3(0, math.pi / 2, math.pi / 2, -math.pi / 2),
        gates.U3(1, -math.pi / 2, math.pi / 2, -math.pi / 2),
    ],
    gates.SYC: [
        gates.U3(0, 0, -2 * math.pi / 3, -2 * math.pi / 3),
        gates.U3(1, _SYC_ANGLE_1 - math.pi, 7 * math.pi / 12, -math.pi / 12),
        gates.SYC(0, 1),
        gates.U3(0, math.pi - _SYC_ANGLE_2, -7 * math.pi / 12, -math.pi / 2),
        gates.U3(1, -math.pi, -5 * math.pi / 6, math.pi / 2),
        gates.SYC(0, 1),
        gates.U3(0, -math.pi, math.pi / 12, 7 * math.pi / 12),
        gates.U3(1, -_SYC_ANGLE_1, 7 * math.pi / 12, -math.pi / 3),
    ],
}
