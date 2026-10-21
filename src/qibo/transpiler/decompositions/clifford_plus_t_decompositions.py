import math
from importlib.util import find_spec

from qibo import gates
from qibo.transpiler.decompositions.abstract import GateDecompositions

# the decomposition needs these optional packages, which are not imported to check them
if find_spec("pygridsynth") and find_spec("mpmath"):
    METHOD = "clifford_plus_t"
    clifford_plus_t = GateDecompositions()
    clifford_plus_t._set_precision_cliff_plus_t()
    clifford_plus_t.add(
        gates.SDG,
        [gates.Y(0), gates.S(0), gates.Y(0), gates.GlobalPhase(-math.pi / 2)],
    )
    clifford_plus_t.add(
        gates.TDG,
        [
            gates.T(0),
            gates.Y(0),
            gates.S(0),
            gates.Y(0),
            gates.GlobalPhase(-math.pi / 2),
        ],
    )
    clifford_plus_t.add(
        gates.RX,
        lambda gate: (
            [gates.H(0)]
            + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
            + [gates.H(0)]
        ),
    )
    clifford_plus_t.add(
        gates.RY,
        lambda gate: (
            [gates.SDG(0), gates.H(0)]
            + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
            + [gates.H(0), gates.S(0)]
        ),
    )
    clifford_plus_t.add(
        gates.RZ, lambda gate: clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
    )
    clifford_plus_t.add(
        gates.PRX,
        lambda gate: (
            clifford_plus_t._rz_into_cliff_and_t(-gate.parameters[1] - math.pi / 2)
            + gates.RY(0, -gate.parameters[0]).decompose(method=METHOD)
            + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] + math.pi / 2)
        ),
    )
    clifford_plus_t.add(
        gates.U1,
        lambda gate: (
            clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
            + [gates.GlobalPhase(gate.parameters[0] / 2)]
        ),
    )
    clifford_plus_t.add(
        gates.U2,
        lambda gate: (
            clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] - math.pi)
            + [gates.SX(0)]
            + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0] + math.pi)
        ),
    )
    clifford_plus_t.add(
        gates.U3,
        lambda gate: (
            clifford_plus_t._rz_into_cliff_and_t(gate.parameters[2])
            + [gates.SX(0)]
            + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0] + math.pi)
            + [gates.SX(0)]
            + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] + math.pi)
            + [gates.GlobalPhase(math.pi / 2)]
        ),
    )
    clifford_plus_t.add(
        gates.PRX,
        lambda gate: (
            clifford_plus_t._rz_into_cliff_and_t(-gate.parameters[1] - math.pi / 2)
            + gates.RY(0, -gate.parameters[0]).decompose(method=METHOD)
            + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] + math.pi / 2)
        ),
    )
    clifford_plus_t.add(gates.CY, [gates.SDG(1), gates.CNOT(0, 1), gates.S(1)])
    clifford_plus_t.add(gates.CZ, [gates.H(1), gates.CNOT(0, 1), gates.H(1)])
    clifford_plus_t.add(
        gates.CRX,
        lambda gate: (
            gates.RX(1, gate.parameters[0] / 2.0).decompose(method=METHOD)
            + [gates.H(1), gates.CNOT(0, 1)]
            + gates.RZ(1, -gate.parameters[0] / 2.0).decompose(method=METHOD)
            + [gates.CNOT(0, 1), gates.H(1)]
        ),
    )
    clifford_plus_t.add(
        gates.CRY,
        lambda gate: (
            gates.RY(1, gate.parameters[0] / 4).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.RY(1, -gate.parameters[0] / 2).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.RY(1, gate.parameters[0] / 4).decompose(method=METHOD)
        ),
    )
    clifford_plus_t.add(
        gates.CRZ,
        lambda gate: (
            gates.RZ(1, gate.parameters[0] / 2.0).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.RZ(1, -gate.parameters[0] / 2.0).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
        ),
    )
    clifford_plus_t.add(
        gates.CU1,
        lambda gate: (
            gates.RZ(0, gate.parameters[0] / 2.0).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.RZ(1, -gate.parameters[0] / 2.0).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.RZ(1, gate.parameters[0] / 2.0).decompose(method=METHOD)
            + [gates.GlobalPhase(gate.parameters[0] / 4.0)]
        ),
    )
    clifford_plus_t.add(
        gates.CU2,
        lambda gate: (
            gates.RZ(1, (gate.parameters[1] - gate.parameters[0]) / 2.0).decompose(
                method=METHOD
            )
            + [gates.CNOT(0, 1)]
            + gates.U3(
                1, -math.pi / 4, 0, -(gate.parameters[1] + gate.parameters[0]) / 2.0
            ).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.U3(1, math.pi / 4, gate.parameters[0], 0).decompose(method=METHOD)
        ),
    )
    clifford_plus_t.add(
        gates.CU3,
        lambda gate: (
            gates.RZ(1, (gate.parameters[2] - gate.parameters[1]) / 2.0).decompose(
                method=METHOD
            )
            + [gates.CNOT(0, 1)]
            + gates.U3(
                1,
                -gate.parameters[0] / 2.0,
                0,
                -(gate.parameters[2] + gate.parameters[1]) / 2.0,
            ).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.U3(1, gate.parameters[0] / 2.0, gate.parameters[1], 0).decompose(
                method=METHOD
            )
        ),
    )
    clifford_plus_t.add(
        gates.CSX,
        [gates.H(1)]
        + gates.CU1(0, 1, math.pi / 2).decompose(method=METHOD)
        + [gates.H(1)],
    )
    clifford_plus_t.add(
        gates.CSXDG,
        [gates.H(1)]
        + gates.CU1(0, 1, -math.pi / 2).decompose(method=METHOD)
        + [gates.H(1)],
    )
    clifford_plus_t.add(
        gates.SWAP, [gates.CNOT(0, 1), gates.CNOT(1, 0), gates.CNOT(0, 1)]
    )
    clifford_plus_t.add(
        gates.iSWAP,
        [
            gates.S(0),
            gates.S(1),
            gates.H(0),
            gates.CNOT(0, 1),
            gates.CNOT(1, 0),
            gates.H(1),
        ],
    )
    clifford_plus_t.add(
        gates.SiSWAP,
        [gates.SX(0), gates.SX(1)]
        + gates.RZ(0, math.pi / 2).decompose(method=METHOD)
        + [gates.CNOT(0, 1), gates.SX(0)]
        + gates.RZ(1, -math.pi / 4).decompose(method=METHOD)
        + gates.RZ(0, -math.pi / 4).decompose(method=METHOD)
        + [gates.SX(1), gates.SX(0)]
        + gates.RZ(0, math.pi / 2).decompose(method=METHOD)
        + [gates.CNOT(0, 1), gates.SX(0)],
    )
    clifford_plus_t.add(
        gates.SiSWAPDG,
        [
            gate.dagger()
            for gate in reversed(gates.SiSWAP(0, 1).decompose(method=METHOD))
        ],
    )
    clifford_plus_t.add(
        gates.RXX,
        lambda gate: (
            [gates.H(0), gates.H(1), gates.CNOT(0, 1)]
            + gates.RZ(1, gate.parameters[0]).decompose(method=METHOD)
            + [gates.CNOT(0, 1), gates.H(1), gates.H(0)]
        ),
    )
    clifford_plus_t.add(
        gates.RYY,
        lambda gate: (
            gates.RX(0, math.pi / 2).decompose(method=METHOD)
            + gates.RX(1, math.pi / 2).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.RZ(1, gate.parameters[0]).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
            + gates.RX(1, -math.pi / 2).decompose(method=METHOD)
            + gates.RX(0, -math.pi / 2).decompose(method=METHOD)
        ),
    )
    clifford_plus_t.add(
        gates.RZZ,
        lambda gate: (
            [gates.CNOT(0, 1)]
            + gates.RZ(1, gate.parameters[0]).decompose(method=METHOD)
            + [gates.CNOT(0, 1)]
        ),
    )
    clifford_plus_t.add(
        gates.RZX,
        lambda gate: (
            [gates.H(1), gates.CNOT(0, 1)]
            + gates.RZ(1, gate.parameters[0]).decompose(method=METHOD)
            + [gates.CNOT(0, 1), gates.H(1)]
        ),
    )
    clifford_plus_t.add(
        gates.RXXYY,
        lambda gate: (
            gates.RZ(1, -math.pi / 2).decompose(method=METHOD)
            + [gates.S(0), gates.SX(1)]
            + gates.RZ(1, math.pi / 2).decompose(method=METHOD)
            + [gates.CNOT(1, 0)]
            + gates.RY(0, -gate.parameters[0] / 2).decompose(method=METHOD)
            + gates.RY(1, -gate.parameters[0] / 2).decompose(method=METHOD)
            + [gates.CNOT(1, 0), gates.SDG(0)]
            + gates.RZ(1, -math.pi / 2).decompose(method=METHOD)
            + [gates.SXDG(1)]
            + gates.RZ(1, math.pi / 2).decompose(method=METHOD)
        ),
    )
    clifford_plus_t.add(
        gates.GIVENS,
        lambda gate: gates.RBS(0, 1, -gate.parameters[0]).decompose(method=METHOD),
    )
    clifford_plus_t.add(
        gates.FSWAP,
        [gates.X(1)]
        + gates.GIVENS(0, 1, math.pi / 2).decompose(method=METHOD)
        + [gates.X(0)],
    )
    clifford_plus_t.add(
        gates.ECR,
        [
            gates.S(0),
            gates.S(1),
            gates.H(1),
            gates.S(1),
            gates.Y(1),
            gates.Z(1),
            gates.CNOT(0, 1),
            gates.X(0),
        ],
    )
    clifford_plus_t.add(
        gates.CCZ,
        [
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
            gates.CNOT(0, 1),
        ],
    )
    clifford_plus_t.add(
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
    clifford_plus_t.add(
        gates.FanOut,
        lambda gate: [gates.CNOT(0, qub) for qub in range(1, len(gate.qubits))],
    )
