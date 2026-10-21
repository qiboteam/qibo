import math

from qibo import gates
from qibo.transpiler.decompositions.abstract import GateDecompositions
from qibo.transpiler.decompositions.unitary_decompositions import u3_decomposition

# Decompose single qubit gates using U3
u3_dec = GateDecompositions()
u3_dec.add(
    gates.H, [gates.U3(0, -math.pi / 2, math.pi, 0), gates.GlobalPhase(math.pi / 2)]
)
u3_dec.add(gates.X, [gates.U3(0, math.pi, 0, math.pi), gates.GlobalPhase(math.pi / 2)])
u3_dec.add(gates.Y, [gates.U3(0, math.pi, 0, 0), gates.GlobalPhase(math.pi / 2)])
u3_dec.add(gates.Z, [gates.Z(0)])
u3_dec.add(gates.S, [gates.RZ(0, math.pi / 2), gates.GlobalPhase(math.pi / 4)])
u3_dec.add(gates.SDG, [gates.RZ(0, -math.pi / 2), gates.GlobalPhase(7 * math.pi / 4)])
u3_dec.add(gates.T, [gates.RZ(0, math.pi / 4), gates.GlobalPhase(math.pi / 8)])
u3_dec.add(gates.TDG, [gates.RZ(0, -math.pi / 4), gates.GlobalPhase(15 * math.pi / 8)])
u3_dec.add(
    gates.SX,
    [
        gates.U3(0, math.pi / 2, -math.pi / 2, math.pi / 2),
        gates.GlobalPhase(math.pi / 4),
    ],
)
u3_dec.add(
    gates.RX,
    lambda gate: [gates.U3(0, gate.parameters[0], -math.pi / 2, math.pi / 2)],
)
u3_dec.add(gates.RY, lambda gate: [gates.U3(0, gate.parameters[0], 0, 0)])
u3_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
u3_dec.add(
    gates.PRX,
    lambda gate: [
        gates.RZ(0, -gate.parameters[1]),
        gates.U3(0, gate.parameters[0], -math.pi / 2, math.pi / 2),
        gates.RZ(0, gate.parameters[1]),
    ],
)
u3_dec.add(
    gates.GPI2,
    lambda gate: [
        gates.U3(
            0,
            math.pi / 2,
            gate.parameters[0] - math.pi / 2,
            math.pi / 2 - gate.parameters[0],
        ),
    ],
)
u3_dec.add(
    gates.U1,
    lambda gate: [
        gates.RZ(0, gate.parameters[0]),
        gates.GlobalPhase((gate.parameters[0] / 2) % (2 * math.pi)),
    ],
)
u3_dec.add(
    gates.U2,
    lambda gate: [gates.U3(0, math.pi / 2, gate.parameters[0], gate.parameters[1])],
)
u3_dec.add(
    gates.U3,
    lambda gate: [
        gates.U3(0, gate.parameters[0], gate.parameters[1], gate.parameters[2])
    ],
)
# The U3 gate of ``u3_decomposition`` equals the unitary divided by the square root of
# its determinant, so the phase of the latter is added to get an exact decomposition.
u3_dec.add(
    gates.Unitary,
    lambda gate, backend: [
        gates.U3(0, *u3_decomposition(gate.parameters[0], backend)),
        gates.GlobalPhase(
            (float(backend.angle(backend.det(backend.cast(gate.parameters[0])))) / 2)
            % (2 * math.pi)
        ),
    ],
)
u3_dec.add(
    gates.FusedGate,
    lambda gate, backend: [
        gates.U3(0, *u3_decomposition(gate.matrix(backend), backend)),
        gates.GlobalPhase(
            (float(backend.angle(backend.det(gate.matrix(backend)))) / 2)
            % (2 * math.pi)
        ),
    ],
)
