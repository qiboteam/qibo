import math

from qibo import gates
from qibo.transpiler.decompositions.abstract import GateDecompositions
from qibo.transpiler.unitary_decompositions import u3_decomposition


def _u3_to_gpi2(t: float, p: float, l: float) -> list[gates.Gate]:
    """Decompose a :class:`qibo.gates.U3` gate into :class:`qibo.gates.GPI2` gates.

    The decomposition is optimized to use the minimum number of gates.

    Args:
        t (float): first parameter of :class:`qibo.gates.U3` gate.
        p (float): second parameter of :class:`qibo.gates.U3` gate.
        l (float): third parameter of :class:`qibo.gates.U3` gate.

    Returns:
        list: Native gates that decompose the :class:`qibo.gates.U3` gate.
    """
    decomposition = []
    if l != 0.0:
        decomposition.append(gates.RZ(0, l))
    decomposition.append(gates.GPI2(0, 0))
    if t != -math.pi:
        decomposition.append(gates.RZ(0, t + math.pi))
    decomposition.append(gates.GPI2(0, 0))
    if p != -math.pi:
        decomposition.append(gates.RZ(0, p + math.pi))
    return decomposition


# Decompose single qubit gates using GPI2 (more efficient on hardware)
gpi2_dec = GateDecompositions()
gpi2_dec.add(gates.H, [gates.Z(0), gates.GPI2(0, math.pi / 2)])
gpi2_dec.add(
    gates.X, [gates.GPI2(0, math.pi / 2), gates.GPI2(0, math.pi / 2), gates.Z(0)]
)
gpi2_dec.add(gates.Y, [gates.Z(0), gates.GPI2(0, 0), gates.GPI2(0, 0)])
gpi2_dec.add(gates.Z, [gates.Z(0)])
gpi2_dec.add(gates.S, [gates.RZ(0, math.pi / 2)])
gpi2_dec.add(gates.SDG, [gates.RZ(0, -math.pi / 2)])
gpi2_dec.add(gates.T, [gates.RZ(0, math.pi / 4)])
gpi2_dec.add(gates.TDG, [gates.RZ(0, -math.pi / 4)])
gpi2_dec.add(gates.SX, [gates.GPI2(0, 0)])
gpi2_dec.add(
    gates.RX,
    lambda gate: [
        gates.Z(0),
        gates.GPI2(0, math.pi / 2),
        gates.RZ(0, gate.parameters[0] + math.pi),
        gates.GPI2(0, math.pi / 2),
    ],
)
gpi2_dec.add(
    gates.RY,
    lambda gate: [
        gates.GPI2(0, 0),
        gates.RZ(0, gate.parameters[0] + math.pi),
        gates.GPI2(0, 0),
        gates.Z(0),
    ],
)
gpi2_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
gpi2_dec.add(gates.GPI2, lambda gate: [gates.GPI2(0, gate.parameters[0])])
gpi2_dec.add(gates.U1, lambda gate: [gates.RZ(0, gate.parameters[0])])
gpi2_dec.add(
    gates.U2,
    lambda gate: _u3_to_gpi2(math.pi / 2, gate.parameters[0], gate.parameters[1]),
)
gpi2_dec.add(gates.U3, lambda gate: _u3_to_gpi2(*gate.parameters))
gpi2_dec.add(
    gates.Unitary,
    lambda gate, backend: _u3_to_gpi2(*u3_decomposition(gate.parameters[0], backend)),
)
gpi2_dec.add(
    gates.FusedGate,
    lambda gate, backend: _u3_to_gpi2(*u3_decomposition(gate.matrix(backend), backend)),
)
