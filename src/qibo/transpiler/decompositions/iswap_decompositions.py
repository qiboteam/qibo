import math

from qibo import gates
from qibo.transpiler.decompositions.abstract import GateDecompositions

# register the iSWAP decompositions
iswap_dec = GateDecompositions()
iswap_dec.add(
    gates.CNOT,
    [
        gates.U3(0, 3 * math.pi / 2, math.pi, 0),
        gates.U3(1, math.pi / 2, math.pi, math.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, math.pi, 0, math.pi),
        gates.U3(1, math.pi / 2, math.pi, math.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, math.pi / 2, math.pi / 2, math.pi),
        gates.U3(1, math.pi / 2, math.pi, -math.pi / 2),
        gates.GlobalPhase(7 * math.pi / 4, trainable=False),
    ],
)
iswap_dec.add(
    gates.CZ,
    [
        gates.U3(0, -math.pi / 2, math.pi, 0),
        gates.U3(1, -math.pi / 2, math.pi, 0),
        gates.U3(1, math.pi / 2, math.pi, math.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, math.pi, 0, math.pi),
        gates.U3(1, math.pi / 2, math.pi, math.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, math.pi / 2, math.pi / 2, math.pi),
        gates.U3(1, math.pi / 2, math.pi, -math.pi / 2),
        gates.U3(1, -math.pi / 2, math.pi, 0),
        gates.GlobalPhase(7 * math.pi / 4, trainable=False),
    ],
)
iswap_dec.add(
    gates.SWAP,
    [
        gates.iSWAP(0, 1),
        gates.U3(1, math.pi / 2, -math.pi / 2, math.pi / 2),
        gates.iSWAP(0, 1),
        gates.U3(0, math.pi / 2, -math.pi / 2, math.pi / 2),
        gates.iSWAP(0, 1),
        gates.U3(1, math.pi / 2, -math.pi / 2, math.pi / 2),
        gates.GlobalPhase(math.pi / 4, trainable=False),
    ],
)
iswap_dec.add(gates.iSWAP, [gates.iSWAP(0, 1)])
