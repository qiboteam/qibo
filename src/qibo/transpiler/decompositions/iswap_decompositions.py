from qibo import gates
from qibo.transpiler.decompositions.abstract import GateDecompositions

# register the iSWAP decompositions
iswap_dec = GateDecompositions()
iswap_dec.add(
    gates.CNOT,
    [
        gates.X(1),
        gates.H(0),
        gates.H(1),
        gates.iSWAP(0, 1),
        gates.X(0),
        gates.X(1),
        gates.H(1),
        gates.iSWAP(0, 1),
        gates.S(1),
        gates.X(1),
        gates.H(0),
        gates.H(1),
        gates.S(0),
    ],
)
iswap_dec.add(
    gates.CZ,
    [
        gates.H(0),
        gates.Z(1),
        gates.iSWAP(0, 1),
        gates.X(0),
        gates.X(1),
        gates.H(1),
        gates.iSWAP(0, 1),
        gates.S(1),
        gates.X(1),
        gates.H(0),
        gates.S(0),
    ],
)
iswap_dec.add(
    gates.SWAP,
    [gates.H(0), gates.SDG(0), gates.SDG(1), gates.iSWAP(0, 1)]
    + iswap_dec(gates.CZ(0, 1))
    + [gates.H(1)],
)
iswap_dec.add(gates.iSWAP, [gates.iSWAP(0, 1)])
