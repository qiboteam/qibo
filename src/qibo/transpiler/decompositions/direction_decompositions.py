import math

from qibo import gates

# register the decompositions of the native two-qubit gates that are not symmetric, i.e. the
# ones that change when their qubits are swapped, into the same gate acting in the opposite
# direction, and single-qubit gates. The decomposed gate acts on the qubits 0 and 1, and
# the gate of the decomposition acts on the qubits 1 and 0.
direction_dec = {
    gates.CH: [
        gates.RY(0, math.pi / 4),
        gates.RY(1, -math.pi / 4),
        gates.CH(1, 0),
        gates.RY(0, -math.pi / 4),
        gates.RY(1, math.pi / 4),
    ],
    gates.CNOT: [
        gates.H(0),
        gates.H(1),
        gates.CNOT(1, 0),
        gates.H(0),
        gates.H(1),
    ],
    gates.CSX: [
        gates.H(0),
        gates.H(1),
        gates.CSX(1, 0),
        gates.H(0),
        gates.H(1),
    ],
    gates.CSXDG: [
        gates.H(0),
        gates.H(1),
        gates.CSXDG(1, 0),
        gates.H(0),
        gates.H(1),
    ],
    gates.CY: [
        gates.H(0),
        gates.S(0),
        gates.SDG(1),
        gates.H(1),
        gates.CY(1, 0),
        gates.SDG(0),
        gates.H(0),
        gates.H(1),
        gates.S(1),
    ],
    gates.ECR: [
        gates.H(0),
        gates.H(1),
        gates.ECR(1, 0),
        gates.Z(0),
        gates.H(0),
        gates.H(1),
        gates.Z(1),
    ],
}
