import numpy as np

from qibo import gates
from qibo.transpiler.unitary_decompositions import (
    two_qubit_decomposition,
    u3_decomposition,
)


class GateDecompositions:
    """Abstract data structure that holds decompositions of gates."""

    def __init__(self):
        self.decompositions = {}

    def add(self, gate, decomposition):
        """Register a decomposition for a gate."""
        self.decompositions[gate] = decomposition

    def _check_instance(self, gate, backend=None):
        special_gates = (
            gates.FusedGate,
            gates.Unitary,
            gates.GeneralizedfSim,
            gates.fSim,
        )
        decomposition = self.decompositions[gate.__class__]
        if gate.parameters:
            decomposition = (
                decomposition(gate, backend)
                if isinstance(gate, special_gates)
                else decomposition(gate)
            )
        return decomposition

    def count_2q(self, gate, backend):
        """Count the number of two-qubit gates in the decomposition of the given gate."""
        decomposition = self._check_instance(gate, backend)
        return len(tuple(g for g in decomposition if len(g.qubits) > 1))

    def count_1q(self, gate, backend):
        """Count the number of single qubit gates in the decomposition of the given gate."""
        decomposition = self._check_instance(gate, backend)
        return len(tuple(g for g in decomposition if len(g.qubits) == 1))

    def __call__(self, gate, backend=None):
        """Decompose a gate."""
        decomposition = self._check_instance(gate, backend)
        return [
            g.on_qubits({i: q for i, q in enumerate(gate.qubits)})
            for g in decomposition
        ]


def _u3_to_gpi2(t, p, l):
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
    gates.Unitary,
    lambda gate, backend: _u3_to_gpi2(*u3_decomposition(gate.parameters[0], backend)),
)
gpi2_dec.add(
    gates.FusedGate,
    lambda gate, backend: _u3_to_gpi2(*u3_decomposition(gate.matrix(backend), backend)),
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
    gates.RX,
    lambda gate: [gates.U3(0, gate.parameters[0], -np.pi / 2, np.pi / 2)],
)
u3_dec.add(gates.RY, lambda gate: [gates.U3(0, gate.parameters[0], 0, 0)])
u3_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
u3_dec.add(
    gates.PRX,
    lambda gate: [
        gates.RZ(0, gate.parameters[1] - np.pi / 2),
        gates.RY(0, -gate.parameters[0]),
        gates.RZ(0, gate.parameters[1] + np.pi / 2),
    ],
)
u3_dec.add(
    gates.GPI2,
    lambda gate: [
        gates.U3(
            0, np.pi / 2, gate.parameters[0] - np.pi / 2, np.pi / 2 - gate.parameters[0]
        ),
    ],
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
    lambda gate, backend: [gates.U3(0, *u3_decomposition(gate.parameters[0], backend))],
)
u3_dec.add(
    gates.FusedGate,
    lambda gate, backend: [
        gates.U3(0, *u3_decomposition(gate.matrix(backend), backend))
    ],
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

# temporary CNOT decompositions for CNOT, CZ, SWAP
cnot_dec_temp = GateDecompositions()
cnot_dec_temp.add(gates.CNOT, [gates.CNOT(0, 1)])
cnot_dec_temp.add(gates.CZ, [gates.H(1), gates.CNOT(0, 1), gates.H(1)])
cnot_dec_temp.add(gates.SWAP, [gates.CNOT(0, 1), gates.CNOT(1, 0), gates.CNOT(0, 1)])

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


def _decomposition_generalized_RBS(ins, outs, theta, phi, controls):
    """Generalized RBS gate as in Fig. 2 of arXiv:2405.20408"""
    rotation_controls = ins[:-1] + outs
    if controls is not None:
        rotation_controls += controls

    list_gates = []
    list_gates.append(gates.X(ins[-1]))
    list_gates.append(gates.X(outs[0]))
    for target in ins[:-1]:
        list_gates.append(gates.CNOT(ins[-1], target))
    for target in outs[1:][::-1]:
        list_gates.append(gates.CNOT(outs[0], target))
    list_gates.append(gates.X(ins[-1]))
    list_gates.append(gates.X(outs[0]))
    list_gates.append(gates.CNOT(ins[-1], outs[0]))
    list_gates.append(gates.RY(ins[-1], -2 * theta).controlled_by(*rotation_controls))
    if phi != 0.0:
        list_gates.append(gates.RZ(ins[-1], 2 * phi).controlled_by(*rotation_controls))
    list_gates.append(gates.CNOT(ins[-1], outs[0]))
    list_gates.append(gates.X(outs[0]))
    list_gates.append(gates.X(ins[-1]))
    for target in outs[1:]:
        list_gates.append(gates.CNOT(outs[0], target))
    for target in ins[:-1][::-1]:
        list_gates.append(gates.CNOT(ins[-1], target))
    list_gates.append(gates.X(outs[0]))
    list_gates.append(gates.X(ins[-1]))

    return list_gates


# standard gate decompositions used by :meth:`qibo.gates.gates.Gate.decompose`
standard_decompositions = GateDecompositions()
standard_decompositions.add(gates.SX, [gates.RX(0, np.pi / 2, trainable=False)])
standard_decompositions.add(gates.SXDG, [gates.RX(0, -np.pi / 2, trainable=False)])
standard_decompositions.add(
    gates.PRX,
    lambda gate: [
        gates.RZ(0, -gate.parameters[1] - np.pi / 2),
        gates.RY(0, -gate.parameters[0]),
        gates.RZ(0, gate.parameters[1] + np.pi / 2),
    ],
)
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
    gates.CRY,
    lambda gate: [
        gates.RY(1, gate.parameters[0] / 4),
        gates.CNOT(0, 1),
        gates.RY(1, -gate.parameters[0] / 2),
        gates.CNOT(0, 1),
        gates.RY(1, gate.parameters[0] / 4),
    ],
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
        gates.RY(0, gate.parameters[0]),
        gates.RY(1, gate.parameters[0]),
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
standard_decompositions.add(
    gates.GeneralizedRBS,
    lambda gate: _decomposition_generalized_RBS(
        ins=list(range(len(gate.init_args[0]))),
        outs=list(
            range(
                len(gate.init_args[0]),
                len(gate.init_args[0]) + len(gate.init_args[1]),
            )
        ),
        theta=gate.init_kwargs["theta"],
        phi=gate.init_kwargs["phi"],
        controls=list(
            range(
                len(gate.init_args[0]) + len(gate.init_args[1]),
                len(gate.init_args[0])
                + len(gate.init_args[1])
                + len(gate.control_qubits),
            )
        ),
    ),
)
