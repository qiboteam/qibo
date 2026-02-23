import math

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

        if isinstance(gate, gates.FanOut):
            decomposition = decomposition(gate)
        elif gate.parameters:
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
        return [g.on_qubits(dict(enumerate(gate.qubits))) for g in decomposition]

    def _set_precision_cliff_plus_t(
        self, epsilon: float = 1e-16, mpmath_dps: int = 256
    ):
        import mpmath  # pylint: disable=C0415

        mpmath.mp.dps = mpmath_dps
        self._epsilon = mpmath.mpmathify(epsilon)

    def _rz_into_cliff_and_t(
        self,
        theta: float,
        qubit: int = 0,
    ):
        import mpmath  # pylint: disable=C0415
        from pygridsynth.gridsynth import gridsynth_gates  # pylint: disable=C0415

        theta = float(theta)
        theta = mpmath.mpmathify(theta)

        sequence = gridsynth_gates(theta=theta, epsilon=self._epsilon)
        sequence = sequence.split("W")
        num_global_phase = len(sequence[1:])
        sequence = sequence[0]

        global_phase = math.exp(1j * num_global_phase * math.pi / 4)

        gate_list = [
            getattr(gates, gate_name)(qubit) for gate_name in reversed(sequence)
        ]

        return gate_list


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

# Decompose single qubit gates using U3
u3_dec = GateDecompositions()
u3_dec.add(gates.H, [gates.U3(0, -math.pi / 2, math.pi, 0)])
u3_dec.add(gates.X, [gates.U3(0, math.pi, 0, math.pi)])
u3_dec.add(gates.Y, [gates.U3(0, math.pi, 0, 0)])
u3_dec.add(gates.Z, [gates.Z(0)])
u3_dec.add(gates.S, [gates.RZ(0, math.pi / 2)])
u3_dec.add(gates.SDG, [gates.RZ(0, -math.pi / 2)])
u3_dec.add(gates.T, [gates.RZ(0, math.pi / 4)])
u3_dec.add(gates.TDG, [gates.RZ(0, -math.pi / 4)])
u3_dec.add(gates.SX, [gates.U3(0, math.pi / 2, -math.pi / 2, math.pi / 2)])
u3_dec.add(
    gates.RX,
    lambda gate: [gates.U3(0, gate.parameters[0], -math.pi / 2, math.pi / 2)],
)
u3_dec.add(gates.RY, lambda gate: [gates.U3(0, gate.parameters[0], 0, 0)])
u3_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
u3_dec.add(
    gates.PRX,
    lambda gate: [
        gates.RZ(0, gate.parameters[1] - math.pi / 2),
        gates.RY(0, -gate.parameters[0]),
        gates.RZ(0, gate.parameters[1] + math.pi / 2),
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
u3_dec.add(gates.U1, lambda gate: [gates.RZ(0, gate.parameters[0])])
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
        gates.U3(0, 3 * math.pi / 2, math.pi, 0),
        gates.U3(1, math.pi / 2, math.pi, math.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, math.pi, 0, math.pi),
        gates.U3(1, math.pi / 2, math.pi, math.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, math.pi / 2, math.pi / 2, math.pi),
        gates.U3(1, math.pi / 2, math.pi, -math.pi / 2),
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
        gates.U3(0, math.pi / 2.0, 0, -math.pi / 2.0),
        gates.U3(1, math.pi / 2.0, 0, -math.pi / 2.0),
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
        gates.U3(0, math.pi / 2, -math.pi / 2, math.pi),
        gates.U3(1, math.pi / 2, math.pi / 2, math.pi / 2),
        gates.CZ(0, 1),
        gates.U3(0, math.pi / 2, 0, -math.pi / 2),
        gates.U3(1, math.pi / 2, 0, math.pi / 2),
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
        gates.U3(1, math.pi / 2, 0, math.pi / 2),
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


def _decomposition_generalized_RBS(gate):
    """Generalized RBS gate as in Fig. 2 of arXiv:2405.20408"""
    ins = list(range(len(gate.init_args[0])))
    outs = list(range(len(ins), len(ins) + len(gate.init_args[1])))
    theta = gate.init_kwargs["theta"]
    phi = gate.init_kwargs["phi"]

    rotation_controls = ins[:-1] + outs

    list_gates = []
    if len(ins) > 1:
        list_gates.append(gates.X(ins[-1]))
        list_gates.append(gates.FanOut(ins[-1], *ins[:-1]))
        list_gates.append(gates.X(ins[-1]))
    if len(outs) > 1:
        list_gates.append(gates.X(outs[0]))
        list_gates.append(gates.FanOut(outs[0], *outs[1:][::-1]))
        list_gates.append(gates.X(outs[0]))
    list_gates.append(gates.CNOT(ins[-1], outs[0]))
    list_gates.append(gates.RY(ins[-1], -2 * theta).controlled_by(*rotation_controls))
    if phi != 0.0:
        list_gates.append(gates.RZ(ins[-1], 2 * phi).controlled_by(*rotation_controls))
    list_gates.append(gates.CNOT(ins[-1], outs[0]))
    if len(outs) > 1:
        list_gates.append(gates.X(outs[0]))
        list_gates.append(gates.FanOut(outs[0], *outs[1:][::-1]))
        list_gates.append(gates.X(outs[0]))
    if len(ins) > 1:
        list_gates.append(gates.X(ins[-1]))
        list_gates.append(gates.FanOut(ins[-1], *ins[:-1]))
        list_gates.append(gates.X(ins[-1]))

    return list_gates


# standard gate decompositions used by :meth:`qibo.gates.gates.Gate.decompose`
standard_decompositions = GateDecompositions()
standard_decompositions.add(gates.SX, [gates.RX(0, math.pi / 2, trainable=False)])
standard_decompositions.add(gates.SXDG, [gates.RX(0, -math.pi / 2, trainable=False)])
standard_decompositions.add(
    gates.PRX,
    lambda gate: [
        gates.RZ(0, -gate.parameters[1] - math.pi / 2),
        gates.RY(0, -gate.parameters[0]),
        gates.RZ(0, gate.parameters[1] + math.pi / 2),
    ],
)
standard_decompositions.add(
    gates.U3,
    lambda gate: [
        gates.RZ(0, gate.parameters[2]),
        gates.SX(0),
        gates.RZ(0, gate.parameters[0] + math.pi),
        gates.SX(0),
        gates.RZ(0, gate.parameters[1] + math.pi),
    ],
)
standard_decompositions.add(gates.CY, [gates.SDG(1), gates.CNOT(0, 1), gates.S(1)])
standard_decompositions.add(gates.CZ, [gates.H(1), gates.CNOT(0, 1), gates.H(1)])
standard_decompositions.add(
    gates.CSX, [gates.H(1), gates.CU1(0, 1, math.pi / 2), gates.H(1)]
)
standard_decompositions.add(
    gates.CSXDG, [gates.H(1), gates.CU1(0, 1, -math.pi / 2), gates.H(1)]
)
standard_decompositions.add(
    gates.CRX,
    lambda gate: [
        gates.RX(1, gate.parameters[0] / 2.0),
        gates.H(1),
        gates.CNOT(0, 1),
        gates.RZ(1, -gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.H(1),
    ],
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
    gates.CRZ,
    lambda gate: [
        gates.RZ(1, gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.RZ(1, -gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
    ],
)
standard_decompositions.add(
    gates.CU1,
    lambda gate: [
        gates.RZ(0, gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.RZ(1, -gate.parameters[0] / 2.0),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0] / 2.0),
    ],
)
standard_decompositions.add(
    gates.CU2,
    lambda gate: [
        gates.RZ(1, (gate.parameters[1] - gate.parameters[0]) / 2.0),
        gates.CNOT(0, 1),
        gates.U3(1, -math.pi / 4, 0, -(gate.parameters[1] + gate.parameters[0]) / 2.0),
        gates.CNOT(0, 1),
        gates.U3(1, math.pi / 4, gate.parameters[0], 0),
    ],
)
standard_decompositions.add(
    gates.CU3,
    lambda gate: [
        gates.RZ(1, (gate.parameters[2] - gate.parameters[1]) / 2.0),
        gates.CNOT(0, 1),
        gates.U3(
            1,
            -gate.parameters[0] / 2.0,
            0,
            -(gate.parameters[2] + gate.parameters[1]) / 2.0,
        ),
        gates.CNOT(0, 1),
        gates.U3(1, gate.parameters[0] / 2.0, gate.parameters[1], 0),
    ],
)
standard_decompositions.add(
    gates.SWAP, [gates.CNOT(0, 1), gates.CNOT(1, 0), gates.CNOT(0, 1)]
)
standard_decompositions.add(
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
standard_decompositions.add(
    gates.SiSWAP,
    [
        gates.SX(0),
        gates.SX(1),
        gates.RZ(0, math.pi / 2),
        gates.CNOT(0, 1),
        gates.SX(0),
        gates.RZ(1, -math.pi / 4),
        gates.RZ(0, -math.pi / 4),
        gates.SX(1),
        gates.SX(0),
        gates.RZ(0, math.pi / 2),
        gates.CNOT(0, 1),
        gates.SX(0),
    ],
)
standard_decompositions.add(
    gates.SiSWAPDG, [gate.dagger() for gate in gates.SiSWAP(0, 1).decompose()[::-1]]
)
standard_decompositions.add(
    gates.RXX,
    lambda gate: [
        gates.H(0),
        gates.H(1),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
        gates.H(1),
        gates.H(0),
    ],
)
standard_decompositions.add(
    gates.RYY,
    lambda gate: [
        gates.RX(0, math.pi / 2),
        gates.RX(1, math.pi / 2),
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
        gates.RX(1, -math.pi / 2),
        gates.RX(0, -math.pi / 2),
    ],
)
standard_decompositions.add(
    gates.RZZ,
    lambda gate: [
        gates.CNOT(0, 1),
        gates.RZ(1, gate.parameters[0]),
        gates.CNOT(0, 1),
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
        gates.RZ(1, -math.pi / 2),
        gates.S(0),
        gates.SX(1),
        gates.RZ(1, math.pi / 2),
        gates.CNOT(1, 0),
        gates.RY(0, -gate.parameters[0] / 2),
        gates.RY(1, -gate.parameters[0] / 2),
        gates.CNOT(1, 0),
        gates.SDG(0),
        gates.RZ(1, -math.pi / 2),
        gates.SXDG(1),
        gates.RZ(1, math.pi / 2),
    ],
)
standard_decompositions.add(
    gates.GIVENS, lambda gate: gates.RBS(0, 1, -gate.parameters[0]).decompose()
)
standard_decompositions.add(
    gates.FSWAP,
    [gates.X(1)] + gates.GIVENS(0, 1, math.pi / 2).decompose() + [gates.X(0)],
)
standard_decompositions.add(
    gates.ECR, [gates.S(0), gates.SX(1), gates.CNOT(0, 1), gates.X(0)]
)
standard_decompositions.add(
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
    gates.DEUTSCH,
    lambda gate: [
        gates.Z(2).controlled_by(0, 1),
        gates.Y(2).controlled_by(0, 1),
        gates.RX(2, 2 * gate.parameters[0]).controlled_by(0, 1),
        gates.X(2).controlled_by(0, 1),
    ],
)
standard_decompositions.add(
    gates.FanOut,
    lambda gate: [gates.CNOT(0, qub) for qub in range(1, len(gate.qubits))],
)
standard_decompositions.add(
    gates.GeneralizedRBS, lambda gate: _decomposition_generalized_RBS(gate)
)

method = "clifford_plus_t"
clifford_plus_t = GateDecompositions()
clifford_plus_t._set_precision_cliff_plus_t()
clifford_plus_t.add(gates.SDG, [gates.Y(0), gates.S(0), gates.Y(0)])
clifford_plus_t.add(gates.TDG, [gates.T(0), gates.Y(0), gates.S(0), gates.Y(0)])
clifford_plus_t.add(
    gates.RX,
    lambda gate: [gates.H(0)]
    + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
    + [gates.H(0)],
)
clifford_plus_t.add(
    gates.RY,
    lambda gate: [gates.SDG(0), gates.H(0)]
    + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
    + [gates.H(0), gates.S(0)],
)
clifford_plus_t.add(
    gates.RZ, lambda gate: clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
)
clifford_plus_t.add(
    gates.PRX,
    lambda gate: clifford_plus_t._rz_into_cliff_and_t(-gate.parameters[1] - math.pi / 2)
    + gates.RY(0, -gate.parameters[0]).decompose(method=method)
    + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] + math.pi / 2),
)
clifford_plus_t.add(
    gates.U1, lambda gate: clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0])
)
clifford_plus_t.add(
    gates.U2,
    lambda gate: clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] - math.pi)
    + [gates.SX(0)]
    + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0] + math.pi),
)
clifford_plus_t.add(
    gates.U3,
    lambda gate: clifford_plus_t._rz_into_cliff_and_t(gate.parameters[2])
    + [gates.SX(0)]
    + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[0] + math.pi)
    + [gates.SX(0)]
    + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] + math.pi),
)
clifford_plus_t.add(
    gates.PRX,
    lambda gate: clifford_plus_t._rz_into_cliff_and_t(-gate.parameters[1] - math.pi / 2)
    + gates.RY(0, -gate.parameters[0]).decompose(method=method)
    + clifford_plus_t._rz_into_cliff_and_t(gate.parameters[1] + math.pi / 2),
)
clifford_plus_t.add(gates.CY, [gates.SDG(1), gates.CNOT(0, 1), gates.S(1)])
clifford_plus_t.add(gates.CZ, [gates.H(1), gates.CNOT(0, 1), gates.H(1)])
clifford_plus_t.add(
    gates.CRX,
    lambda gate: gates.RX(1, gate.parameters[0] / 2.0).decompose(method=method)
    + [gates.H(1), gates.CNOT(0, 1)]
    + gates.RZ(1, -gate.parameters[0] / 2.0).decompose(method=method)
    + [gates.CNOT(0, 1), gates.H(1)],
)
clifford_plus_t.add(
    gates.CRY,
    lambda gate: gates.RY(1, gate.parameters[0] / 4).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.RY(1, -gate.parameters[0] / 2).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.RY(1, gate.parameters[0] / 4).decompose(method=method),
)
clifford_plus_t.add(
    gates.CRZ,
    lambda gate: gates.RZ(1, gate.parameters[0] / 2.0).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.RZ(1, -gate.parameters[0] / 2.0).decompose(method=method)
    + [gates.CNOT(0, 1)],
)
clifford_plus_t.add(
    gates.CU1,
    lambda gate: gates.RZ(0, gate.parameters[0] / 2.0).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.RZ(1, -gate.parameters[0] / 2.0).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.RZ(1, gate.parameters[0] / 2.0).decompose(method=method),
)
clifford_plus_t.add(
    gates.CU2,
    lambda gate: gates.RZ(1, (gate.parameters[1] - gate.parameters[0]) / 2.0).decompose(
        method=method
    )
    + [gates.CNOT(0, 1)]
    + gates.U3(
        1, -math.pi / 4, 0, -(gate.parameters[1] + gate.parameters[0]) / 2.0
    ).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.U3(1, math.pi / 4, gate.parameters[0], 0).decompose(method=method),
)
clifford_plus_t.add(
    gates.CU3,
    lambda gate: gates.RZ(1, (gate.parameters[2] - gate.parameters[1]) / 2.0).decompose(
        method=method
    )
    + [gates.CNOT(0, 1)]
    + gates.U3(
        1,
        -gate.parameters[0] / 2.0,
        0,
        -(gate.parameters[2] + gate.parameters[1]) / 2.0,
    ).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.U3(1, gate.parameters[0] / 2.0, gate.parameters[1], 0).decompose(
        method=method
    ),
)
clifford_plus_t.add(
    gates.CSX,
    [gates.H(1)] + gates.CU1(0, 1, math.pi / 2).decompose(method=method) + [gates.H(1)],
)
clifford_plus_t.add(
    gates.CSXDG,
    [gates.H(1)]
    + gates.CU1(0, 1, -math.pi / 2).decompose(method=method)
    + [gates.H(1)],
)
clifford_plus_t.add(gates.SWAP, [gates.CNOT(0, 1), gates.CNOT(1, 0), gates.CNOT(0, 1)])
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
    + gates.RZ(0, math.pi / 2).decompose(method=method)
    + [gates.CNOT(0, 1), gates.SX(0)]
    + gates.RZ(1, -math.pi / 4).decompose(method=method)
    + gates.RZ(0, -math.pi / 4).decompose(method=method)
    + [gates.SX(1), gates.SX(0)]
    + gates.RZ(0, math.pi / 2).decompose(method=method)
    + [gates.CNOT(0, 1), gates.SX(0)],
)
clifford_plus_t.add(
    gates.SiSWAPDG,
    [gate.dagger() for gate in reversed(gates.SiSWAP(0, 1).decompose(method=method))],
)
clifford_plus_t.add(
    gates.RXX,
    lambda gate: [gates.H(0), gates.H(1), gates.CNOT(0, 1)]
    + gates.RZ(1, gate.parameters[0]).decompose(method=method)
    + [gates.CNOT(0, 1), gates.H(1), gates.H(0)],
)
clifford_plus_t.add(
    gates.RYY,
    lambda gate: gates.RX(0, math.pi / 2).decompose(method=method)
    + gates.RX(1, math.pi / 2).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.RZ(1, gate.parameters[0]).decompose(method=method)
    + [gates.CNOT(0, 1)]
    + gates.RX(1, -math.pi / 2).decompose(method=method)
    + gates.RX(0, -math.pi / 2).decompose(method=method),
)
clifford_plus_t.add(
    gates.RZZ,
    lambda gate: [gates.CNOT(0, 1)]
    + gates.RZ(1, gate.parameters[0]).decompose(method=method)
    + [gates.CNOT(0, 1)],
)
clifford_plus_t.add(
    gates.RZX,
    lambda gate: [gates.H(1), gates.CNOT(0, 1)]
    + gates.RZ(1, gate.parameters[0]).decompose(method=method)
    + [gates.CNOT(0, 1), gates.H(1)],
)
clifford_plus_t.add(
    gates.RXXYY,
    lambda gate: gates.RZ(1, -math.pi / 2).decompose(method=method)
    + [gates.S(0), gates.SX(1)]
    + gates.RZ(1, math.pi / 2).decompose(method=method)
    + [gates.CNOT(1, 0)]
    + gates.RY(0, -gate.parameters[0] / 2).decompose(method=method)
    + gates.RY(1, -gate.parameters[0] / 2).decompose(method=method)
    + [gates.CNOT(1, 0), gates.SDG(0)]
    + gates.RZ(1, -math.pi / 2).decompose(method=method)
    + [gates.SXDG(1)]
    + gates.RZ(1, math.pi / 2).decompose(method=method),
)
clifford_plus_t.add(
    gates.GIVENS,
    lambda gate: gates.RBS(0, 1, -gate.parameters[0]).decompose(method=method),
)
clifford_plus_t.add(
    gates.FSWAP,
    [gates.X(1)]
    + gates.GIVENS(0, 1, math.pi / 2).decompose(method=method)
    + [gates.X(0)],
)
clifford_plus_t.add(gates.ECR, [gates.S(0), gates.SX(1), gates.CNOT(0, 1), gates.X(0)])
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
