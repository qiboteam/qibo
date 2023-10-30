import numpy as np
from qibolab.native import NativeType

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.transpiler.abstract import Unroller
from qibo.transpiler.unitary_decompositions import (
    two_qubit_decomposition,
    u3_decomposition,
)

backend = NumpyBackend()


# TODO: Make setting single-qubit native gates more flexible
class NativeGates(Unroller):
    """Translates a circuit to native gates.

    Args:
        circuit (qibo.models.Circuit): circuit model to translate into native gates.
        single_qubit_natives (Tuple): single qubit native gates.
        two_qubit_natives (NativeType): two qubit native gates supported by the quantum hardware.

    Returns:
        translated_circuit (qibo.models.Circuit): equivalent circuit with native gates.
    """

    def __init__(
        self,
        two_qubit_natives: NativeType,
        single_qubit_natives=(gates.I, gates.Z, gates.RZ, gates.U3),
        translate_single_qubit: bool = True,
    ):
        self.two_qubit_natives = two_qubit_natives
        self.single_qubit_natives = single_qubit_natives
        self.translate_single_qubit = translate_single_qubit

    def __call__(self, circuit: Circuit):
        two_qubit_translated_circuit = circuit.__class__(circuit.nqubits)
        translated_circuit = circuit.__class__(circuit.nqubits)
        for gate in circuit.queue:
            if len(gate.qubits) > 1 or self.translate_single_qubit:
                two_qubit_translated_circuit.add(
                    translate_gate(gate, self.two_qubit_natives)
                )
            else:
                two_qubit_translated_circuit.add(gate)
        if self.translate_single_qubit:
            for gate in two_qubit_translated_circuit.queue:
                if len(gate.qubits) == 1:
                    translated_circuit.add(translate_gate(gate, self.two_qubit_natives))
                else:
                    translated_circuit.add(gate)
        else:
            translated_circuit = two_qubit_translated_circuit
        return translated_circuit


class DecompositionError(Exception):
    """A decomposition error is raised when, during transpiling, gates are not correctly decomposed in native gates"""


def assert_decomposition(
    circuit: Circuit,
    two_qubit_natives: NativeType,
    single_qubit_natives=(gates.I, gates.Z, gates.RZ, gates.U3),
):
    """Checks if a circuit has been correctly decmposed into native gates.

    Args:
        circuit (qibo.models.Circuit): circuit model to check.
    """
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue
        if len(gate.qubits) == 1:
            if not isinstance(gate, single_qubit_natives):
                raise DecompositionError(
                    f"{gate.name} is not a single qubit native gate."
                )
        elif len(gate.qubits) == 2:
            try:
                native_type_gate = NativeType.from_gate(gate)
                if not (native_type_gate in two_qubit_natives):
                    raise DecompositionError(
                        f"{gate.name} is not a two qubit native gate."
                    )
            except ValueError:
                raise DecompositionError(f"{gate.name} is not a two qubit native gate.")
        else:
            raise DecompositionError(f"{gate.name} acts on more than two qubits.")


def translate_gate(gate, native_gates: NativeType):
    """Maps Qibo gates to a hardware native implementation.

    Args:
        gate (qibo.gates.abstract.Gate): gate to be decomposed.
        native_gates (NativeType): two qubit native gates supported by the quantum hardware.

    Returns:
        List of native gates
    """
    if isinstance(gate, (gates.M, gates.I, gates.Align)):
        return gate

    if len(gate.qubits) == 1:
        return onequbit_dec(gate)

    if native_gates is NativeType.CZ | NativeType.iSWAP:
        # Check for a special optimized decomposition.
        if gate.__class__ in opt_dec.decompositions:
            return opt_dec(gate)
        # Check if the gate has a CZ decomposition
        if not gate.__class__ in iswap_dec.decompositions:
            return cz_dec(gate)
        # Check the decomposition with less 2 qubit gates.
        else:
            if cz_dec.count_2q(gate) < iswap_dec.count_2q(gate):
                return cz_dec(gate)
            elif cz_dec.count_2q(gate) > iswap_dec.count_2q(gate):
                return iswap_dec(gate)
            # If equal check the decomposition with less 1 qubit gates.
            # This is never used for now but may be useful for future generalization
            elif cz_dec.count_1q(gate) < iswap_dec.count_1q(gate):  # pragma: no cover
                return cz_dec(gate)
            else:  # pragma: no cover
                return iswap_dec(gate)
    elif native_gates is NativeType.CZ:
        return cz_dec(gate)
    elif native_gates is NativeType.iSWAP:
        if gate.__class__ in iswap_dec.decompositions:
            return iswap_dec(gate)
        else:
            # First decompose into CZ
            cz_decomposed = cz_dec(gate)
            # Then CZ are decomposed into iSWAP
            iswap_decomposed = []
            for g in cz_decomposed:
                # Need recursive function as gates.Unitary is not in iswap_dec
                for g_translated in translate_gate(g, NativeType.iSWAP):
                    iswap_decomposed.append(g_translated)
            return iswap_decomposed
    else:  # pragma: no cover
        raise_error(NotImplementedError, "Use only CZ and/or iSWAP as native gates")


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


onequbit_dec = GateDecompositions()
onequbit_dec.add(gates.H, [gates.U3(0, 7 * np.pi / 2, np.pi, 0)])
onequbit_dec.add(gates.X, [gates.U3(0, np.pi, 0, np.pi)])
onequbit_dec.add(gates.Y, [gates.U3(0, np.pi, 0, 0)])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.Z, [gates.Z(0)])
onequbit_dec.add(gates.S, [gates.RZ(0, np.pi / 2)])
onequbit_dec.add(gates.SDG, [gates.RZ(0, -np.pi / 2)])
onequbit_dec.add(gates.T, [gates.RZ(0, np.pi / 4)])
onequbit_dec.add(gates.TDG, [gates.RZ(0, -np.pi / 4)])
onequbit_dec.add(
    gates.RX, lambda gate: [gates.U3(0, gate.parameters[0], -np.pi / 2, np.pi / 2)]
)
onequbit_dec.add(gates.RY, lambda gate: [gates.U3(0, gate.parameters[0], 0, 0)])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.RZ, lambda gate: [gates.RZ(0, gate.parameters[0])])
# apply virtually by changing ``phase`` instead of using pulses
onequbit_dec.add(gates.GPI2, lambda gate: [gates.GPI2(0, gate.parameters[0])])
# implemented as single RX90 pulse
onequbit_dec.add(gates.U1, lambda gate: [gates.RZ(0, gate.parameters[0])])
onequbit_dec.add(
    gates.U2,
    lambda gate: [gates.U3(0, np.pi / 2, gate.parameters[0], gate.parameters[1])],
)
onequbit_dec.add(
    gates.U3,
    lambda gate: [
        gates.U3(0, gate.parameters[0], gate.parameters[1], gate.parameters[2])
    ],
)
onequbit_dec.add(
    gates.Unitary,
    lambda gate: [gates.U3(0, *u3_decomposition(gate.parameters[0]))],
)
onequbit_dec.add(
    gates.FusedGate,
    lambda gate: [gates.U3(0, *u3_decomposition(gate.matrix(backend)))],
)

# register the iSWAP decompositions
iswap_dec = GateDecompositions()
iswap_dec.add(
    gates.CNOT,
    [
        gates.U3(0, 7 * np.pi / 2, np.pi, 0),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi / 2),
    ],
)
iswap_dec.add(
    gates.CZ,
    [
        gates.U3(0, 7 * np.pi / 2, np.pi, 0),
        gates.U3(1, 7 * np.pi / 2, np.pi, 0),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi, 0, np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi),
        gates.iSWAP(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, -np.pi, -np.pi / 2),
        gates.U3(1, 7 * np.pi / 2, np.pi, 0),
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
        gates.U3(0, np.pi / 2, -np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, np.pi / 2, np.pi / 2),
        gates.CZ(0, 1),
        gates.U3(0, np.pi / 2, 0, -np.pi / 2),
        gates.U3(1, np.pi / 2, 0, np.pi / 2),
        gates.CZ(0, 1),
        gates.U3(0, np.pi / 2, np.pi / 2, -np.pi),
        gates.U3(1, np.pi / 2, 0, -np.pi),
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
        gates.U3(1, np.pi / 2, np.pi / 2, -np.pi),
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
    gates.Unitary, lambda gate: two_qubit_decomposition(0, 1, gate.parameters[0])
)
cz_dec.add(gates.fSim, lambda gate: two_qubit_decomposition(0, 1, gate.matrix(backend)))
cz_dec.add(
    gates.GeneralizedfSim,
    lambda gate: two_qubit_decomposition(0, 1, gate.matrix(backend)),
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
