from qibo import gates
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.transpiler.abstract import NativeType, Unroller
from qibo.transpiler.decompositions import cz_dec, gpi2_dec, iswap_dec, opt_dec, u3_dec
from qibo.transpiler.exceptions import DecompositionError


# TODO: Make setting single-qubit native gates more flexible
class NativeGates(Unroller):
    """Translates a circuit to native gates.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to translate
            into native gates.
        single_qubit_natives (tuple): single qubit native gates.
        two_qubit_natives (:class:`qibo.transpiler.abstract.NativeType`): two-qubit native gates
            supported by the quantum hardware.
        single_qubit_natives (tuple): single qubit native gates.

    Returns:
        (:class:`qibo.models.circuit.Circuit`): equivalent circuit with native gates.
    """

    def __init__(
        self,
        two_qubit_natives: NativeType,
        single_qubit_natives=(gates.I, gates.Z, gates.RZ, gates.GPI2),
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
                    translate_gate(
                        gate,
                        self.two_qubit_natives,
                        single_qubit_natives=self.single_qubit_natives,
                    )
                )
            else:
                two_qubit_translated_circuit.add(gate)
        if self.translate_single_qubit:
            for gate in two_qubit_translated_circuit.queue:
                if len(gate.qubits) == 1:
                    translated_circuit.add(
                        translate_gate(
                            gate,
                            self.two_qubit_natives,
                            single_qubit_natives=self.single_qubit_natives,
                        )
                    )
                else:
                    translated_circuit.add(gate)
        else:
            translated_circuit = two_qubit_translated_circuit
        return translated_circuit

    def is_satisfied(self, circuit: Circuit):
        """Return True if a circuit is correctly decomposed into native gates, otherwise False."""
        try:
            assert_decomposition(
                circuit, self.two_qubit_natives, self.single_qubit_natives
            )
        except DecompositionError:
            return False
        return True


def assert_decomposition(
    circuit: Circuit,
    two_qubit_natives: NativeType,
    single_qubit_natives=(gates.I, gates.Z, gates.RZ, gates.GPI2),
):
    """Checks if a circuit has been correctly decmposed into native gates.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to check.
        two_qubit_natives (:class:`qibo.transpiler.abstract.NativeType`):
            two-qubit native gates supported by the quantum hardware.
        single_qubit_natives (tuple): single qubit native gates.
    """
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue
        if len(gate.qubits) == 1:
            if not isinstance(gate, single_qubit_natives):
                raise_error(
                    DecompositionError,
                    f"{gate.name} is not a single qubit native gate.",
                )
        elif len(gate.qubits) == 2:
            try:
                native_type_gate = NativeType.from_gate(gate)
                if not (native_type_gate in two_qubit_natives):
                    raise_error(
                        DecompositionError,
                        f"{gate.name} is not a two qubits native gate.",
                    )
            except ValueError:
                raise_error(
                    DecompositionError, f"{gate.name} is not a two qubits native gate."
                )
        else:
            raise_error(
                DecompositionError, f"{gate.name} acts on more than two qubits."
            )


def translate_gate(
    gate,
    two_qubit_natives: NativeType,
    single_qubit_natives=(gates.I, gates.Z, gates.RZ, gates.GPI2),
):
    """Maps gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to be decomposed.
        two_qubit_natives (:class:`qibo.transpiler.abstract.NativeType`):
            two-qubit native gates supported by the quantum hardware.
        single_qubit_natives (tuple): single qubit native gates.

    Returns:
        (list): List of native gates
    """
    if isinstance(gate, (gates.M, gates.I, gates.Align)):
        return gate

    if len(gate.qubits) == 1:
        if gates.U3 in single_qubit_natives:
            return u3_dec(gate)
        if gates.GPI2 in single_qubit_natives:
            return gpi2_dec(gate)
        else:
            raise DecompositionError("Use U3 or GPI2 as single qubit native gates")

    if two_qubit_natives is NativeType.CZ | NativeType.iSWAP:
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
    elif two_qubit_natives is NativeType.CZ:
        return cz_dec(gate)
    elif two_qubit_natives is NativeType.iSWAP:
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
        raise_error(DecompositionError, "Use only CZ and/or iSWAP as native gates")
