from enum import Flag, auto

from qibo import gates
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.transpiler._exceptions import DecompositionError
from qibo.transpiler.decompositions import (
    cnot_dec_temp,
    cz_dec,
    gpi2_dec,
    iswap_dec,
    opt_dec,
    u3_dec,
)


class NativeGates(Flag):
    """Define native gates supported by the unroller. A native gate set should contain at least
    one two-qubit gate (:class:`qibo.gates.gates.CZ` or :class:`qibo.gates.gates.iSWAP`),
    and at least one single-qubit gate
    (:class:`qibo.gates.gates.GPI2` or :class:`qibo.gates.gates.U3`).

    Possible gates are:
        - :class:`qibo.gates.gates.I`
        - :class:`qibo.gates.gates.Z`
        - :class:`qibo.gates.gates.RZ`
        - :class:`qibo.gates.gates.M`
        - :class:`qibo.gates.gates.GPI2`
        - :class:`qibo.gates.gates.U3`
        - :class:`qibo.gates.gates.CZ`
        - :class:`qibo.gates.gates.iSWAP`
        - :class:`qibo.gates.gates.CNOT`
    """

    I = auto()
    Z = auto()
    RZ = auto()
    M = auto()
    GPI2 = auto()
    U3 = auto()
    CZ = auto()
    iSWAP = auto()
    CNOT = auto()  # For testing purposes

    @classmethod
    def default(cls):
        """Return default native gates set."""
        return cls.CZ | cls.GPI2 | cls.I | cls.Z | cls.RZ | cls.M

    @classmethod
    def from_gatelist(cls, gatelist: list):
        """Create a NativeGates object containing all gates from a ``gatelist``."""
        natives = cls(0)
        for gate in gatelist:
            natives |= cls.from_gate(gate)
        return natives

    @classmethod
    def from_gate(cls, gate: gates.Gate):
        """Create a :class:`qibo.transpiler.unroller.NativeGates`
        object from a :class:`qibo.gates.gates.Gate`."""
        if isinstance(gate, gates.Gate):
            return cls.from_gate(gate.__class__)

        try:
            return getattr(cls, gate.__name__)
        except AttributeError:
            raise_error(ValueError, f"Gate {gate} cannot be used as native.")


# TODO: Make setting single-qubit native gates more flexible
class Unroller:
    """Translates a circuit to native gates.

    Args:
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): native gates to use
            in the transpiled circuit.

    Returns:
        (:class:`qibo.models.circuit.Circuit`): equivalent circuit with native gates.
    """

    def __init__(
        self,
        native_gates: NativeGates,
    ):
        self.native_gates = native_gates

    def __call__(self, circuit: Circuit):
        """Decomposes a circuit into native gates.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit model to decompose.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): equivalent circuit with native gates.
        """
        translated_circuit = Circuit(**circuit.init_kwargs)
        for gate in circuit.queue:
            translated_circuit.add(
                translate_gate(
                    gate,
                    self.native_gates,
                )
            )
        return translated_circuit


def assert_decomposition(
    circuit: Circuit,
    native_gates: NativeGates,
):
    """Checks if a circuit has been correctly decomposed into native gates.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to check.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`):
            native gates in the transpiled circuit.
    """
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue
        if len(gate.qubits) <= 2:
            try:
                native_type_gate = NativeGates.from_gate(gate)
                if not native_type_gate & native_gates:
                    raise_error(
                        DecompositionError,
                        f"{gate.name} is not a native gate.",
                    )
            except ValueError:
                raise_error(
                    DecompositionError,
                    f"{gate.name} is not a native gate.",
                )
        else:
            raise_error(
                DecompositionError, f"{gate.name} acts on more than two qubits."
            )


def translate_gate(
    gate,
    native_gates: NativeGates,
):
    """Maps gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to be decomposed.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): native gates
            to use in the decomposition.

    Returns:
        list: List of native gates that decompose the input gate.
    """
    if isinstance(gate, (gates.I, gates.Align)):
        return gate

    if isinstance(gate, gates.M):
        gate.basis_gates = len(gate.basis_gates) * [gates.Z]
        gate.basis = []
        return gate

    if len(gate.qubits) == 1:
        return _translate_single_qubit_gates(gate, native_gates)

    decomposition_2q = _translate_two_qubit_gates(gate, native_gates)
    final_decomposition = []
    for decomposed_2q_gate in decomposition_2q:
        if len(decomposed_2q_gate.qubits) == 1:
            final_decomposition += _translate_single_qubit_gates(
                decomposed_2q_gate, native_gates
            )
        else:
            final_decomposition.append(decomposed_2q_gate)
    return final_decomposition


def _translate_single_qubit_gates(gate: gates.Gate, single_qubit_natives: NativeGates):
    """Helper method for :meth:`translate_gate`.

    Maps single qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to be decomposed.
        single_qubit_natives (:class:`qibo.transpiler.unroller.NativeGates`): single
            qubit native gates.

    Returns:
        list: List of native gates that decompose the input gate.
    """
    if NativeGates.U3 & single_qubit_natives:
        return u3_dec(gate)

    if NativeGates.GPI2 & single_qubit_natives:
        return gpi2_dec(gate)

    raise_error(DecompositionError, "Use U3 or GPI2 as single qubit native gates")


def _translate_two_qubit_gates(gate: gates.Gate, native_gates: NativeGates):
    """Helper method for :meth:`translate_gate`.

    Maps two qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): gate to be decomposed.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): native gates
            supported by the quantum hardware.

    Returns:
        list: List of native gates that decompose the input gate.
    """
    if (
        native_gates & (NativeGates.CZ | NativeGates.iSWAP)
    ) is NativeGates.CZ | NativeGates.iSWAP:
        # Check for a special optimized decomposition.
        if gate.__class__ in opt_dec.decompositions:
            return opt_dec(gate)
        # Check if the gate has a CZ decomposition
        if not gate.__class__ in iswap_dec.decompositions:
            return cz_dec(gate)
        # Check the decomposition with less 2 qubit gates.

        if cz_dec.count_2q(gate) < iswap_dec.count_2q(gate):
            return cz_dec(gate)
        if cz_dec.count_2q(gate) > iswap_dec.count_2q(gate):
            return iswap_dec(gate)
        # If equal check the decomposition with less 1 qubit gates.
        # This is never used for now but may be useful for future generalization
        if cz_dec.count_1q(gate) < iswap_dec.count_1q(gate):  # pragma: no cover
            return cz_dec(gate)
        return iswap_dec(gate)  # pragma: no cover

    if native_gates & NativeGates.CZ:
        return cz_dec(gate)

    if native_gates & NativeGates.iSWAP:
        if gate.__class__ in iswap_dec.decompositions:
            return iswap_dec(gate)

        # First decompose into CZ
        cz_decomposed = cz_dec(gate)
        # Then CZ are decomposed into iSWAP
        iswap_decomposed = []
        for g in cz_decomposed:
            # Need recursive function as gates.Unitary is not in iswap_dec
            for g_translated in translate_gate(g, native_gates=native_gates):
                iswap_decomposed.append(g_translated)
        return iswap_decomposed

    # For testing purposes
    # No CZ, iSWAP gates in the native gate set
    # Decompose CNOT, CZ, SWAP gates into CNOT gates
    if native_gates & NativeGates.CNOT:
        return cnot_dec_temp(gate)

    raise_error(
        DecompositionError,
        "Use only CZ and/or iSWAP as native gates. CNOT is allowed in circuits where the two-qubit gates are limited to CZ, CNOT, and SWAP.",
    )  # pragma: no cover
