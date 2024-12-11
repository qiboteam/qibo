from enum import EnumMeta, Flag, auto
from functools import reduce
from operator import or_

from qibo import gates
from qibo.backends import _check_backend
from qibo.config import log, raise_error
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


class FlagMeta(EnumMeta):
    """Metaclass for :class:`qibo.transpiler.unroller.NativeGates` that allows initialization with a list of gate name strings."""

    def __getitem__(cls, keys):
        if isinstance(keys, str):
            try:
                return super().__getitem__(keys)
            except KeyError:
                return super().__getitem__("NONE")
        return reduce(or_, [cls[key] for key in keys])  # pylint: disable=E1136


class NativeGates(Flag, metaclass=FlagMeta):
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

    NONE = 0
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
    def from_gate(cls, gate):
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
    """Decomposes a circuit to native gates."""

    def __init__(
        self,
        native_gates: NativeGates,
        backend=None,
    ):
        self.native_gates = native_gates
        self.backend = backend
        """Initializes the unroller.

        Args:
            native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates to use in the transpiled circuit.
            backend (:class:`qibo.backends.Backend`): Backend to use for gate matrix.
        """

    def __call__(self, circuit: Circuit):
        """Decomposes a circuit to native gates.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be decomposed.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): Decomposed circuit.
        """
        translated_circuit = Circuit(**circuit.init_kwargs)
        for gate in circuit.queue:
            translated_circuit.add(
                translate_gate(
                    gate,
                    self.native_gates,
                    backend=self.backend,
                )
            )
        return translated_circuit


def translate_gate(
    gate,
    native_gates: NativeGates,
    backend=None,
):
    """Maps gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Gate to be decomposed.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates supported by the hardware.
        backend (:class:`qibo.backends.Backend`): Backend to use for gate matrix.

    Returns:
        list: List of native gates that decompose the input gate.
    """
    backend = _check_backend(backend)

    if isinstance(gate, (gates.I, gates.Align)):
        return gate

    if isinstance(gate, gates.M):
        gate.basis_gates = len(gate.basis_gates) * [gates.Z]
        gate.basis = []
        return gate

    if len(gate.qubits) == 1:
        return _translate_single_qubit_gates(gate, native_gates, backend)

    decomposition_2q = _translate_two_qubit_gates(gate, native_gates, backend)
    final_decomposition = []
    for decomposed_2q_gate in decomposition_2q:
        if len(decomposed_2q_gate.qubits) == 1:
            final_decomposition += _translate_single_qubit_gates(
                decomposed_2q_gate, native_gates, backend
            )
        else:
            final_decomposition.append(decomposed_2q_gate)
    return final_decomposition


def _translate_single_qubit_gates(
    gate: gates.Gate, single_qubit_natives: NativeGates, backend
):
    """Helper method for :meth:`translate_gate`.

    Maps single qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Gate to be decomposed.
        single_qubit_natives (:class:`qibo.transpiler.unroller.NativeGates`): Single qubit native gates supported by the hardware.
        backend (:class:`qibo.backends.Backend`): Backend to use for gate matrix.

    Returns:
        list: List of native gates that decompose the input gate.
    """
    if NativeGates.U3 & single_qubit_natives:
        return u3_dec(gate, backend)

    if NativeGates.GPI2 & single_qubit_natives:
        return gpi2_dec(gate, backend)

    raise_error(DecompositionError, "Use U3 or GPI2 as single qubit native gates")


def _translate_two_qubit_gates(gate: gates.Gate, native_gates: NativeGates, backend):
    """Helper method for :meth:`translate_gate`.

    Maps two qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Gate to be decomposed.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates supported by the hardware.
        backend (:class:`qibo.backends.Backend`): Backend to use for gate matrix.

    Returns:
        list: List of native gates that decompose the input gate.
    """
    if (
        native_gates & (NativeGates.CZ | NativeGates.iSWAP)
    ) is NativeGates.CZ | NativeGates.iSWAP:
        # Check for a special optimized decomposition.
        if gate.__class__ in opt_dec.decompositions:
            return opt_dec(gate, backend)
        # Check if the gate has a CZ decomposition
        if gate.__class__ not in iswap_dec.decompositions:
            return cz_dec(gate, backend)
        # Check the decomposition with less 2 qubit gates.

        if cz_dec.count_2q(gate, backend) < iswap_dec.count_2q(gate, backend):
            return cz_dec(gate)
        if cz_dec.count_2q(gate, backend) > iswap_dec.count_2q(gate, backend):
            return iswap_dec(gate, backend)
        # If equal check the decomposition with less 1 qubit gates.
        # This is never used for now but may be useful for future generalization
        if cz_dec.count_1q(gate, backend) < iswap_dec.count_1q(
            gate, backend
        ):  # pragma: no cover
            return cz_dec(gate, backend)
        return iswap_dec(gate, backend)  # pragma: no cover

    if native_gates & NativeGates.CZ:
        return cz_dec(gate, backend)

    if native_gates & NativeGates.iSWAP:
        if gate.__class__ in iswap_dec.decompositions:
            return iswap_dec(gate, backend)

        # First decompose into CZ
        cz_decomposed = cz_dec(gate, backend)
        # Then CZ are decomposed into iSWAP
        iswap_decomposed = []
        for g in cz_decomposed:
            # Need recursive function as gates.Unitary is not in iswap_dec
            for g_translated in translate_gate(
                g, native_gates=native_gates, backend=backend
            ):
                iswap_decomposed.append(g_translated)
        return iswap_decomposed

    # For testing purposes
    # No CZ, iSWAP gates in the native gate set
    # Decompose CNOT, CZ, SWAP gates into CNOT gates
    if native_gates & NativeGates.CNOT:
        return cnot_dec_temp(gate, backend)

    raise_error(
        DecompositionError,
        "Use only CZ and/or iSWAP as native gates. CNOT is allowed in circuits where the two-qubit gates are limited to CZ, CNOT, and SWAP.",
    )  # pragma: no cover
