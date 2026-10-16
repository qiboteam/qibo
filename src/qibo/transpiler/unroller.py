from enum import EnumMeta, Flag, auto
from functools import reduce
from operator import or_

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.gates import Gate
from qibo.transpiler._exceptions import DecompositionError
from qibo.transpiler.decompositions import (
    cz_dec,
    gpi2_dec,
    iswap_dec,
    opt_dec,
    standard_decompositions,
    u3_dec,
)
from qibo.transpiler.unitary_decompositions import (
    _euler_frames,
    single_qubit_decomposition,
    two_qubit_decomposition,
    u3_decomposition,
)


class FlagMeta(EnumMeta):
    """Metaclass for :class:`qibo.transpiler.unroller.NativeGates`
    that allows initialization with a list of gate name strings."""

    def __getitem__(cls, keys: str | list[str]) -> "NativeGates":
        if isinstance(keys, str):
            try:
                return super().__getitem__(keys)
            except KeyError:
                return super().__getitem__("NONE")
        return reduce(or_, [cls[key] for key in keys])


class NativeGates(Flag, metaclass=FlagMeta):
    """Define native gates supported by the unroller.

    A native gate set should contain at least one two-qubit gate
    (:class:`qibo.gates.gates.CZ`, :class:`qibo.gates.gates.iSWAP` or
    :class:`qibo.gates.gates.CNOT`), and single-qubit gates that can implement any
    single-qubit unitary. These are :class:`qibo.gates.gates.U3`,
    :class:`qibo.gates.gates.GPI2`, or a combination of the other single-qubit gates
    that has two rotations with free angles about orthogonal axes, see
    :func:`qibo.transpiler.unitary_decompositions.single_qubit_decomposition`.
    For example, :class:`qibo.gates.gates.RX` and :class:`qibo.gates.gates.RZ`, or
    :class:`qibo.gates.gates.RZ` and :class:`qibo.gates.gates.H`.
    :class:`qibo.gates.gates.Z` and :class:`qibo.gates.gates.RZ` are virtual gates,
    which are assumed to be available together with :class:`qibo.gates.gates.GPI2`
    and :class:`qibo.gates.gates.U3`.

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
        - :class:`qibo.gates.gates.X`
        - :class:`qibo.gates.gates.Y`
        - :class:`qibo.gates.gates.H`
        - :class:`qibo.gates.gates.S`
        - :class:`qibo.gates.gates.SDG`
        - :class:`qibo.gates.gates.T`
        - :class:`qibo.gates.gates.TDG`
        - :class:`qibo.gates.gates.SX`
        - :class:`qibo.gates.gates.SXDG`
        - :class:`qibo.gates.gates.RX`
        - :class:`qibo.gates.gates.RY`
        - :class:`qibo.gates.gates.PRX`
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
    X = auto()
    Y = auto()
    H = auto()
    S = auto()
    SDG = auto()
    T = auto()
    TDG = auto()
    SX = auto()
    SXDG = auto()
    RX = auto()
    RY = auto()
    PRX = auto()

    @classmethod
    def default(cls) -> "NativeGates":
        """Return default native gates set."""
        return cls.CZ | cls.GPI2 | cls.I | cls.Z | cls.RZ | cls.M

    @classmethod
    def from_gatelist(cls, gatelist: list[Gate]) -> "NativeGates":
        """Create a NativeGates object containing all gates from a ``gatelist``."""
        natives = cls(0)
        for gate in gatelist:
            natives |= cls.from_gate(gate)
        return natives

    @classmethod
    def from_gate(cls, gate: Gate | type[Gate]) -> "NativeGates":
        """Create a :class:`qibo.transpiler.unroller.NativeGates`
        object from a :class:`qibo.gates.gates.Gate`."""
        if isinstance(gate, Gate):
            return cls.from_gate(gate.__class__)

        try:
            return getattr(cls, gate.__name__)
        except AttributeError:
            raise_error(ValueError, f"Gate {gate} cannot be used as native.")

    @property
    def is_universal(self) -> bool:
        """Whether the native gates can implement any two-qubit unitary.

        This holds when the set contains at least one entangling gate among
        :class:`qibo.gates.gates.CZ`, :class:`qibo.gates.gates.iSWAP` and
        :class:`qibo.gates.gates.CNOT`, and single-qubit gates that implement any
        single-qubit unitary exactly: :class:`qibo.gates.gates.U3`,
        :class:`qibo.gates.gates.GPI2`, or gates with two rotations about orthogonal
        axes with free angles, see :attr:`single_qubit_gates`. Single-qubit unitaries
        together with any entangling two-qubit gate can implement every unitary on
        any number of connected qubits [1].

        References:
            1. J.-L. Brylinski and R. Brylinski,
            *Universal quantum gates*,
            in *Mathematics of Quantum Computation*, Chapman & Hall/CRC (2002).
        """
        return bool(
            self & (NativeGates.CZ | NativeGates.iSWAP | NativeGates.CNOT)
        ) and bool(
            self & (NativeGates.GPI2 | NativeGates.U3)
            or _euler_frames(self.single_qubit_gates) is not None
        )

    @property
    def single_qubit_gates(self) -> tuple[type[Gate], ...]:
        """Classes of the single-qubit gates in the native gates."""
        return tuple(
            getattr(gates, native.name)
            for native in NativeGates
            if native & self
            and native
            not in (
                NativeGates.M,
                NativeGates.CZ,
                NativeGates.iSWAP,
                NativeGates.CNOT,
            )
        )


class Unroller:
    """Decomposes a circuit to native gates.

    Args:
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`):
            Native gates to use in the transpiled circuit. They must be universal,
            see :attr:`qibo.transpiler.unroller.NativeGates.is_universal`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use for
            gate matrix. Defaults to ``None``.
        use_dirty_ancillas (bool, optional): If ``True``, the qubits of the circuit
            that a multi-controlled gate does not act on are used as dirty auxiliary
            qubits in its decomposition, which makes the cost of a multi-controlled
            :class:`qibo.gates.X` linear in the number of controls. They can be in
            any state and are left unchanged. Not using dirty auxiliary qubits makes
            the gate count quadratic in the number of controls. Defaults to ``False``.
        min_qubits (int, optional): Only gates acting on at least ``min_qubits`` qubits
            are decomposed, the others are left unchanged. For example,
            ``min_qubits=3`` decomposes only gates acting on three or more qubits,
            which is useful to make a circuit executable by a router before routing.
            Defaults to :math:`1`, which decomposes every gate.
        qubits_initially_zero (bool, optional): If ``True``, the circuit is assumed to
            start in the state :math:`\\ket{0}^{\\otimes n}`. The qubits on which no previous
            gate of the circuit acts are then known to be in the state :math:`\\ket{0}`, and
            are used as clean auxiliary qubits in the decomposition of multi-controlled
            gates, being left in that state. By default, clean auxiliary qubits are used as
            dirty ones, so they only lower the cost if ``minimize_toffolis`` or
            ``minimize_depth`` is ``True``. It must be ``False`` if the circuit is executed
            with another initial state. Defaults to ``False``.
        minimize_toffolis (bool, optional): If ``True``, a multi-controlled
            :class:`qibo.gates.X` with at least one auxiliary qubit is decomposed with
            the lowest number of Toffoli gates. With clean auxiliary qubits, see
            ``qubits_initially_zero``, this makes the decomposition shallower and, for many
            controls, also shorter in number of native gates. With dirty auxiliary
            qubits only, it needs more native gates than the default decomposition. It
            cannot be used together with ``minimize_depth``. Defaults to ``False``.
        minimize_depth (bool, optional): If ``True``, a multi-controlled
            :class:`qibo.gates.X` with at least one auxiliary qubit is decomposed with a
            depth that is logarithmic in the number of controls. It cannot be used together
            with ``minimize_toffolis``. See
            :func:`qibo.transpiler.multicontrolled_decompositions.multi_controlled_decomposition`.
            Defaults to ``False``.
    """

    def __init__(
        self,
        native_gates: NativeGates,
        backend: Backend | None = None,
        use_dirty_ancillas: bool = False,
        min_qubits: int = 1,
        qubits_initially_zero: bool = False,
        minimize_toffolis: bool = False,
        minimize_depth: bool = False,
    ):
        if not native_gates.is_universal:
            raise_error(
                DecompositionError,
                "The native gates are not universal. They must contain at least one of "
                + "CZ, iSWAP or CNOT, and single-qubit gates that implement any "
                + "single-qubit unitary: U3, GPI2, or rotations about two orthogonal "
                + "axes with free angles.",
            )
        if minimize_toffolis and minimize_depth:
            raise_error(
                ValueError,
                "``minimize_toffolis`` and ``minimize_depth`` cannot be used together.",
            )
        self.native_gates = native_gates
        self.backend = backend
        self.use_dirty_ancillas = use_dirty_ancillas
        self.min_qubits = min_qubits
        self.qubits_initially_zero = qubits_initially_zero
        self.minimize_toffolis = minimize_toffolis
        self.minimize_depth = minimize_depth

    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> Circuit:
        """Decomposes a circuit to native gates.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be decomposed.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use for
                gate matrix. If ``None``, it uses the backend of the unroller. Defaults
                to ``None``.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): Decomposed circuit.
        """
        backend = self.backend if backend is None else backend
        translated_circuit = Circuit(**circuit.init_kwargs)
        # qubits that a previous gate acts on, so that they may not be in the state 0
        touched = set()
        for gate in circuit.queue:
            if len(gate.qubits) < self.min_qubits:
                translated_circuit.add(gate)
            else:
                idle = (
                    tuple(q for q in range(circuit.nqubits) if q not in gate.qubits)
                    if gate.is_controlled_by
                    else ()
                )
                clean = (
                    tuple(q for q in idle if q not in touched)
                    if self.qubits_initially_zero
                    else ()
                )
                free = (
                    tuple(q for q in idle if q not in clean)
                    if self.use_dirty_ancillas
                    else ()
                )
                translated_circuit.add(
                    translate_gate(
                        gate,
                        self.native_gates,
                        backend=backend,
                        free=free,
                        clean=clean,
                        minimize_toffolis=self.minimize_toffolis,
                        minimize_depth=self.minimize_depth,
                    )
                )
            if not isinstance(gate, (gates.Align, gates.Barrier, gates.I)):
                touched.update(gate.qubits)
        return translated_circuit


def translate_gate(
    gate: Gate,
    native_gates: NativeGates,
    backend: Backend | None = None,
    free: tuple[int, ...] = (),
    clean: tuple[int, ...] = (),
    minimize_toffolis: bool = False,
    minimize_depth: bool = False,
) -> list[Gate]:
    """Maps gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Gate to be decomposed.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`):
            Native gates supported by the hardware.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use
            for gate matrix. If ``None``, defaults to the global backend.
            Defaults to ``None``.
        free (tuple[int, ...], optional): Ids of free qubits that can be used as
            dirty auxiliary qubits to decompose multi-controlled gates. Defaults
            to an empty tuple, which uses no auxiliary qubits.
        clean (tuple[int, ...], optional): Ids of free qubits that are known to be in
            the state :math:`\\ket{0}` and are left in that state, which can be used as
            clean auxiliary qubits to decompose multi-controlled gates. They must not
            be in ``free``. Defaults to an empty tuple.
        minimize_toffolis (bool, optional): If ``True``, a multi-controlled
            :class:`qibo.gates.X` is decomposed with the lowest number of Toffoli gates
            when it has auxiliary qubits, see
            :func:`qibo.transpiler.multicontrolled_decompositions.multi_controlled_decomposition`.
            Defaults to ``False``.
        minimize_depth (bool, optional): If ``True``, a multi-controlled
            :class:`qibo.gates.X` is decomposed with a logarithmic depth when it has
            auxiliary qubits. Defaults to ``False``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Native gates that decompose the input gate.

    Raises:
        DecompositionError: If the gate acts on more than two qubits and is not a
            controlled gate that can be decomposed, or if the native gates are not
            sufficient.
    """
    backend = _check_backend(backend)

    if isinstance(gate, (gates.Align, gates.Barrier, gates.I, gates.ResetChannel)):
        return [gate]

    if isinstance(gate, gates.M):
        gate.basis_gates = len(gate.basis_gates) * [gates.Z]
        gate.basis = []
        return [gate]

    if len(gate.qubits) > 2 and (
        gate.is_controlled_by
        or (
            gate.__class__ not in cz_dec.decompositions
            and not isinstance(gate, (gates.FusedGate, gates.Unitary))
        )
    ):
        # The gate is decomposed by itself into gates acting on less qubits, which
        # happens for the controlled gates and for the gates with no registered
        # decomposition, like the ones that other gates are built from.
        try:
            decomposed_gates = gate.decompose(
                *free,
                clean=clean,
                minimize_toffolis=minimize_toffolis,
                minimize_depth=minimize_depth,
            )
        except RecursionError:
            decomposed_gates = [gate]
        if any(
            decomposed_gate.is_controlled_by
            and len(decomposed_gate.qubits) == len(gate.qubits)
            for decomposed_gate in decomposed_gates
        ):
            raise_error(
                DecompositionError,
                f"Cannot decompose {gate.name} controlled by {len(gate.control_qubits)} "
                + f"qubit(s) with {len(gate.target_qubits)} target qubit(s).",
            )
        if not gate.is_controlled_by and any(
            len(decomposed_gate.qubits) >= len(gate.qubits)
            for decomposed_gate in decomposed_gates
        ):
            raise_error(
                DecompositionError,
                f"Cannot decompose {gate.name} acting on {len(gate.qubits)} qubits.",
            )
        translated = []
        for decomposed_gate in decomposed_gates:
            translated.extend(translate_gate(decomposed_gate, native_gates, backend))
        return translated

    if len(gate.qubits) == 1:
        return _translate_single_qubit_gates(gate, native_gates, backend)

    if len(gate.qubits) > 2 and (
        isinstance(gate, (gates.FusedGate, gates.Unitary))
        or gate.__class__ not in cz_dec.decompositions
    ):
        raise_error(
            DecompositionError,
            f"Cannot decompose {gate.name} acting on {len(gate.qubits)} qubits.",
        )

    if gate.__class__ not in cz_dec.decompositions or (
        gate.is_controlled_by and isinstance(gate, (gates.FusedGate, gates.Unitary))
    ):
        # The gate has no registered decomposition, or it is a controlled unitary.
        # Its matrix is decomposed instead.
        circuit = Circuit(2)
        circuit.add(gate.on_qubits(dict(zip(gate.qubits, (0, 1)))))
        translated = []
        for decomposed_gate in two_qubit_decomposition(
            *gate.qubits, circuit.unitary(backend), backend=backend
        ):
            translated.extend(translate_gate(decomposed_gate, native_gates, backend))
        return translated

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
    gate: Gate, single_qubit_natives: NativeGates, backend: Backend
) -> list[Gate]:
    """Helper method for :meth:`translate_gate`.

    Maps single-qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Gate to be decomposed.
        single_qubit_natives (:class:`qibo.transpiler.unroller.NativeGates`):
            Single qubit native gates supported by the hardware.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use
            for gate matrix. If ``None``, defaults to the global backend.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Native gates that decompose the input gate.
    """
    if not (NativeGates.U3 | NativeGates.GPI2) & single_qubit_natives:
        if gate.__class__ in single_qubit_natives.single_qubit_gates:
            return [gate]
        return single_qubit_decomposition(
            gate.matrix(backend),
            gate.qubits[0],
            single_qubit_natives.single_qubit_gates,
            backend=backend,
        )

    decomposer = gpi2_dec if NativeGates.GPI2 & single_qubit_natives else u3_dec
    if gate.__class__ not in decomposer.decompositions:
        gate = gates.U3(
            gate.qubits[0], *u3_decomposition(gate.matrix(backend), backend)
        )

    return decomposer(gate, backend)


def _translate_two_qubit_gates(
    gate: Gate, native_gates: NativeGates, backend: Backend
) -> list[Gate]:
    """Helper method for :meth:`translate_gate`.

    Maps two-qubit gates to a hardware-native implementation.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Gate to be decomposed.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates
            supported by the hardware.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use
            for gate matrix. If ``None``, defaults to the global backend.
            Defaults to ``None``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Native gates that decompose the input gate.
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
            return cz_dec(gate, backend)

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
            iswap_decomposed.extend(
                translate_gate(g, native_gates=native_gates, backend=backend)
            )
        return iswap_decomposed

    # For testing purposes
    # No CZ, iSWAP gates in the native gate set
    # Decompose CNOT, CZ, SWAP gates into CNOT gates
    if native_gates & NativeGates.CNOT:
        if gate.__class__ in standard_decompositions.decompositions:
            return standard_decompositions(gate, backend)

        # First decompose into CZ, which are then decomposed into CNOT
        cz_decomposed = cz_dec(gate, backend)
        cnot_decomposed = []
        for g in cz_decomposed:
            cnot_decomposed.extend(
                translate_gate(g, native_gates=native_gates, backend=backend)
            )
        return cnot_decomposed

    raise_error(
        DecompositionError,
        "Use only CZ and/or iSWAP as native gates. CNOT is allowed in circuits"
        + " where the two-qubit gates are limited to CZ, CNOT, and SWAP.",
    )  # pragma: no cover
