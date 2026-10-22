import math
from enum import EnumMeta, Flag, auto
from functools import reduce
from operator import or_

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.gates import Gate
from qibo.gates.gates import is_nonzero_phase
from qibo.transpiler._exceptions import DecompositionError
from qibo.transpiler.decompositions.abstract import merge_global_phases
from qibo.transpiler.decompositions.cz_decompositions import cz_dec, native_cz_dec
from qibo.transpiler.decompositions.gpi2_decompositions import gpi2_dec
from qibo.transpiler.decompositions.iswap_decompositions import iswap_dec
from qibo.transpiler.decompositions.opt_decompositions import opt_dec
from qibo.transpiler.decompositions.standard_decompositions import (
    standard_decompositions,
)
from qibo.transpiler.decompositions.u3_decompositions import u3_dec
from qibo.transpiler.decompositions.unitary_decompositions import (
    _euler_frames,
    quantum_shannon_decomposition,
    single_qubit_decomposition,
    two_qubit_decomposition,
    u3_decomposition,
)

# Names of the two-qubit gates of :class:`qibo.transpiler.unroller.NativeGates`. When
# several entangling gates are native, the unroller uses the first one in this order.
_TWO_QUBIT_GATES = (
    "CZ",
    "iSWAP",
    "CNOT",
    "ECR",
    "CY",
    "CH",
    "CSX",
    "CSXDG",
    "FSWAP",
    "SiSWAP",
    "SiSWAPDG",
    "SYC",
    "SWAP",
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

    A native gate set should contain at least one entangling two-qubit gate, which is
    any of the two-qubit gates below except :class:`qibo.gates.gates.SWAP`, and
    single-qubit gates that can implement any
    single-qubit unitary. These are :class:`qibo.gates.gates.U3`,
    :class:`qibo.gates.gates.GPI2`, or a combination of the other single-qubit gates
    that has two rotations with free angles about orthogonal axes, see
    :func:`qibo.transpiler.decompositions.unitary_decompositions.single_qubit_decomposition`.
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
        - :class:`qibo.gates.gates.ECR`
        - :class:`qibo.gates.gates.CY`
        - :class:`qibo.gates.gates.CH`
        - :class:`qibo.gates.gates.CSX`
        - :class:`qibo.gates.gates.CSXDG`
        - :class:`qibo.gates.gates.FSWAP`
        - :class:`qibo.gates.gates.SiSWAP`
        - :class:`qibo.gates.gates.SiSWAPDG`
        - :class:`qibo.gates.gates.SYC`
        - :class:`qibo.gates.gates.SWAP`

    If several entangling gates are native, any other gate is decomposed into the first
    one among :class:`qibo.gates.gates.CZ`, :class:`qibo.gates.gates.iSWAP`,
    :class:`qibo.gates.gates.CNOT` and the gates above, in this order. A native
    two-qubit gate is never decomposed.
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
    ECR = auto()
    CY = auto()
    CH = auto()
    CSX = auto()
    CSXDG = auto()
    FSWAP = auto()
    SiSWAP = auto()
    SiSWAPDG = auto()
    SYC = auto()
    SWAP = auto()

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

        This holds when the set contains at least one entangling two-qubit gate, which
        is any of them except :class:`qibo.gates.gates.SWAP`, and single-qubit gates
        that implement any single-qubit unitary exactly: :class:`qibo.gates.gates.U3`,
        :class:`qibo.gates.gates.GPI2`, or gates with two rotations about orthogonal
        axes with free angles, see :attr:`single_qubit_gates`. Single-qubit unitaries
        together with any entangling two-qubit gate can implement every unitary on
        any number of connected qubits [1].

        References:
            1. J.-L. Brylinski and R. Brylinski,
            *Universal quantum gates*,
            in *Mathematics of Quantum Computation*, Chapman & Hall/CRC (2002).
        """
        return any(gate is not gates.SWAP for gate in self.two_qubit_gates) and bool(
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
            and native is not NativeGates.M
            and native.name not in _TWO_QUBIT_GATES
        )

    @property
    def two_qubit_gates(self) -> tuple[type[Gate], ...]:
        """Classes of the two-qubit gates in the native gates."""
        return tuple(
            getattr(gates, native.name)
            for native in NativeGates
            if native & self and native.name in _TWO_QUBIT_GATES
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
            gates, being left in that state. They are used as clean qubits if that lowers
            the number of native gates, and as dirty ones otherwise. It must be ``False``
            if the circuit is executed with another initial state. Defaults to ``False``.
        minimize_toffolis (bool, optional): If ``True``, a multi-controlled
            :class:`qibo.gates.X` with at least one auxiliary qubit is decomposed with
            the lowest number of Toffoli gates, which is useful if they are expensive.
            This is not the decomposition with the fewest native gates, which is the
            default one, and it needs more of them with dirty auxiliary qubits. With clean
            auxiliary qubits (see ``qubits_initially_zero``), it is shallower. It
            cannot be used together with ``minimize_depth``. Defaults to ``False``.
        minimize_depth (bool, optional): If ``True``, a multi-controlled
            :class:`qibo.gates.X` with at least one auxiliary qubit is decomposed with a
            depth that is logarithmic in the number of controls. It cannot be used together
            with ``minimize_toffolis``. See
            :func:`qibo.transpiler.multi_controlled_decomposition`.
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
                "The native gates are not universal. They must contain at least one "
                + "entangling two-qubit gate (any but SWAP), and single-qubit gates that "
                + "implement any single-qubit unitary: U3, GPI2, or rotations about two "
                + "orthogonal axes with free angles.",
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
            (:class:`qibo.models.circuit.Circuit`): Decomposed circuit. The decompositions
            that are equal to a gate only up to a global phase end with the
            :class:`qibo.gates.GlobalPhase` gate that corrects them. All of them, including
            the ones of the input circuit, are added modulo :math:`2 \\pi` into a single
            :class:`qibo.gates.GlobalPhase` gate at the end of the circuit, which is not
            added if the total phase is zero, see
            :attr:`qibo.models.circuit.Circuit.global_phase`.
        """
        backend = self.backend if backend is None else backend
        translated_circuit = Circuit(**circuit.init_kwargs)
        global_phase = 0.0
        # qubits that a previous gate acts on, so that they may not be in the state 0
        touched = set()
        for gate in circuit.queue:
            if len(gate.qubits) < self.min_qubits:
                translated = [gate]
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
                translated = translate_gate(
                    gate,
                    self.native_gates,
                    backend=backend,
                    free=free,
                    clean=clean,
                    minimize_toffolis=self.minimize_toffolis,
                    minimize_depth=self.minimize_depth,
                )
            # the global phase, if any, is the last gate of a translation
            if translated and isinstance(translated[-1], gates.GlobalPhase):
                global_phase = (global_phase + translated[-1].parameters[0]) % (
                    2 * math.pi
                )
                translated = translated[:-1]
            translated_circuit.add(translated)
            if not isinstance(gate, (gates.Align, gates.Barrier, gates.I)):
                touched.update(gate.qubits)
        if is_nonzero_phase(global_phase):
            translated_circuit.add(gates.GlobalPhase(global_phase, trainable=False))
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
            :func:`qibo.transpiler.multi_controlled_decomposition`.
            Defaults to ``False``.
        minimize_depth (bool, optional): If ``True``, a multi-controlled
            :class:`qibo.gates.X` is decomposed with a logarithmic depth when it has
            auxiliary qubits. Defaults to ``False``.

    Returns:
        list[:class:`qibo.gates.abstract.Gate`]: Native gates that decompose the input gate.
        If the decomposition is equal to the gate only up to a global phase, its last gate
        is the :class:`qibo.gates.GlobalPhase` gate that corrects it.

    Raises:
        DecompositionError: If the gate acts on more than two qubits and is not a
            controlled gate, a fused gate or a unitary that can be decomposed, or if
            the native gates are not sufficient.
    """
    backend = _check_backend(backend)

    if isinstance(
        gate,
        (gates.Align, gates.Barrier, gates.GlobalPhase, gates.I, gates.ResetChannel),
    ):
        return [gate]

    if isinstance(gate, gates.M):
        gate.basis_gates = len(gate.basis_gates) * [gates.Z]
        gate.basis = []
        return [gate]

    if isinstance(gate, gates.FusedGate) and len(gate.qubits) > 2:
        # A fused gate is a container, so its gates are translated one by one. The qubits
        # that it does not act on are free for all of them, as they do for the fused gate.
        translated = []
        for fused_gate in gate.gates:
            translated.extend(
                translate_gate(
                    fused_gate,
                    native_gates,
                    backend=backend,
                    free=free,
                    clean=clean,
                    minimize_toffolis=minimize_toffolis,
                    minimize_depth=minimize_depth,
                )
            )
        return merge_global_phases(translated)

    if len(gate.qubits) == 2 and gate.__class__ in native_gates.two_qubit_gates:
        return [gate]

    if len(gate.qubits) > 2 and (
        gate.is_controlled_by
        or (
            gate.__class__ not in cz_dec.decompositions
            and not isinstance(gate, gates.Unitary)
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
        return merge_global_phases(translated)

    if len(gate.qubits) == 1:
        return _translate_single_qubit_gates(gate, native_gates, backend)

    if isinstance(gate, gates.Unitary) and len(gate.qubits) > 2:
        # A unitary on more than two qubits is decomposed into gates on less qubits.
        # The controlled ones are decomposed above.
        translated = []
        for decomposed_gate in quantum_shannon_decomposition(
            gate.qubits, gate.matrix(backend), backend=backend
        ):
            translated.extend(translate_gate(decomposed_gate, native_gates, backend))
        return merge_global_phases(translated)

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
        return merge_global_phases(translated)

    decomposition_2q = _translate_two_qubit_gates(gate, native_gates, backend)

    final_decomposition = []
    for decomposed_2q_gate in decomposition_2q:
        if len(decomposed_2q_gate.qubits) == 1:
            final_decomposition += _translate_single_qubit_gates(
                decomposed_2q_gate, native_gates, backend
            )
        else:
            final_decomposition.append(decomposed_2q_gate)
    return merge_global_phases(final_decomposition)


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
    phase = []
    if gate.__class__ not in decomposer.decompositions:
        matrix = gate.matrix(backend)
        # the U3 gate equals the gate only up to half the phase of its determinant
        phase = [
            gates.GlobalPhase(
                (float(backend.angle(backend.det(matrix))) / 2) % (2 * math.pi),
                trainable=False,
            )
        ]
        gate = gates.U3(gate.qubits[0], *u3_decomposition(matrix, backend))

    return merge_global_phases(decomposer(gate, backend) + phase)


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

    # Any other entangling native gate implements the CZ gate, which the other gates
    # are decomposed into.
    native_gate = next(
        (native for native in native_gates.two_qubit_gates if native in native_cz_dec),
        None,
    )
    if native_gate is not None:
        translated = []
        for decomposed_gate in cz_dec(gate, backend):
            if isinstance(decomposed_gate, gates.CZ):
                qubits = dict(enumerate(decomposed_gate.qubits))
                translated.extend(
                    native.on_qubits(qubits) for native in native_cz_dec[native_gate]
                )
            else:
                translated.append(decomposed_gate)
        return translated

    raise_error(
        DecompositionError,
        "Use only CZ and/or iSWAP as native gates. CNOT is allowed in circuits"
        + " where the two-qubit gates are limited to CZ, CNOT, and SWAP.",
    )  # pragma: no cover
