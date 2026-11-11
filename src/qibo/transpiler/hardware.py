import math
from dataclasses import InitVar, dataclass, field, replace
from functools import reduce
from typing import Self

import networkx as nx

from qibo import gates
from qibo.backends import get_backend
from qibo.config import log, raise_error
from qibo.noise import (
    DepolarizingError,
    NoiseModel,
    ReadoutError,
    ThermalRelaxationError,
    _Conditions,
)
from qibo.quantum_info import average_gate_fidelity, to_liouville
from qibo.transpiler.unroller import NativeGates


@dataclass(eq=False, frozen=True)
class HardwareModel:
    """Description of a quantum processor.

    Includes its qubits, couplings, native gates, gate durations, and the errors of the gates.
    The noise model of the processor follows from them, so it cannot get out of sync. The
    attributes cannot be reassigned, and the dictionaries are copies of the arguments, which
    are not meant to be changed afterwards.

    Example:
        .. testcode::

            from qibo.transpiler import HardwareModel

            # content of the ``calibration.json`` file that ``qibocal`` exports
            calibration = {
                "single_qubits": {
                    "0": {
                        "t1": [60000.0, 500.0],
                        "t2": [45000.0, 500.0],
                        "rb_fidelity": [0.9985, 0.0001],
                        "readout": {"fidelity": 0.92},
                    },
                    "1": {
                        "t1": [50000.0, 500.0],
                        "t2": [40000.0, 500.0],
                        "rb_fidelity": [0.9980, 0.0001],
                        "readout": {"fidelity": 0.90},
                    },
                },
                "two_qubits": {"0-1": {"cz_fidelity": [0.99, 0.001]}},
            }
            # gate durations in ns
            hardware = HardwareModel.from_calibration(calibration, {"1q": 40.0, "2q": 60.0})

            # error and duration of the native gates on given qubits
            single_qubit_error = hardware.error((0,))
            two_qubit_error = hardware.error((0, 1))
            two_qubit_duration = hardware.duration((0, 1))

            # noise model that follows from the errors, to simulate the processor
            noise_model = hardware.noise_model

    Args:
        qubits (tuple): Qubit names, as in the wire names of a transpiled circuit.
            The noise model refers to the qubits by their position in this tuple. The names
            cannot be repeated.
        connectivity (:class:`networkx.Graph` or list[tuple]): Couplings, as a graph with the
            qubit names as nodes or as a list of pairs of qubit names, *e.g.*
            ``[(0, 1), (1, "A2")]``, which is how ``qibolab`` lists them. They are stored as
            a graph that also has the qubits without couplings. A directed graph has an edge
            ``(a, b)`` for a gate that only acts with its first qubit on ``a`` and its second
            qubit on ``b``, as in :func:`qibo.transpiler.pipeline.generate_preset_passes`.
            A qubit cannot be coupled to itself.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native
            gates of every qubit. Defaults to ``NativeGates.default()``.
        durations (dict, optional): Gate durations in ns. The keys ``"1q"`` and ``"2q"`` set
            the default of each kind, and a pair of qubit names that are coupled sets that of
            the coupling, in any order. The durations must be positive. It must have the
            duration of every gate whose qubits have their :math:`T_{1}` and :math:`T_{2}` in
            ``coherence``, because of the thermal relaxation of the noise model. Defaults to an
            empty dictionary.
        errors (dict, optional): Error of the native gates, as one minus their average gate
            fidelity. The key ``"1q"`` has the error of the single-qubit gates, except for the
            rotations around Z, which are virtual and have no error, with qubit names as
            keys. The key ``"2q"`` has the error of ``CZ``, with the pairs of qubit names of
            the couplings as keys, in the direction of a directed connectivity. The key
            ``"readout"`` has the probability that a measurement of a qubit gives the opposite
            result, with qubit names as keys. Gates without an error have none. Defaults to an
            empty dictionary.
        coherence (dict, optional): Coherence times of the qubits in ns, as pairs
            :math:`(T_{1}, T_{2})` with qubit names as keys, *e.g.* ``{0: (60000.0, 45000.0)}``.
            They must be positive. A :math:`T_{2}` larger than :math:`2 T_{1}` is not
            physical, so the noise model uses :math:`2 T_{1}` instead and warns.
            Defaults to an empty dictionary.
        symmetric (dict, optional): Only for a list of couplings. It tells if the two-qubit
            gate of a coupling is symmetric, as ``symmetric`` of the natives of ``qibolab``, with
            pairs as keys, *e.g.* ``{(0, 1): False}``. The pairs must be couplings. The
            couplings that are not symmetric are only in the direction of their pair, and the
            connectivity is then a directed graph. The pairs that are not in ``symmetric`` are
            symmetric. Defaults to ``None``, so all of them are symmetric.

    Attributes:
        noise_model (:class:`qibo.noise.NoiseModel`): Errors that follow the native gates, on
            the qubit positions. After each gate, there is the thermal relaxation of its
            qubits during the gate, if their :math:`T_{1}` and :math:`T_{2}` are in
            ``coherence``, and a depolarizing error that completes the error of the gate.
            Without them, the depolarizing error is the whole error. The errors of a coupling
            are only in the directions of the connectivity, and only ``CZ`` has them. Each
            measurement has its readout error, before the measurement. A device without
            errors has an empty noise model.

    Raises:
        ValueError: If an argument is not valid, *e.g.* a repeated qubit name, a coupling
            of a qubit with itself, or a key of ``durations`` or ``errors`` that is not a
            qubit or a coupling, or if ``durations`` has no duration that the thermal
            relaxation needs, or if the qubits fully relax during a gate, which happens if
            ``durations`` and ``coherence`` have different units.
    """

    qubits: tuple[int | str, ...]
    connectivity: nx.Graph
    native_gates: NativeGates = field(default_factory=NativeGates.default)
    durations: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    coherence: dict = field(default_factory=dict)
    symmetric: InitVar[dict | None] = None
    noise_model: NoiseModel = field(init=False, repr=False)

    def __post_init__(self, symmetric: dict | None):
        qubits = tuple(self.qubits)
        if len(set(qubits)) != len(qubits):
            raise_error(ValueError, "``qubits`` cannot have repeated names.")

        if isinstance(self.connectivity, nx.Graph):
            if symmetric is not None:
                raise_error(
                    ValueError,
                    "``symmetric`` is only for a list of couplings, not a graph.",
                )
            graph = self.connectivity.copy()
        else:
            pairs = [tuple(pair) for pair in self.connectivity]
            if any(len(pair) != 2 for pair in pairs):
                raise_error(ValueError, "Each coupling must be a pair of qubit names.")
            symmetric = {} if symmetric is None else symmetric
            if unknown := [
                pair
                for pair in symmetric
                if pair not in pairs and pair[::-1] not in pairs
            ]:
                raise_error(
                    ValueError,
                    f"``symmetric`` has pairs that are not couplings: {unknown}.",
                )
            flags = [
                symmetric.get(pair, symmetric.get(pair[::-1], True)) for pair in pairs
            ]
            graph = nx.Graph() if all(flags) else nx.DiGraph()
            for pair, flag in zip(pairs, flags):
                graph.add_edge(*pair)
                if flag:
                    graph.add_edge(*pair[::-1])
        if unknown := set(graph.nodes) - set(qubits):
            raise_error(
                ValueError,
                f"The couplings have qubits that are not in ``qubits``: {unknown}.",
            )
        if nx.number_of_selfloops(graph) > 0:
            raise_error(ValueError, "A qubit cannot be coupled to itself.")
        graph.add_nodes_from(qubits)
        couplings = {*graph.edges, *(edge[::-1] for edge in graph.edges)}
        # the gate of a coupling of a directed graph only acts in its direction
        oriented = {*graph.edges} if graph.is_directed() else couplings

        durations = dict(self.durations)
        for key, value in durations.items():
            if key not in ("1q", "2q") and key not in couplings:
                raise_error(
                    ValueError,
                    f"The key {key!r} of ``durations`` is not ``'1q'``, ``'2q'`` "
                    + "or a coupling.",
                )
            if not value > 0:
                raise_error(ValueError, "The gate durations must be positive.")

        errors = {kind: dict(values) for kind, values in self.errors.items()}
        if unknown := set(errors) - {"1q", "2q", "readout"}:
            raise_error(
                ValueError,
                f"The keys of ``errors`` must be '1q', '2q' or 'readout', not {unknown}.",
            )
        for kind, values in errors.items():
            for key, value in values.items():
                if key not in (oriented if kind == "2q" else qubits):
                    raise_error(
                        ValueError,
                        f"The key {key!r} of ``errors['{kind}']`` is not "
                        + f"a {'coupling' if kind == '2q' else 'qubit'}.",
                    )
                if not 0 <= value <= 1:
                    raise_error(ValueError, "The errors must be between 0 and 1.")

        coherence = {qubit: tuple(times) for qubit, times in self.coherence.items()}
        for qubit, times in coherence.items():
            if qubit not in qubits:
                raise_error(
                    ValueError, f"The key {qubit!r} of ``coherence`` is not a qubit."
                )
            if len(times) != 2 or not all(time > 0 for time in times):
                raise_error(
                    ValueError,
                    "The coherence times must be a pair of positive numbers.",
                )

        # the dataclass is frozen, so its attributes are set through ``object``
        for name, value in (
            ("qubits", qubits),
            ("connectivity", graph),
            ("durations", durations),
            ("errors", errors),
            ("coherence", coherence),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "noise_model", self._build_noise_model())

    def duration(self, qubits: tuple) -> float:
        """Duration of the native gates on some qubits.

        Args:
            qubits (tuple): Names of the qubits of the gate. For a two-qubit gate, the order
                does not matter.

        Returns:
            float: Duration in ns, which is the one of the coupling if ``durations`` has it,
            and the default of its kind otherwise, *i.e.* ``"1q"`` or ``"2q"``.

        Raises:
            ValueError: If ``durations`` has neither of them.
        """
        qubits = tuple(qubits)
        for key in (qubits, qubits[::-1], f"{len(qubits)}q"):
            if key in self.durations:
                return self.durations[key]

        raise_error(
            ValueError, f"``durations`` has no duration for the gates on {qubits}."
        )

    def error(self, qubits: tuple) -> float | None:
        """Error of the native gates on some qubits.

        Args:
            qubits (tuple): Names of the qubits of the gate. For a two-qubit gate, the order
                does not matter.

        Returns:
            float | None: One minus the average gate fidelity of the gate, or ``None`` if
            ``errors`` has no error for it, which is a gate without error. The error of a
            single-qubit gate is the one of the rotations that are not virtual.
        """
        qubits = tuple(qubits)
        if len(qubits) == 1:
            return self.errors.get("1q", {}).get(qubits[0])

        two_qubits = self.errors.get("2q", {})
        return two_qubits.get(qubits, two_qubits.get(qubits[::-1]))

    @classmethod
    def from_calibration(
        cls,
        calibration: dict,
        durations: dict,
        connectivity: nx.Graph | list[tuple] | None = None,
        native_gates: NativeGates | None = None,
        pulses_per_clifford: float = 1.875,
        symmetric: dict | None = None,
    ) -> Self:
        """Build the model from the content of the ``calibration.json`` that ``qibocal``
        exports, for example loaded with ``json.load``.

        The error of each native gate is the one measured with randomized benchmarking,
        and the readout error is half of the infidelity of the readout. Gates without data
        have no error. The error of a gate is split into the thermal relaxation of its
        qubits during the gate, from ``t1`` and ``t2``, and a depolarizing error with the
        rest, see :class:`HardwareModel`. The split is exact, because it uses the average
        gate fidelity of the thermal relaxation, with
        :func:`qibo.quantum_info.average_gate_fidelity`, which is not the sum of the
        infidelities of the qubits for a two-qubit gate [1].

        Args:
            calibration (dict): Content of the ``calibration.json`` file.
            durations (dict): Gate durations in ns, as in :class:`HardwareModel`.
            connectivity (:class:`networkx.Graph` or list[tuple], optional): Couplings, as in
                :class:`HardwareModel`, with the qubit names of the file. The qubit names of
                a pair of the file can have hyphens if they are not ambiguous.
                Defaults to ``None``, which is the pairs that are in the file.
            native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): It must
                contain ``CZ``, the only two-qubit gate that ``qibocal`` benchmarks.
                Defaults to ``NativeGates.default()``.
            pulses_per_clifford (float, optional): Average number of pulses in a single-qubit
                Clifford gate, which converts its error to that of a pulse.
                Defaults to :math:`1.875`.
            symmetric (dict, optional): Which couplings have a symmetric two-qubit gate, as in
                :class:`HardwareModel`. Defaults to ``None``.

        Returns:
            :class:`HardwareModel`: Model with the errors and the coherence times of the
            calibration, which only use the means of its measures.

        Raises:
            ValueError: If ``calibration`` is not the content of a ``calibration.json``, if
                ``native_gates`` has no ``CZ``, if a pair of the file is not two qubits of the
                file, or if ``durations`` has no duration that the noise model needs.

        References:
            1. M. A. Nielsen, *A simple formula for the average gate fidelity of a quantum
            dynamical operation*, `Phys. Lett. A 303, 249 (2002)
            <https://arxiv.org/abs/quant-ph/0205035>`_.
        """
        if not isinstance(calibration, dict) or not isinstance(
            calibration.get("single_qubits"), dict
        ):
            raise_error(
                ValueError,
                "``calibration`` must be the content of a ``calibration.json`` exported "
                + "by ``qibocal``.",
            )

        native_gates = NativeGates.default() if native_gates is None else native_gates

        if gates.CZ not in native_gates.two_qubit_gates:
            raise_error(ValueError, "The native gates must contain ``CZ``.")

        single = {
            _name(key): value for key, value in calibration["single_qubits"].items()
        }

        # the keys of the pairs have the names of the two qubits, like "0-1" or "A1-A2",
        # and the names can have hyphens, so the pair is the split with two qubits of the file
        pairs = {}
        for key, value in (calibration.get("two_qubits") or {}).items():
            splits = [
                (_name(key[:index]), _name(key[index + 1 :]))
                for index, char in enumerate(key)
                if char == "-"
            ]
            splits = [pair for pair in splits if set(pair) <= set(single)]
            if len(splits) != 1:
                raise_error(
                    ValueError,
                    f"The coupling ``{key}`` of ``calibration`` must be two qubits "
                    + "of the file, in only one way.",
                )
            pairs[splits[0]] = value

        # the model without errors has the connectivity as a graph, with the directions
        hardware = cls(
            tuple(single),
            list(pairs) if connectivity is None else connectivity,
            native_gates,
            durations,
            symmetric=symmetric,
        )

        coherence = {
            qubit: (t_1, t_2)
            for qubit, data in single.items()
            if (t_1 := _mean(data.get("t1"))) is not None
            and (t_2 := _mean(data.get("t2"))) is not None
        }
        errors = {
            "1q": {
                qubit: (1 - rb) / pulses_per_clifford
                for qubit, data in single.items()
                if (rb := _mean(data.get("rb_fidelity"))) is not None
            },
            "2q": {
                (first, second): 1 - fidelity
                for first, second in hardware.connectivity.edges
                if (
                    fidelity := _mean(
                        (
                            pairs.get((first, second))
                            or pairs.get((second, first))
                            or {}
                        ).get("cz_fidelity")
                    )
                )
                is not None
            },
            "readout": {
                qubit: (1 - fidelity) / 2
                for qubit, data in single.items()
                if (fidelity := (data.get("readout") or {}).get("fidelity")) is not None
            },
        }

        return replace(hardware, errors=errors, coherence=coherence)

    def _build_noise_model(self) -> NoiseModel:
        """Noise model that follows from the errors, the durations and the coherence times."""
        coherence = {}
        for qubit, (t_1, t_2) in self.coherence.items():
            if t_2 > 2 * t_1:
                log.warning(
                    f"The t2 of qubit {qubit} is larger than twice its t1, "
                    + "so it is set to twice its t1."
                )
            coherence[qubit] = (t_1, min(t_2, 2 * t_1))

        position = {qubit: index for index, qubit in enumerate(self.qubits)}
        # the rotations around Z are virtual, so they have no error
        physical = [
            gate
            for gate in self.native_gates.single_qubit_gates
            if gate not in _VIRTUAL
        ]

        noise_model = NoiseModel()
        for kind, natives in (("1q", physical), ("2q", [gates.CZ])):
            for key, error in self.errors.get(kind, {}).items():
                names = key if kind == "2q" else (key,)
                thermal = all(name in coherence for name in names)
                time = self.duration(names) if thermal else None
                fidelity = (
                    _thermal_fidelity([coherence[name] for name in names], time)
                    if thermal
                    else 1.0
                )

                # the depolarizing error completes the average gate fidelity of the gate
                dim = 2 ** len(names)
                if math.isclose(fidelity, 1 / dim):
                    raise_error(
                        ValueError,
                        f"The qubits {names} fully relax during their gate, check that "
                        + "``durations`` and the coherence times have the same unit.",
                    )
                lam = (fidelity - 1 + error) / (fidelity - 1 / dim)
                if lam < 0:
                    log.warning(
                        f"The error of the gate on {names} is smaller than its thermal "
                        + "relaxation, so it has no depolarizing error."
                    )
                    lam = 0.0

                # a coupling of a directed graph is only in one direction
                orderings = (
                    [names]
                    if kind == "1q" or self.connectivity.is_directed()
                    else [names, names[::-1]]
                )
                for ordered in orderings:
                    indices = tuple(position[name] for name in ordered)
                    conditions = (
                        None if kind == "1q" else _Conditions(indices).condition_qubits
                    )
                    for gate in natives:
                        noise_model.add(
                            DepolarizingError(lam), gate, indices, conditions=conditions
                        )
                        if thermal:
                            for name in ordered:
                                noise_model.add(
                                    ThermalRelaxationError(*coherence[name], time),
                                    gate,
                                    position[name],
                                    conditions=conditions,
                                )

        for qubit, flip in self.errors.get("readout", {}).items():
            probabilities = [[1 - flip, flip], [flip, 1 - flip]]
            noise_model.add(ReadoutError(probabilities), gates.M, position[qubit])

        return noise_model


_VIRTUAL = (gates.I, gates.Z, gates.RZ)


def _mean(measure: list | None) -> float | None:
    """Mean of a ``qibocal`` measure, a pair of mean and uncertainty, or ``None``."""
    return None if measure is None else measure[0]


def _name(key: str) -> int | str:
    """Qubit name of a key of ``calibration.json``, which is an integer if it looks like one."""
    return int(key) if key.isdigit() else key


def _thermal_fidelity(coherence: list[tuple], time: float) -> float:
    """Average gate fidelity of the thermal relaxation of some qubits during ``time``.

    Args:
        coherence (list[tuple]): Pair :math:`(T_{1}, \\, T_{2})` of each qubit, in the same
            unit as ``time``.
        time (float): Duration of the relaxation.

    Returns:
        float: Average gate fidelity, with :func:`qibo.quantum_info.average_gate_fidelity`,
        of the thermal relaxation of all the qubits, which is the product of the channels.
    """
    backend = get_backend()
    channels = [
        to_liouville(
            gates.ThermalRelaxationChannel(0, [t_1, t_2, time, 0.0]), backend=backend
        )
        for t_1, t_2 in coherence
    ]
    # the fidelity with the identity only depends on the trace of the channel, and the trace
    # of a tensor product is the product of the traces, for any order of the vectorization
    return average_gate_fidelity(reduce(backend.kron, channels), backend=backend)
