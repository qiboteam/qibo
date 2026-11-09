import math
from dataclasses import InitVar, dataclass, field

import networkx as nx

from qibo import gates
from qibo.config import raise_error
from qibo.noise import (
    DepolarizingError,
    NoiseModel,
    ReadoutError,
    ThermalRelaxationError,
    _Conditions,
)
from qibo.transpiler.unroller import NativeGates


@dataclass(eq=False)
class HardwareModel:
    """Description of a quantum processor.

    Includes its qubits, couplings, native gates, gate durations, and the noise calibration.

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
            hardware = HardwareModel.from_json(calibration, {"1q": 40.0, "2q": 60.0})

    Args:
        qubits (tuple): Qubit names, as in the wire names of a transpiled circuit.
            The noise model refers to the qubits by their position in this tuple.
        connectivity (:class:`networkx.Graph` or list[tuple]): Couplings, as a graph with the
            qubit names as nodes or as a list of pairs of qubit names, *e.g.*
            ``[(0, 1), (1, "A2")]``, which is how ``qibolab`` lists them. They are stored as
            a graph that also has the qubits without couplings. A directed graph has an edge
            ``(a, b)`` for a gate that only acts with its first qubit on ``a`` and its second
            qubit on ``b``, as in :func:`qibo.transpiler.pipeline.generate_preset_passes`.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native
            gates of every qubit. Defaults to ``NativeGates.default()``.
        durations (dict, optional): Gate durations in ns. The keys ``"1q"`` and ``"2q"`` set
            the default of each kind, and a pair of qubit names sets that of a coupling.
            Defaults to an empty dictionary.
        noise_model (:class:`qibo.noise.NoiseModel`, optional): Errors that follow the native
            gates, on the qubit positions. Defaults to ``None``, which is a noiseless device.
        measures (dict, optional): Measures of the calibration that the errors come from, as
            pairs of the mean and its uncertainty, which is ``None`` if it is not known. The
            keys are ``"t1"``, ``"t2"`` and ``"rb_fidelity"``, with qubit names as the keys of
            their values, and ``"cz_fidelity"``, with pairs of qubit names, *e.g.*
            ``{"t1": {0: (60000.0, 500.0)}}``. Defaults to an empty dictionary.
        symmetric (dict, optional): Only for a list of couplings. It tells if the two-qubit
            gate of a coupling is symmetric, as ``symmetric`` of the natives of ``qibolab``, with
            pairs as keys, *e.g.* ``{(0, 1): False}``. The couplings that are not symmetric are
            only in the direction of their pair, and the connectivity is then a directed graph.
            The pairs that are not in ``symmetric`` are symmetric. Defaults to ``None``, so all
            of them are symmetric.
    """

    qubits: tuple
    connectivity: nx.Graph | list[tuple]
    native_gates: NativeGates = field(default_factory=NativeGates.default)
    durations: dict = field(default_factory=dict)
    noise_model: NoiseModel | None = None
    measures: dict = field(default_factory=dict)
    symmetric: InitVar[dict | None] = None

    def __post_init__(self, symmetric: dict | None):
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
            flags = [
                symmetric.get(pair, symmetric.get(pair[::-1], True)) for pair in pairs
            ]
            graph = nx.Graph() if all(flags) else nx.DiGraph()
            for pair, flag in zip(pairs, flags):
                graph.add_edge(*pair)
                if flag:
                    graph.add_edge(*pair[::-1])
        if unknown := set(graph.nodes) - set(self.qubits):
            raise_error(
                ValueError,
                f"The couplings have qubits that are not in ``qubits``: {unknown}.",
            )
        graph.add_nodes_from(self.qubits)
        self.connectivity = graph

    @classmethod
    def from_json(
        cls,
        calibration: dict,
        durations: dict,
        connectivity: nx.Graph | list[tuple] | None = None,
        native_gates: NativeGates | None = None,
        pulses_per_clifford: float = 1.875,
        symmetric: dict | None = None,
    ):
        """Build the model from the content of the ``calibration.json`` that ``qibocal``
        exports, for example loaded with ``json.load``.

        The error of each native gate is the one measured with randomized benchmarking,
        split into the thermal relaxation during the gate, from ``t1`` and ``t2``, and a
        depolarizing error with the rest. Gates without data have no error.

        Args:
            calibration (dict): Content of the ``calibration.json`` file.
            durations (dict): Gate durations in ns, as in :class:`HardwareModel`.
            connectivity (:class:`networkx.Graph` or list[tuple], optional): Couplings, as in
                :class:`HardwareModel`, with the qubit names of the file.
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
            :class:`HardwareModel`: Model with the noise model of the calibration, which only
            uses the means, and with its measures and their uncertainties. The errors of a
            coupling are only in the directions of the connectivity.
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

        # the keys of the pairs have the names of the two qubits, like "0-1" or "A1-A2"
        two_qubits = calibration.get("two_qubits") or {}
        pairs = {}
        for key, value in two_qubits.items():
            names = [_name(name) for name in key.split("-")]
            pairs[tuple(names)] = value

        measures = {
            key: {
                name: tuple(data[key])
                for name, data in single.items()
                if data.get(key) is not None
            }
            for key in ("t1", "t2", "rb_fidelity")
        }
        measures["cz_fidelity"] = {
            pair: tuple(data["cz_fidelity"])
            for pair, data in pairs.items()
            if data.get("cz_fidelity") is not None
        }

        hardware = cls(
            tuple(single),
            list(pairs) if connectivity is None else connectivity,
            native_gates,
            dict(durations),
            measures=measures,
            symmetric=symmetric,
        )

        position = {qubit: index for index, qubit in enumerate(hardware.qubits)}
        # the rotations around Z are virtual, so they have no error
        physical = [
            gate for gate in native_gates.single_qubit_gates if gate not in _VIRTUAL
        ]
        noise_model = NoiseModel()
        coherence = {}
        for qubit, data in single.items():
            index = position[qubit]
            t_1, t_2 = measures["t1"].get(qubit), measures["t2"].get(qubit)
            if t_1 is not None and t_2 is not None:
                coherence[qubit] = (t_1[0], min(t_2[0], 2 * t_1[0]))
            if (
                rb := measures["rb_fidelity"].get(qubit)
            ) is not None and qubit in coherence:
                time = durations["1q"]
                error = (1 - rb[0]) / pulses_per_clifford
                lam = 2 * max(error - _thermal_infidelity(*coherence[qubit], time), 0)
                for gate in physical:
                    noise_model.add(DepolarizingError(lam), gate, index)
                    noise_model.add(
                        ThermalRelaxationError(*coherence[qubit], time), gate, index
                    )
            if (fidelity := (data.get("readout") or {}).get("fidelity")) is not None:
                flip = (1 - fidelity) / 2
                probabilities = [[1 - flip, flip], [flip, 1 - flip]]
                noise_model.add(ReadoutError(probabilities), gates.M, index)

        directed = hardware.connectivity.is_directed()
        for first, second in hardware.connectivity.edges:
            cz_fidelity = measures["cz_fidelity"]
            measure = cz_fidelity.get((first, second), cz_fidelity.get((second, first)))
            if measure is None or not {first, second} <= set(coherence):
                continue
            fidelity = measure[0]
            time = durations.get(
                (first, second), durations.get((second, first), durations["2q"])
            )
            thermal = [
                _thermal_infidelity(*coherence[q], time) for q in (first, second)
            ]
            lam = 4 / 3 * max(1 - fidelity - sum(thermal), 0)
            # a coupling of a directed graph is only in one direction
            for ordered in (
                [(first, second)] if directed else [(first, second), (second, first)]
            ):
                indices = tuple(position[q] for q in ordered)
                on_pair = _Conditions(indices).condition_qubits
                noise_model.add(
                    DepolarizingError(lam), gates.CZ, indices, conditions=on_pair
                )
                for qubit in ordered:
                    noise_model.add(
                        ThermalRelaxationError(*coherence[qubit], time),
                        gates.CZ,
                        position[qubit],
                        conditions=on_pair,
                    )
        hardware.noise_model = noise_model

        return hardware


_VIRTUAL = (gates.I, gates.Z, gates.RZ)


def _name(key: str) -> int | str:
    """Qubit name of a key of ``calibration.json``, which is an integer if it looks like one."""
    return int(key) if key.lstrip("-").isdigit() else key


def _thermal_infidelity(t_1: float, t_2: float, time: float) -> float:
    """Average gate infidelity of the thermal relaxation of a qubit during ``time``."""
    return 1 - (1 / 2 + math.exp(-time / t_1) / 6 + math.exp(-time / t_2) / 3)
