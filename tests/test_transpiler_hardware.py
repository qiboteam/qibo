import dataclasses
import logging
import math

import networkx as nx
import pytest

from qibo import gates
from qibo.noise import DepolarizingError, ReadoutError, ThermalRelaxationError
from qibo.transpiler import HardwareModel
from qibo.transpiler.hardware import _thermal_fidelity
from qibo.transpiler.unroller import NativeGates


def test_duration():
    hardware = HardwareModel(
        ("a", "b", "c"),
        [("a", "b"), ("b", "c")],
        durations={"1q": 10.0, "2q": 50.0, ("a", "b"): 40.0},
    )
    assert hardware.duration(("a",)) == 10.0
    assert hardware.duration(("a", "b")) == 40.0
    assert hardware.duration(("b", "a")) == 40.0
    assert hardware.duration(("b", "c")) == 50.0

    with pytest.raises(ValueError, match="no duration"):
        _hardware().duration((0,))


def test_error():
    hardware = _hardware(errors={"1q": {0: 0.01}, "2q": {(0, 1): 0.02}})
    assert hardware.error((0,)) == 0.01
    assert hardware.error((1,)) is None
    assert hardware.error((0, 1)) == 0.02
    assert hardware.error((1, 0)) == 0.02
    assert hardware.error((1, 2)) is None
    assert _hardware().error((0,)) is None
    assert _hardware().error((0, 1)) is None


def test_from_calibration():
    calibration = {
        "single_qubits": {
            "0": {
                "t1": [100000.0, 5.0],
                "t2": [90000.0, 5.0],
                "rb_fidelity": [0.99, 0.001],
                "readout": {"fidelity": 0.96},
            },
            "A1": {
                "t1": [80000.0, 1.0],
                "t2": [60000.0, 1.0],
                "rb_fidelity": [0.995, 0.001],
            },
            "q-3": {"t1": [1.0, 0.1]},
        },
        "two_qubits": {"0-A1": {"cz_fidelity": [0.98, 0.01]}},
    }
    durations = {"1q": 20.0, "2q": 40.0}

    hardware = HardwareModel.from_calibration(calibration, durations)
    assert hardware.qubits == (0, "A1", "q-3")
    assert not hardware.connectivity.is_directed()
    assert set(hardware.connectivity.edges) == {(0, "A1")}
    assert hardware.errors["1q"] == pytest.approx(
        {0: 0.01 / 1.875, "A1": 0.005 / 1.875}
    )
    assert hardware.errors["2q"] == pytest.approx({(0, "A1"): 0.02})
    assert hardware.errors["readout"] == pytest.approx({0: 0.02})
    # a qubit needs both times, and the qubits without data have no error
    assert hardware.coherence == {0: (100000.0, 90000.0), "A1": (80000.0, 60000.0)}

    # the pair of the file in the other order, the error of a pulse, and the directions
    calibration["two_qubits"] = {"A1-0": {"cz_fidelity": [0.97, 0.01]}}
    hardware = HardwareModel.from_calibration(
        calibration,
        durations,
        connectivity=[(0, "A1"), (0, "q-3")],
        pulses_per_clifford=1.0,
        symmetric={(0, "A1"): False},
    )
    assert hardware.connectivity.is_directed()
    assert hardware.errors["1q"] == pytest.approx({0: 0.01, "A1": 0.005})
    assert hardware.errors["2q"] == pytest.approx({(0, "A1"): 0.03})

    # without couplings or readout in the file
    calibration["two_qubits"] = None
    calibration["single_qubits"]["0"]["readout"] = None
    hardware = HardwareModel.from_calibration(calibration, durations)
    assert hardware.connectivity.number_of_edges() == 0
    assert hardware.errors["2q"] == {}
    assert hardware.errors["readout"] == {}


def test_from_calibration_errors():
    durations = {"1q": 20.0, "2q": 40.0}
    for calibration in ([], {}, {"single_qubits": []}):
        with pytest.raises(ValueError, match="calibration.json"):
            HardwareModel.from_calibration(calibration, durations)

    single = {"single_qubits": {"0": {}, "1": {}}, "two_qubits": {}}
    with pytest.raises(ValueError, match="``CZ``"):
        HardwareModel.from_calibration(
            single, durations, native_gates=NativeGates.default() & ~NativeGates.CZ
        )

    # a pair with a qubit that is not in the file, and a pair with two ways to read it
    unknown = {"single_qubits": {"0": {}, "1": {}}, "two_qubits": {"0-9": {}}}
    ambiguous = {
        "single_qubits": {"a": {}, "b-c": {}, "a-b": {}, "c": {}},
        "two_qubits": {"a-b-c": {}},
    }
    for calibration in (unknown, ambiguous):
        with pytest.raises(ValueError, match="only one way"):
            HardwareModel.from_calibration(calibration, durations)


def test_hardware_model_connectivity_graph():
    graph = nx.path_graph(2)
    hardware = HardwareModel((0, 1, 2), graph)
    # the qubits without couplings are nodes, and the graph is a copy
    assert set(hardware.connectivity.nodes) == {0, 1, 2}
    assert set(graph.nodes) == {0, 1}
    assert hardware.connectivity is not graph
    assert not hardware.native_gates & NativeGates.CNOT

    directed = nx.DiGraph([(0, 1)])
    hardware = HardwareModel((0, 1), directed, errors={"2q": {(0, 1): 0.1}})
    assert hardware.connectivity.is_directed()
    with pytest.raises(ValueError, match="is not a coupling"):
        HardwareModel((0, 1), directed, errors={"2q": {(1, 0): 0.1}})


def test_hardware_model_couplings():
    hardware = HardwareModel((0, 1, 2), [(0, 1), (1, 2)])
    assert not hardware.connectivity.is_directed()
    assert set(map(frozenset, hardware.connectivity.edges)) == {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }

    # the flag of a coupling can be given in the other order
    hardware = HardwareModel((0, 1, 2), [(0, 1), (1, 2)], symmetric={(1, 0): False})
    assert hardware.connectivity.is_directed()
    assert set(hardware.connectivity.edges) == {(0, 1), (1, 2), (2, 1)}


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"qubits": (0, 0)}, "repeated names"),
        ({"connectivity": nx.path_graph(2), "symmetric": {}}, "only for a list"),
        ({"connectivity": [(0, 1, 2)]}, "must be a pair"),
        ({"symmetric": {(1, 2): False}}, "not couplings"),
        ({"connectivity": [(0, 2)]}, "not in ``qubits``"),
        ({"connectivity": [(0, 0)]}, "coupled to itself"),
        ({"durations": {(0, 2): 1.0}}, "of ``durations``"),
        ({"durations": {"1q": 0.0}}, "must be positive"),
        ({"errors": {"3q": {}}}, "must be '1q'"),
        ({"errors": {"1q": {5: 0.1}}}, "is not a qubit"),
        ({"errors": {"readout": {5: 0.1}}}, "is not a qubit"),
        ({"errors": {"2q": {(0, 5): 0.1}}}, "is not a coupling"),
        ({"errors": {"1q": {0: 1.5}}}, "between 0 and 1"),
        ({"coherence": {5: (1.0, 1.0)}}, "is not a qubit"),
        ({"coherence": {0: (1.0,)}}, "positive numbers"),
        ({"coherence": {0: (1.0, -1.0)}}, "positive numbers"),
    ],
)
def test_hardware_model_error(kwargs, message):
    kwargs = {"qubits": (0, 1), "connectivity": [(0, 1)]} | kwargs
    with pytest.raises(ValueError, match=message):
        HardwareModel(**kwargs)


def test_hardware_model_frozen():
    hardware = _hardware()
    with pytest.raises(dataclasses.FrozenInstanceError):
        hardware.qubits = ()


def test_noise_model_depolarizing():
    natives = NativeGates.default() | NativeGates.RX
    hardware = _hardware(
        natives=natives,
        errors={"1q": {0: 0.01}, "2q": {(0, 1): 0.03}, "readout": {1: 0.05}},
    )
    errors = hardware.noise_model.errors
    # the rotations around Z and the identity are virtual, so they have no error
    assert {gates.GPI2, gates.RX, gates.CZ, gates.M} == set(errors)

    # without relaxation, the depolarizing error is the whole error
    for gate in (gates.GPI2, gates.RX):
        [(condition, error, qubits)] = errors[gate]
        assert condition is None and qubits == (0,)
        assert isinstance(error, DepolarizingError)
        assert error.options == pytest.approx(0.01 / (1 - 1 / 2))

    # the coupling is symmetric, so it is in both orderings
    assert [qubits for _, _, qubits in errors[gates.CZ]] == [(0, 1), (1, 0)]
    assert all(
        error.options == pytest.approx(0.03 / (1 - 1 / 4))
        for _, error, _ in errors[gates.CZ]
    )

    [(_, error, qubits)] = errors[gates.M]
    assert isinstance(error, ReadoutError) and qubits == (1,)
    assert error.options[0] == pytest.approx([0.95, 0.05])
    assert error.options[1] == pytest.approx([0.05, 0.95])


def test_noise_model_directed():
    hardware = _hardware(symmetric={(0, 1): False}, errors={"2q": {(0, 1): 0.03}})
    assert [qubits for _, _, qubits in hardware.noise_model.errors[gates.CZ]] == [
        (0, 1)
    ]


def test_noise_model_error_smaller_than_relaxation(caplog):
    with caplog.at_level(logging.WARNING):
        hardware = _hardware(
            durations={"1q": 2000.0},
            errors={"1q": {0: 1e-6}},
            coherence={0: (1e4, 2e4)},
        )
    assert "smaller than its thermal relaxation" in caplog.text
    depolarizing = [
        error
        for _, error, _ in hardware.noise_model.errors[gates.GPI2]
        if isinstance(error, DepolarizingError)
    ]
    assert depolarizing[0].options == 0.0


def test_noise_model_full_relaxation():
    with pytest.raises(ValueError, match="fully relax"):
        _hardware(
            durations={"1q": 50.0, "2q": 50.0},
            coherence={0: (1.0, 2.0), 1: (1.0, 2.0)},
            errors={"1q": {0: 0.01}},
        )


def test_noise_model_missing_duration():
    with pytest.raises(ValueError, match="no duration"):
        _hardware(errors={"1q": {0: 0.01}}, coherence={0: (100.0, 100.0)})


def test_noise_model_t2_clip(caplog):
    with caplog.at_level(logging.WARNING):
        hardware = _hardware(
            durations={"1q": 20.0},
            errors={"1q": {0: 0.01}},
            coherence={0: (1e5, 5e5)},
        )
    assert "larger than twice its t1" in caplog.text
    thermal = [
        error
        for _, error, _ in hardware.noise_model.errors[gates.GPI2]
        if isinstance(error, ThermalRelaxationError)
    ]
    assert thermal[0].options[:2] == [1e5, 2e5]


def test_noise_model_thermal():
    t_1, time_1, time_2 = 1e5, 20.0, 40.0
    hardware = _hardware(
        durations={"1q": time_1, "2q": time_2},
        errors={"1q": {0: 0.01}, "2q": {(0, 1): 0.03, (1, 2): 0.03}},
        coherence={0: (t_1, 2 * t_1), 1: (t_1, 2 * t_1)},
    )
    errors = hardware.noise_model.errors

    # the depolarizing error completes the average gate fidelity of the gate, which has
    # a closed form if the thermal relaxation is a pure amplitude damping, *i.e.* T2 = 2 T1
    for gate, qubits, error, nqubits, time in (
        (gates.GPI2, (0,), 0.01, 1, time_1),
        (gates.CZ, (0, 1), 0.03, 2, time_2),
    ):
        average = _fidelity(t_1, time, nqubits)
        expected = (average - 1 + error) / (average - 1 / 2**nqubits)
        assert 0 < expected < 1
        assert _depolarizing(errors, gate, qubits) == pytest.approx(expected)

    # the relaxation of each qubit during the gate
    thermal = [
        (qubits, error.options)
        for _, error, qubits in errors[gates.CZ]
        if isinstance(error, ThermalRelaxationError)
    ]
    assert sorted(thermal) == sorted(
        [((q,), [t_1, 2 * t_1, time_2, 0]) for q in (0, 1, 1, 0)]
    )

    # the coupling of a qubit without coherence has no relaxation, only depolarizing
    assert _depolarizing(errors, gates.CZ, (1, 2)) == pytest.approx(0.03 / (1 - 1 / 4))
    assert not any(
        isinstance(error, ThermalRelaxationError) and qubits == (2,)
        for _, error, qubits in errors[gates.CZ]
    )


def test_thermal_fidelity():
    t_1, time = 100.0, 30.0
    # a pure amplitude damping, with an entanglement fidelity that is a product for
    # independent qubits and an average gate fidelity of (d F + 1) / (d + 1)
    entanglement = (1 + math.exp(-time / (2 * t_1))) ** 2 / 4
    assert _thermal_fidelity([(t_1, 2 * t_1)], time) == pytest.approx(
        (2 * entanglement + 1) / 3
    )
    assert _thermal_fidelity([(t_1, 2 * t_1)] * 2, time) == pytest.approx(
        (4 * entanglement**2 + 1) / 5
    )


def test_weighted_connectivity():
    hardware = _hardware(
        errors={
            "1q": {0: 0.01, 1: 0.02, 2: 0.03},
            "2q": {(0, 1): 0.1, (1, 2): 0.2},
            "readout": {0: 0.04, 1: 0.05, 2: 0.06},
        },
    )
    graph = hardware.weighted_connectivity()
    for qubit, gate, readout in ((0, 0.01, 0.04), (1, 0.02, 0.05), (2, 0.03, 0.06)):
        assert graph.nodes[qubit]["gate"] == pytest.approx(_weight(gate))
        assert graph.nodes[qubit]["readout"] == pytest.approx(_weight(readout))
        assert graph.nodes[qubit]["weight"] == pytest.approx(
            _weight(gate) + _weight(readout)
        )
    assert graph.edges[0, 1]["weight"] == pytest.approx(_weight(0.1))
    assert graph.edges[2, 1]["weight"] == pytest.approx(_weight(0.2))
    # the model is not changed
    assert "weight" not in hardware.connectivity.nodes[0]


def test_weighted_connectivity_directed(caplog):
    hardware = _hardware(
        symmetric={(0, 1): False},
        errors={"2q": {(0, 1): 0.1, (1, 2): 0.2, (2, 1): 0.3}, "1q": {0: 0.01}},
    )
    with caplog.at_level(logging.WARNING):
        graph = hardware.weighted_connectivity()
    assert graph.is_directed() and not graph.has_edge(1, 0)
    assert graph.edges[1, 2]["weight"] == pytest.approx(_weight(0.2))
    assert graph.edges[2, 1]["weight"] == pytest.approx(_weight(0.3))


def test_weighted_connectivity_infinite_and_zero():
    # a certain failure has an infinite weight, and the data that is missing is zero
    hardware = _hardware(errors={"1q": {0: 1.0, 1: 0.5, 2: 0.0}})
    graph = hardware.weighted_connectivity()
    assert graph.nodes[0]["gate"] == math.inf
    assert graph.nodes[2]["gate"] == 0.0
    assert all(graph.nodes[qubit]["readout"] == 0.0 for qubit in range(3))
    assert all(data["weight"] == 0.0 for _, _, data in graph.edges(data=True))


def test_weighted_connectivity_missing(caplog):
    hardware = _hardware(
        errors={
            "1q": {0: 0.01, 2: 0.03},
            "2q": {(0, 1): 0.1},
            "readout": {1: 0.05},
        },
    )
    with caplog.at_level(logging.WARNING):
        graph = hardware.weighted_connectivity()
    # each component with some data and some gaps warns once and uses the mean
    assert "gate error of [1] is missing" in caplog.text
    assert "coupling error of [(1, 2)] is missing" in caplog.text
    assert "readout error of [0, 2] is missing" in caplog.text
    assert graph.nodes[1]["gate"] == pytest.approx(_weight(0.02))
    assert graph.edges[1, 2]["weight"] == pytest.approx(_weight(0.1))
    assert graph.nodes[0]["readout"] == pytest.approx(_weight(0.05))


def _depolarizing(errors, gate, qubits):
    """Parameter of the depolarizing error of a gate on some qubits in a noise model."""
    [error] = [
        error
        for _, error, acting in errors[gate]
        if isinstance(error, DepolarizingError) and acting == qubits
    ]
    return error.options


def _fidelity(t_1, time, nqubits):
    """Average gate fidelity of the relaxation of qubits with T2 = 2 T1 during a gate.

    It is a pure amplitude damping, with an entanglement fidelity that is a product for
    independent qubits, and an average gate fidelity of (d F + 1) / (d + 1).
    """
    entanglement = ((1 + math.exp(-time / (2 * t_1))) ** 2 / 4) ** nqubits
    dim = 2**nqubits
    return (dim * entanglement + 1) / (dim + 1)


def _hardware(natives=None, **kwargs):
    """Model of three qubits in a line, with the options of ``HardwareModel``."""
    if natives is not None:
        kwargs["native_gates"] = natives
    return HardwareModel((0, 1, 2), [(0, 1), (1, 2)], **kwargs)


def _weight(error):
    """Weight of an error, which is the negative logarithm of its success probability."""
    return -math.log1p(-error)
