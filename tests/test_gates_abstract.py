"""Tests methods defined in `qibo/gates/abstract.py` and
`qibo/gates/gates.py`."""

import json
from typing import Optional

import pytest

from qibo import gates, matrices
from qibo.config import PRECISION_TOL
from qibo.gates import abstract


@pytest.mark.parametrize(
    "gatename", ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "I", "Align"]
)
def test_one_qubit_gates_init(gatename):
    gate = getattr(gates, gatename)(0)
    assert gate.target_qubits == (0,)


def gate_from_json(gatename: str, control: Optional[list] = None):
    gate = getattr(gates, gatename)(0)

    control = [] if control is None else control

    json_general = f"""
    {{
        "name": {{}},
        "init_args": [0],
        "init_kwargs": {{}},
        "_target_qubits": [0],
        "_control_qubits": {control}
    }}
    """

    json_gate = json.loads(json_general)
    json_gate["name"] = gate.name

    return gate, json_gate


@pytest.mark.parametrize(
    "gatename", ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "I", "Align"]
)
def test_one_qubit_gates_serialization(gatename):
    gate, json_gate = gate_from_json(gatename)
    raw = gate.raw

    assert isinstance(raw, dict)
    assert gate.to_json() == json.dumps(raw)

    del raw["_class"]

    # raw may contain some objects later converted by JSON (e.g. tuples)
    assert json.loads(json.dumps(raw)) == json_gate
    assert gates.Gate.from_dict(gate.raw).raw == gate.raw


@pytest.mark.parametrize(
    "gatename", ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "I", "Align"]
)
def test_controlled_gates_serialization(gatename):
    gate, _ = gate_from_json(gatename, control=[1, 4])

    assert isinstance(gate.raw, dict)
    assert gates.Gate.from_dict(gate.raw).raw == gate.raw


def test_gates_serialization_errors(monkeypatch):
    with pytest.raises(ValueError, match="Unknown"):
        _ = abstract.Gate.from_dict({"_class": "Ciao"})

    error_tag = "not-control-error"

    def mock_controlled_by(*args, **kwargs):
        raise RuntimeError(error_tag)

    monkeypatch.setattr(abstract.Gate, "controlled_by", mock_controlled_by)

    with pytest.raises(RuntimeError, match=error_tag):
        _ = abstract.Gate.from_dict(
            {"_class": "H", "init_args": (0,), "init_kwargs": {}, "_control_qubits": ()}
        )


@pytest.mark.parametrize(
    "controls,instance", [((1,), "CNOT"), ((1, 2), "TOFFOLI"), ((1, 2, 4), "X")]
)
def test_x_controlled_by(controls, instance):
    gate = gates.X(0).controlled_by(*controls)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == controls
    assert isinstance(gate, getattr(gates, instance))


def test_x_decompose_with_few_controls():
    """Check ``X`` decomposition with less than three controls."""
    gate = gates.X(0)
    decomp = gate.decompose(1, 2)
    assert len(decomp) == 1
    assert isinstance(decomp[0], gates.X)


@pytest.mark.parametrize("use_toffolis", [True, False])
def test_x_decomposition_errors(use_toffolis):
    """Check ``X`` decomposition errors."""
    gate = gates.X(0).controlled_by(1, 2, 3, 4)
    with pytest.raises(ValueError):
        _ = gate.decompose(2, 3, use_toffolis=use_toffolis)


@pytest.mark.parametrize("controls,instance", [((1,), "CZ"), ((1, 2), "Z")])
def test_z_controlled_by(controls, instance):
    gate = gates.Z(0).controlled_by(*controls)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == controls
    assert isinstance(gate, getattr(gates, instance))


@pytest.mark.parametrize(
    "targets,p0,p1",
    [((0,), None, None), ((0, 1, 2), None, None), ((0, 3, 2), 0.2, 0.1)],
)
def test_measurement_init(targets, p0, p1):
    # also tests `_get_bitflip_map`
    gate = gates.M(*targets, p0=p0, p1=p1)
    assert gate.target_qubits == targets
    p0map = {q: 0 if p0 is None else p0 for q in targets}
    p1map = {q: 0 if p1 is None else p1 for q in targets}
    assert gate.bitflip_map == (p0map, p1map)


def test_measurement_add():
    gate = gates.M(0, 2)
    assert gate.target_qubits == (0, 2)
    assert gate.bitflip_map == 2 * ({0: 0, 2: 0},)
    gate.add(gates.M(1, 3, p0=0.3, p1=0.0))
    assert gate.target_qubits == (0, 2, 1, 3)
    assert gate.bitflip_map == ({0: 0, 1: 0.3, 2: 0, 3: 0.3}, {0: 0, 1: 0, 2: 0, 3: 0})


def test_measurement_errors():
    gate = gates.M(0)
    with pytest.raises(NotImplementedError):
        gate.controlled_by(1)


@pytest.mark.parametrize(
    "gatename,params",
    [
        ("RX", (0.1234,)),
        ("RY", (0.1234,)),
        ("RZ", (0.1234,)),
        ("U1", (0.1234,)),
        ("U2", (0.1234, 0.4321)),
        ("U3", (0.1234, 0.4321, 0.5678)),
    ],
)
def test_one_qubit_rotations_init(gatename, params):
    gate = getattr(gates, gatename)(0, *params)
    assert gate.target_qubits == (0,)
    assert gate.parameters == params


@pytest.mark.parametrize(
    "gatename,params",
    [
        ("RX", (0.1234,)),
        ("RY", (0.1234,)),
        ("RZ", (0.1234,)),
        ("U1", (0.1234,)),
        ("U2", (0.1234, 0.4321)),
        ("U3", (0.1234, 0.4321, 0.5678)),
    ],
)
def test_one_qubit_rotations_serialization(gatename, params):
    gate = getattr(gates, gatename)(0, *params)

    json_general = """
    {
        "name": {},
        "init_args": [0],
        "init_kwargs": {},
        "_target_qubits": [0],
        "_control_qubits": []
    }
    """

    json_gate = json.loads(json_general)
    json_gate["name"] = gate.name
    json_gate["init_kwargs"] = gate.init_kwargs
    del json_gate["init_kwargs"]["trainable"]

    raw = gate.raw
    del raw["_class"]

    raw = gate.raw

    assert isinstance(raw, dict)
    assert gate.to_json() == json.dumps(raw)

    del raw["_class"]

    # raw may contain some objects later converted by JSON (e.g. tuples)
    assert json.loads(json.dumps(raw)) == json_gate
    assert gates.Gate.from_dict(gate.raw).raw == gate.raw


@pytest.mark.parametrize(
    "gatename,params",
    [
        ("RX", (0.1234,)),
        ("RY", (0.1234,)),
        ("RZ", (0.1234,)),
        ("U1", (0.1234,)),
        ("U2", (0.1234, 0.4321)),
        ("U3", (0.1234, 0.4321, 0.5678)),
    ],
)
def test_one_qubit_rotations_controlled_by(gatename, params):
    gate = getattr(gates, gatename)(0, *params).controlled_by(1)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == (1,)
    assert isinstance(gate, getattr(gates, f"C{gatename}"))
    gate = getattr(gates, gatename)(1, *params).controlled_by(0, 3)
    assert gate.target_qubits == (1,)
    assert gate.control_qubits == (0, 3)
    assert gate.parameters == params


def test_cnot_and_cy_and_cz_init():
    gate = gates.CNOT(0, 1)
    assert gate.target_qubits == (1,)
    assert gate.control_qubits == (0,)
    gate = gates.CY(4, 7)
    assert gate.target_qubits == (7,)
    assert gate.control_qubits == (4,)
    gate = gates.CZ(3, 2)
    assert gate.target_qubits == (2,)
    assert gate.control_qubits == (3,)


# :meth:`qibo.gates.CNOT.decompose` is tested in
# ``test_x_decompose_with_cirq`` above


@pytest.mark.parametrize(
    "gatename,params",
    [
        ("CRX", (0.1234,)),
        ("CRY", (0.1234,)),
        ("CRZ", (0.1234,)),
        ("CU1", (0.1234,)),
        ("CU2", (0.1234, 0.4321)),
        ("CU3", (0.1234, 0.4321, 0.5678)),
    ],
)
def test_two_qubit_controlled_rotations_init(gatename, params):
    gate = getattr(gates, gatename)(0, 2, *params)
    assert gate.target_qubits == (2,)
    assert gate.control_qubits == (0,)


def test_swap_init():
    gate = gates.SWAP(4, 3)
    assert gate.target_qubits == (4, 3)


def test_fsim_init():
    import numpy as np

    gate = gates.fSim(0, 1, 0.1234, 0.4321)
    assert gate.target_qubits == (0, 1)
    matrix = np.random.random((2, 2))
    gate = gates.GeneralizedfSim(0, 1, matrix, 0.4321)
    assert gate.target_qubits == (0, 1)
    assert gate.parameters == (matrix, 0.4321)
    matrix = np.random.random((3, 3))
    with pytest.raises(ValueError):
        gate = gates.GeneralizedfSim(0, 1, matrix, 0.4321)


def test_toffoli_init():
    gate = gates.TOFFOLI(0, 2, 1)
    assert gate.target_qubits == (1,)
    assert gate.control_qubits == (0, 2)


# :meth:`qibo.gates.TOFFOLI.decompose` and
# :meth:`qibo.gates.TOFFOLI.congruent`
# are tested in `test_x_decompose_with_cirq`


@pytest.mark.parametrize("targets", [(0,), (2, 0), (1, 3, 2)])
def test_unitary_init(targets):
    import numpy as np

    matrix = np.random.random(2 * (2 ** len(targets),))
    gate = gates.Unitary(matrix, *targets)
    assert gate.target_qubits == targets
    assert gate.nparams == 4 ** len(targets)


def test_kraus_channel_init():
    import numpy as np

    qubits = [(0,), (0, 1), (0, 2), (3,)]
    ops = [np.random.random((2 ** len(q), 2 ** len(q))) for q in qubits]
    gate = gates.KrausChannel(qubits, ops)
    gate.target_qubits == (0, 1, 2, 3)
    for g in gate.gates:
        assert isinstance(g, gates.Unitary)
    qubits.append((4,))
    ops.append(np.random.random((4, 4)))
    with pytest.raises(ValueError):
        gate = gates.KrausChannel(qubits, ops)


def test_unitary_channel_init():
    import numpy as np

    qubits = [(0,), (0, 1), (0, 2), (3,)]
    ops = [(0.1, np.random.random((2 ** len(q), 2 ** len(q)))) for q in qubits]
    gate = gates.UnitaryChannel(qubits, ops)
    gate.target_qubits == (0, 1, 2, 3)
    for g in gate.gates:
        assert isinstance(g, gates.Unitary)

    with pytest.raises(ValueError):
        gate = gates.UnitaryChannel(qubits[:2], ops)
    with pytest.raises(ValueError):
        ops[0] = (-0.1, np.random.random((2, 2)))
        gate = gates.UnitaryChannel(qubits, ops)


def test_pauli_noise_channel_init(backend):
    gate = gates.PauliNoiseChannel(0, list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.3])))
    assert gate.target_qubits == (0,)
    for g, p in zip(gate.gates, [matrices.X, matrices.Y, matrices.Z]):
        p = backend.cast(p, dtype=p.dtype)
        backend.assert_allclose(g.matrix(backend), p, atol=PRECISION_TOL)


def test_reset_channel_init():
    gate = gates.ResetChannel(0, [0.1, 0.2])
    assert gate.target_qubits == (0,)


def test_qubit_getter_and_setter():
    from qibo.gates.abstract import Gate

    gate = Gate()
    gate.target_qubits = (0, 3)
    gate.control_qubits = (1, 4, 2)
    assert gate.qubits == (1, 2, 4, 0, 3)

    gate = Gate()
    with pytest.raises(ValueError):
        gate.target_qubits = (1, 1)
    gate = Gate()
    with pytest.raises(ValueError):
        gate.control_qubits = (1, 1)
    gate = Gate()
    gate.target_qubits = (0, 1)
    with pytest.raises(ValueError):
        gate.control_qubits = (1,)


def test_density_matrix_getter_and_setter():
    from qibo.gates.abstract import Gate

    gate = Gate()
    gate.target_qubits = (0, 1)
    gate.control_qubits = (2,)
    gate.density_matrix = True


def test_gates_commute():
    assert gates.H(0).commutes(gates.X(1))
    assert gates.H(0).commutes(gates.H(1))
    assert gates.H(0).commutes(gates.H(0))
    assert not gates.H(0).commutes(gates.Y(0))
    assert not gates.CNOT(0, 1).commutes(gates.SWAP(1, 2))
    assert not gates.CNOT(0, 1).commutes(gates.H(1))
    assert not gates.CNOT(0, 1).commutes(gates.Y(0).controlled_by(2))
    assert not gates.CNOT(2, 3).commutes(gates.CNOT(3, 0))
    assert gates.CNOT(0, 1).commutes(gates.Y(2).controlled_by(0))


def test_on_qubits():
    gate = gates.CNOT(0, 1).on_qubits({0: 2, 1: 3})
    assert gate.target_qubits == (3,)
    assert gate.control_qubits == (2,)
    assert isinstance(gate, gates.CNOT)


def test_controlled_by():
    gate = gates.RX(0, 0.1234).controlled_by(1, 2, 3)
    assert gate.target_qubits == (0,)
    assert gate.control_qubits == (1, 2, 3)
    assert gate.is_controlled_by
    assert isinstance(gate, gates.RX)
    with pytest.raises(RuntimeError):
        gate = gates.CNOT(0, 1).controlled_by(2)


def test_on_qubits_controlled_by():
    gate = gates.H(0).controlled_by(1, 2)
    gate = gate.on_qubits({0: 5, 1: 4, 2: 6})
    assert gate.target_qubits == (5,)
    assert gate.control_qubits == (4, 6)
    assert isinstance(gate, gates.H)
    assert gate.is_controlled_by


def test_decompose():
    decomp_gates = gates.H(0).decompose(1)
    assert len(decomp_gates) == 1
    assert isinstance(decomp_gates[0], gates.H)


def test_special_gate():
    from qibo.gates.abstract import SpecialGate

    gate = SpecialGate()
    assert not gate.commutes(gates.H(0))
    with pytest.raises(NotImplementedError):
        gate.on_qubits({0: 0})


def test_fused_gate():
    gate = gates.FusedGate(0, 1)
    gate.append(gates.H(0))
    gate.append(gates.CNOT(0, 1))
    assert len(gate.gates) == 2
    gate.prepend(gates.TOFFOLI(0, 1, 2))
    assert gate.qubits == (0, 1, 2)
    assert len(gate.gates) == 3
    assert isinstance(gate.gates[0], gates.TOFFOLI)


def test_generator_eigenvalue():
    gate = gates.H(0)
    with pytest.raises(NotImplementedError):
        gate.generator_eigenvalue()


def test_gate_set_parameters():
    gate = gates.RX(0, theta=0)
    assert gate.parameters == (0,)
    gate.parameters = 0.5
    gate2 = gate.__class__(*gate.init_args, **gate.init_kwargs)
    assert gate.parameters == (0.5,)
    assert gate2.parameters == (0.5,)


def test_generalizedfsim_set_parameters():
    import numpy as np

    gate = gates.GeneralizedfSim(0, 1, unitary=np.eye(2), phi=0)
    np.testing.assert_allclose(gate.parameters[0], np.eye(2))
    assert gate.parameters[1] == 0

    new_unitary = np.random.random((2, 2))
    gate.parameters = (new_unitary, 0.5)
    gate2 = gate.__class__(*gate.init_args, **gate.init_kwargs)
    np.testing.assert_allclose(gate.parameters[0], new_unitary)
    np.testing.assert_allclose(gate2.parameters[0], new_unitary)
    assert gate.parameters[1] == 0.5
    assert gate2.parameters[1] == 0.5
