"""Test all methods defined in `qibo/models/circuit.py`."""

from collections import Counter

import numpy as np
import pytest
from networkx import Graph

from qibo import Circuit, gates
from qibo.models.circuit import _resolve_qubits
from qibo.models.utils import initialize
from qibo.transpiler import Passes, Sabre
from qibo.transpiler._exceptions import PlacementError


def test_parametrizedgates_class():
    from qibo.models.circuit import _ParametrizedGates

    paramgates = _ParametrizedGates()
    paramgates.append(gates.RX(0, 0.1234))
    paramgates.append(gates.fSim(0, 1, 0.1234, 0.4321))
    assert len(paramgates.set) == 2
    assert paramgates.nparams == 3


def test_queue_class():
    from qibo.callbacks import EntanglementEntropy
    from qibo.models.circuit import _Queue

    entropy = EntanglementEntropy([0])
    queue = _Queue(4)
    gatelist = [
        gates.H(0),
        gates.H(1),
        gates.X(0),
        gates.H(2),
        gates.CNOT(1, 2),
        gates.Y(3),
        gates.CallbackGate(entropy),
    ]
    for g in gatelist:
        queue.append(g)
    assert queue.moments == [
        [gatelist[0], gatelist[1], gatelist[3], gatelist[5]],
        [gatelist[2], gatelist[4], gatelist[4], None],
        [gatelist[6] for _ in range(4)],
    ]


def test_circuit_init():
    c = Circuit(2)
    assert c.nqubits == 2


def test_resolve_qubits():
    nqubits, wire_names = _resolve_qubits(3, None)
    assert nqubits == 3 and wire_names is None
    nqubits, wire_names = _resolve_qubits(3, ["a", "b", "c"])
    assert nqubits == 3 and wire_names == ["a", "b", "c"]
    nqubits, wire_names = _resolve_qubits(["a", "b", "c"], None)
    assert nqubits == 3 and wire_names == ["a", "b", "c"]
    nqubits, wire_names = _resolve_qubits(None, ["x", "y", "z"])
    assert nqubits == 3 and wire_names == ["x", "y", "z"]

    with pytest.raises(ValueError):
        _resolve_qubits(None, None)
    with pytest.raises(ValueError):
        _resolve_qubits(3, ["a", "b"])
    with pytest.raises(ValueError):
        _resolve_qubits(["a", "b", "c"], ["x", "y"])


def test_circuit_init_resolve_qubits():
    a = Circuit(3)
    assert a.nqubits == 3 and a.wire_names == [0, 1, 2]
    b = Circuit(3, wire_names=["a", "b", "c"])
    assert b.nqubits == 3 and b.wire_names == ["a", "b", "c"]
    c = Circuit(["a", "b", "c"])
    assert c.nqubits == 3 and c.wire_names == ["a", "b", "c"]
    d = Circuit(wire_names=["x", "y", "z"])
    assert d.nqubits == 3 and d.wire_names == ["x", "y", "z"]


def test_circuit_init_resolve_qubits_err():
    with pytest.raises(ValueError):
        a = Circuit()
    with pytest.raises(ValueError):
        b = Circuit(3, wire_names=["a", "b"])
    with pytest.raises(ValueError):
        c = Circuit(["a", "b", "c"], wire_names=["x", "y"])


def test_eigenstate(backend):
    nqubits = 3
    c = Circuit(nqubits)
    c.add(gates.M(*list(range(nqubits))))
    c2 = initialize(nqubits, eigenstate="-")
    assert backend.execute_circuit(c, nshots=100, initial_state=c2).frequencies() == {
        "111": 100
    }
    c2 = initialize(nqubits, eigenstate="+")
    assert backend.execute_circuit(c, nshots=100, initial_state=c2).frequencies() == {
        "000": 100
    }

    with pytest.raises(NotImplementedError):
        c2 = initialize(nqubits, eigenstate="x")


def test_initialize(backend):
    nqubits = 3
    for gate in [gates.X, gates.Y, gates.Z]:
        c = Circuit(nqubits)
        c.add(gates.M(*list(range(nqubits)), basis=gate))
        c2 = initialize(nqubits, basis=gate)
        assert backend.execute_circuit(
            c, nshots=100, initial_state=c2
        ).frequencies() == {"000": 100}


@pytest.mark.parametrize("nqubits", [0, -10, 2.5])
def test_circuit_init_errors(nqubits):
    with pytest.raises((ValueError, TypeError)):
        Circuit(nqubits)


def test_circuit_constructor():
    c = Circuit(5)
    assert isinstance(c, Circuit)
    assert not c.density_matrix
    c = Circuit(5, density_matrix=True)
    assert isinstance(c, Circuit)
    assert c.density_matrix
    c = Circuit(5, accelerators={"/GPU:0": 2})
    with pytest.raises(NotImplementedError):
        Circuit(5, accelerators={"/GPU:0": 2}, density_matrix=True)


def test_circuit_add():
    c = Circuit(2)
    g1, g2, g3 = gates.H(0), gates.H(1), gates.CNOT(0, 1)
    c.add(g1)
    c.add(g2)
    c.add(g3)
    assert c.depth == 2
    assert c.ngates == 3
    assert list(c.queue) == [g1, g2, g3]


def test_circuit_add_errors():
    c = Circuit(2)
    with pytest.raises(TypeError):
        c.add(0)
    with pytest.raises(ValueError):
        c.add(gates.H(2))
    c._final_state = 0
    with pytest.raises(RuntimeError):
        c.add(gates.H(1))


def test_circuit_add_iterable():
    c = Circuit(2)
    # adding list
    gatelist = [gates.H(0), gates.H(1), gates.CNOT(0, 1)]
    c.add(gatelist)
    assert c.depth == 2
    assert c.ngates == 3
    assert list(c.queue) == gatelist
    # adding tuple
    gatetuple = (gates.H(0), gates.H(1), gates.CNOT(0, 1))
    c.add(gatetuple)
    assert c.depth == 4
    assert c.ngates == 6
    assert isinstance(c.queue[-1], gates.CNOT)


def test_circuit_add_generator():
    """Check if `circuit.add` works with generators."""

    def gen():
        yield gates.H(0)
        yield gates.H(1)
        yield gates.CNOT(0, 1)

    c = Circuit(2)
    c.add(gen())
    assert c.depth == 2
    assert c.ngates == 3
    assert isinstance(c.queue[-1], gates.CNOT)


def test_circuit_add_nested_generator():
    def gen():
        yield gates.H(0)
        yield gates.H(1)
        yield gates.CNOT(0, 1)

    c = Circuit(2)
    c.add(gen() for _ in range(3))
    assert c.depth == 6
    assert c.ngates == 9
    assert isinstance(c.queue[2], gates.CNOT)
    assert isinstance(c.queue[5], gates.CNOT)
    assert isinstance(c.queue[7], gates.H)


def test_add_measurement():
    c = Circuit(5)
    g1 = gates.M(0, 2, register_name="a")
    g2 = gates.M(3, register_name="b")
    c.add([g1, g2])
    assert len(c.queue) == 2
    assert c.measurement_tuples == {"a": (0, 2), "b": (3,)}
    assert g1.target_qubits == (0, 2)
    assert g2.target_qubits == (3,)


def test_add_measurement_collapse():
    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.M(0, 1))
    c.add(gates.X(1))
    c.add(gates.M(1))
    c.add(gates.X(2))
    c.add(gates.M(2))
    assert len(c.queue) == 6
    # assert that the first measurement was switched to collapse automatically
    assert c.queue[1].collapse
    assert len(c.measurements) == 2


# :meth:`qibo.core.circuit.Circuit.fuse` is tested in `test_core_fusion.py`


def test_gate_types():
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.X(2))
    c.add(gates.CNOT(0, 2))
    c.add(gates.CNOT(1, 2))
    c.add(gates.TOFFOLI(0, 1, 2))
    target_counter = Counter({gates.H: 2, gates.X: 1, gates.CNOT: 2, gates.TOFFOLI: 1})
    assert c.ngates == 6
    assert c.gate_types == target_counter


def test_gate_names():
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.X(2))
    c.add(gates.CNOT(0, 2))
    c.add(gates.CNOT(1, 2))
    c.add(gates.TOFFOLI(0, 1, 2))
    target_counter = Counter({"h": 2, "x": 1, "cx": 2, "ccx": 1})
    assert c.ngates == 6
    assert c.gate_names == target_counter


def test_gates_of_type():
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 2))
    c.add(gates.X(1))
    c.add(gates.CNOT(1, 2))
    c.add(gates.TOFFOLI(0, 1, 2))
    c.add(gates.H(2))
    h_gates = c.gates_of_type(gates.H)
    cx_gates = c.gates_of_type("cx")
    assert h_gates == [(0, c.queue[0]), (1, c.queue[1]), (6, c.queue[6])]
    assert cx_gates == [(2, c.queue[2]), (4, c.queue[4])]
    with pytest.raises(TypeError):
        c.gates_of_type(5)


def test_summary():
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 2))
    c.add(gates.CNOT(1, 2))
    c.add(gates.TOFFOLI(0, 1, 2))
    c.add(gates.H(2))
    target_summary = "\n".join(
        [
            "Circuit depth = 5",
            "Total number of gates = 6",
            "Number of qubits = 3",
            "Most common gates:",
            "h: 3",
            "cx: 2",
            "ccx: 1",
        ]
    )
    assert c.summary() == target_summary


@pytest.mark.parametrize("measurements", [False, True])
def test_circuit_addition(measurements):
    c1 = Circuit(2)
    g1, g2 = gates.H(0), gates.CNOT(0, 1)
    c1.add(g1)
    c1.add(g2)
    if measurements:
        c1.add(gates.M(0, register_name="a"))

    c2 = Circuit(2)
    g3 = gates.H(1)
    c2.add(g3)
    if measurements:
        c2.add(gates.M(1, register_name="b"))

    c3 = c1 + c2
    assert c3.depth == 3 + int(measurements)
    if measurements:
        assert c3.measurement_tuples == {"a": (0,), "b": (1,)}


def test_circuit_addition_errors():
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.H(1))

    c2 = Circuit(1)
    c2.add(gates.X(0))

    with pytest.raises(ValueError):
        c1 + c2


def test_circuit_on_qubits():
    c = Circuit(3)
    c.add([gates.H(0), gates.X(1), gates.Y(2)])
    c.add([gates.CNOT(0, 1), gates.CZ(1, 2)])
    c.add(gates.H(1).controlled_by(0, 2))
    new_gates = list(c.on_qubits(2, 5, 4))
    assert new_gates[0].target_qubits == (2,)
    assert new_gates[1].target_qubits == (5,)
    assert new_gates[2].target_qubits == (4,)
    assert new_gates[3].target_qubits == (5,)
    assert new_gates[3].control_qubits == (2,)
    assert new_gates[4].target_qubits == (4,)
    assert new_gates[4].control_qubits == (5,)
    assert new_gates[5].target_qubits == (5,)
    assert new_gates[5].control_qubits == (2, 4)


def test_circuit_on_qubits_errors():
    smallc = Circuit(2)
    smallc.add(gates.H(i) for i in range(2))
    with pytest.raises(ValueError):
        next(smallc.on_qubits(0, 1, 2))

    from qibo.callbacks import Callback

    smallc = Circuit(4)
    smallc.add(gates.CallbackGate(Callback()))
    with pytest.raises(NotImplementedError):
        next(smallc.on_qubits(0, 1, 2, 3))


def test_circuit_serialization():
    c = Circuit(4)
    c.add(gates.RY(2, theta=0).controlled_by(0, 1))
    c.add(gates.RX(3, theta=0))
    c.add(gates.CNOT(1, 3))
    c.add(gates.RZ(2, theta=0).controlled_by(0, 3))

    raw = c.raw
    assert isinstance(raw, dict)
    assert Circuit.from_dict(raw).raw == raw

    c.add(gates.M(0))
    raw = c.raw
    assert isinstance(raw, dict)
    assert Circuit.from_dict(raw).raw == raw


def test_circuit_serialization_with_wire_names():

    wire_names = ["a", "b"]
    c = Circuit(2, wire_names=wire_names)
    raw = c.raw
    assert "wire_names" in raw
    new_c = Circuit.from_dict(raw)
    assert new_c.wire_names == c.wire_names

    transpiler = Passes(passes=[Sabre()], connectivity=Graph([wire_names]))

    c, _ = transpiler(c)
    new_c, _ = transpiler(new_c)
    assert new_c.wire_names == c.wire_names

    with pytest.raises(PlacementError):
        c.wire_names = ["c", "b"]
        transpiler(c)


def test_circuit_light_cone():
    from qibo import __version__

    nqubits = 10
    c = Circuit(nqubits)
    c.add(gates.RY(i, theta=0) for i in range(nqubits))
    c.add(gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2))
    c.add(gates.RY(i, theta=0) for i in range(nqubits))
    c.add(gates.CZ(i, i + 1) for i in range(1, nqubits - 1, 2))
    c.add(gates.CZ(0, nqubits - 1))
    sc, qubit_map = c.light_cone(4, 5)
    target_qasm = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
ry(0.0) q[0];
ry(0.0) q[1];
ry(0.0) q[2];
ry(0.0) q[3];
ry(0.0) q[4];
ry(0.0) q[5];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
ry(0.0) q[1];
ry(0.0) q[2];
ry(0.0) q[3];
ry(0.0) q[4];
cz q[1],q[2];
cz q[3],q[4];"""
    assert qubit_map == {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    assert sc.to_qasm() == target_qasm


def test_circuit_light_cone_controlled_by():
    c = Circuit(4)
    c.add(gates.RY(2, theta=0).controlled_by(0, 1))
    c.add(gates.RX(3, theta=0))
    sc, qubit_map = c.light_cone(3)
    assert qubit_map == {3: 0}
    assert sc.nqubits == 1
    assert len(sc.queue) == 1
    assert isinstance(sc.queue[0], gates.RX)
    sc, qubit_map = c.light_cone(2)
    assert qubit_map == {0: 0, 1: 1, 2: 2}
    assert sc.nqubits == 3
    assert len(sc.queue) == 1
    assert isinstance(sc.queue[0], gates.RY)


@pytest.mark.parametrize("deep", [False, True])
def test_circuit_copy(deep):
    c1 = Circuit(2)
    c1.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    c2 = c1.copy(deep)
    assert c1.depth == 2
    assert c2.depth == 2
    assert c1.ngates == 3
    assert c2.ngates == 3
    assert c1.nqubits == 2
    assert c2.nqubits == 2
    for g1, g2 in zip(c1.queue, c2.queue):
        if deep:
            assert g1.__class__ == g2.__class__
            assert g1.target_qubits == g2.target_qubits
            assert g1.control_qubits == g2.control_qubits
        else:
            assert g1 is g2


@pytest.mark.parametrize("deep", [False, True])
def test_circuit_copy_with_measurements(deep):
    c1 = Circuit(4)
    c1.add([gates.H(0), gates.H(3), gates.CNOT(0, 2)])
    c1.add(gates.M(0, 1, register_name="a"))
    c1.add(gates.M(3, register_name="b"))
    c2 = c1.copy(deep)
    assert c1.nqubits == 4
    assert c2.nqubits == 4
    assert c1.depth == 3
    assert c2.depth == 3
    assert c1.ngates == 5
    assert c2.ngates == 5
    assert c2.measurement_tuples == {"a": (0, 1), "b": (3,)}


@pytest.mark.parametrize("measurements", [False, True])
def test_circuit_invert(measurements):
    c = Circuit(3)
    gatelist = [gates.H(0), gates.X(1), gates.Y(2), gates.CNOT(0, 1), gates.CZ(1, 2)]
    c.add(gatelist)
    if measurements:
        c.add(gates.M(0, 2))
    invc = c.invert()
    for g1, g2 in zip(invc.queue, gatelist[::-1]):
        g2 = g2.dagger()
        assert isinstance(g1, g2.__class__)
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    if measurements:
        assert invc.measurement_tuples == {"register0": (0, 2)}


@pytest.mark.parametrize("measurements", [False, True])
def test_circuit_decompose(measurements):
    c = Circuit(4)
    c.add([gates.H(0), gates.X(1), gates.Y(2)])
    c.add([gates.CZ(0, 1), gates.CNOT(2, 3), gates.TOFFOLI(0, 1, 3)])
    if measurements:
        c.add(gates.M(0, 2))
    decompc = c.decompose()

    dgates = []
    for gate in c.queue:
        dgates.extend(gate.decompose())
    for g1, g2 in zip(decompc.queue, dgates):
        assert isinstance(g1, g2.__class__)
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    if measurements:
        assert decompc.measurement_tuples == {"register0": (0, 2)}


@pytest.mark.parametrize("measurements", [False, True])
@pytest.mark.parametrize(
    "noise_map",
    [
        list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.3])),
        {0: list(zip(["X", "Z"], [0.1, 0.2])), 1: list(zip(["Y", "Z"], [0.2, 0.1]))},
    ],
)
def test_circuit_with_pauli_noise(measurements, noise_map):
    with pytest.raises(TypeError):
        n_map = ["X", 0.2]
        circuit = Circuit(1)
        circuit.with_pauli_noise(n_map)

    c = Circuit(2)
    c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    if measurements:
        c.add(gates.M(0, 1))
    noisyc = c.with_pauli_noise(noise_map)

    if not isinstance(noise_map, dict):
        noise_map = {0: noise_map, 1: noise_map}
    targetc = Circuit(2)
    targetc.add(gates.H(0))
    targetc.add(gates.PauliNoiseChannel(0, noise_map[0]))
    targetc.add(gates.H(1))
    targetc.add(gates.PauliNoiseChannel(1, noise_map[1]))
    targetc.add(gates.CNOT(0, 1))
    targetc.add(gates.PauliNoiseChannel(0, noise_map[0]))
    targetc.add(gates.PauliNoiseChannel(1, noise_map[1]))
    for g1, g2 in zip(noisyc.queue, targetc.queue):
        assert isinstance(g1, g2.__class__)
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    if measurements:
        assert noisyc.measurement_tuples == {"register0": (0, 1)}


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("include_not_trainable", [True, False])
@pytest.mark.parametrize("format", ["list", "dict", "flatlist"])
def test_get_parameters(trainable, include_not_trainable, format):
    matrix = np.random.random((2, 2))
    c = Circuit(3)
    c.add(gates.RX(0, theta=0.123))
    c.add(gates.RY(1, theta=0.456, trainable=trainable))
    c.add(gates.CZ(1, 2))
    c.add(gates.Unitary(matrix, 2))
    c.add(gates.fSim(0, 2, theta=0.789, phi=0.987, trainable=trainable))
    c.add(gates.H(2))
    params = c.get_parameters(format, include_not_trainable)
    if trainable or include_not_trainable:
        target_params = {
            "list": [(0.123,), (0.456,), (0.789, 0.987)],
            "dict": {
                c.queue[0]: (0.123,),
                c.queue[1]: (0.456,),
                c.queue[4]: (0.789, 0.987),
            },
            "flatlist": [0.123, 0.456],
        }
        target_params["flatlist"].extend(list(matrix.ravel()))
        target_params["flatlist"].extend([0.789, 0.987])
    else:
        target_params = {
            "list": [(0.123,)],
            "dict": {c.queue[0]: (0.123,)},
            "flatlist": [0.123],
        }
        target_params["flatlist"].extend(list(matrix.ravel()))
    if format == "list":
        i = len(target_params["list"]) // 2 + 1
        np.testing.assert_allclose(params.pop(i)[0], matrix)
    elif format == "dict":
        np.testing.assert_allclose(params.pop(c.queue[3])[0], matrix)
    assert params == target_params[format]
    with pytest.raises(ValueError):
        c.get_parameters("test")


@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_with_list(trainable):
    """Check updating parameters of circuit with list."""
    params = [0.123, 0.456, (0.789, 0.321)]

    c = Circuit(3)
    if trainable:
        c.add(gates.RX(0, theta=0, trainable=trainable))
    else:
        c.add(gates.RX(0, theta=params[0], trainable=trainable))
    c.add(gates.RY(1, theta=0))
    c.add(gates.CZ(1, 2))
    c.add(gates.fSim(0, 2, theta=0, phi=0))
    c.add(gates.H(2))
    if trainable:
        c.set_parameters(params)
        assert c.queue[0].parameters == (params[0],)
    else:
        c.set_parameters(params[1:])
    assert c.queue[1].parameters == (params[1],)
    assert c.queue[3].parameters == params[2]


@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_with_dictionary(trainable):
    """Check updating parameters of circuit with list."""
    params = [0.123, 0.456, 0.789]

    c1 = Circuit(3)
    c1.add(gates.X(0))
    c1.add(gates.X(2))
    if trainable:
        c1.add(gates.U1(0, theta=0, trainable=trainable))
    else:
        c1.add(gates.U1(0, theta=params[0], trainable=trainable))
    c2 = Circuit(3)
    c2.add(gates.RZ(1, theta=0))
    c2.add(gates.CZ(1, 2))
    c2.add(gates.CU1(0, 2, theta=0))
    c2.add(gates.H(2))
    c = c1 + c2

    if trainable:
        params_dict = {c.queue[i]: p for i, p in zip([2, 3, 5], params)}
        c.set_parameters(params_dict)
        assert c.queue[2].parameters == (params[0],)
    else:
        params_dict = {c.queue[3]: params[1], c.queue[5]: params[2]}
        c.set_parameters(params_dict)
    assert c.queue[3].parameters == (params[1],)
    assert c.queue[5].parameters == (params[2],)

    # test not passing all parametrized gates
    c.set_parameters({c.queue[5]: 0.7891})
    if trainable:
        assert c.queue[2].parameters == (params[0],)
    assert c.queue[3].parameters == (params[1],)
    assert c.queue[5].parameters == (0.7891,)


def test_circuit_set_parameters_errors():
    """Check updating parameters errors."""
    c = Circuit(2)
    c.add(gates.RX(0, theta=0.789))
    c.add(gates.RX(1, theta=0.789))
    c.add(gates.fSim(0, 1, theta=0.123, phi=0.456))

    with pytest.raises(KeyError):
        c.set_parameters({gates.RX(0, theta=1.0): 0.568})
    with pytest.raises(ValueError):
        c.set_parameters([0.12586])
    with pytest.raises(TypeError):
        c.set_parameters({0.3568})
    with pytest.raises(ValueError):
        c.queue[2].parameters = [0.1234, 0.4321, 0.156]


def test_circuit_draw():
    """Test circuit text draw."""
    ref = (
        "q0: ─H─U1─U1─U1─U1───────────────────────────x───\n"
        "q1: ───o──|──|──|──H─U1─U1─U1────────────────|─x─\n"
        "q2: ──────o──|──|────o──|──|──H─U1─U1────────|─|─\n"
        "q3: ─────────o──|───────o──|────o──|──H─U1───|─x─\n"
        "q4: ────────────o──────────o───────o────o──H─x───"
    )
    circuit = Circuit(5, wire_names=["q0", "q1", "q2", "q3", "q4"])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))

    assert str(circuit) == ref


def test_circuit_wire_names():
    circuit = Circuit(5)
    assert circuit.wire_names == [0, 1, 2, 3, 4]
    assert circuit._wire_names == None

    circuit.wire_names = ["a", "b", "c", "d", "e"]
    assert circuit.wire_names == ["a", "b", "c", "d", "e"]
    assert circuit._wire_names == ["a", "b", "c", "d", "e"]

    with pytest.raises(TypeError):
        circuit.wire_names = 5
    with pytest.raises(ValueError):
        circuit.wire_names = ["a", "b", "c", "d"]


def test_circuit_draw_wire_names():
    """Test circuit text draw."""
    ref = (
        "a    : ─H─U1─U1─U1─U1───────────────────────────x───\n"
        + "b    : ───o──|──|──|──H─U1─U1─U1────────────────|─x─\n"
        + "hello: ──────o──|──|────o──|──|──H─U1─U1────────|─|─\n"
        + "1    : ─────────o──|───────o──|────o──|──H─U1───|─x─\n"
        + "q4   : ────────────o──────────o───────o────o──H─x───"
    )
    circuit = Circuit(5, wire_names=["a", "b", "hello", "1", "q4"])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))

    assert str(circuit) == ref


def test_circuit_draw_wire_names_int():
    ref = (
        "2133: ─H─U1─U1─U1─U1───────────────────────────x───\n"
        + "8   : ───o──|──|──|──H─U1─U1─U1────────────────|─x─\n"
        + "2319: ──────o──|──|────o──|──|──H─U1─U1────────|─|─\n"
        + "0   : ─────────o──|───────o──|────o──|──H─U1───|─x─\n"
        + "1908: ────────────o──────────o───────o────o──H─x───"
    )
    circuit = Circuit(5, wire_names=[2133, 8, 2319, 0, 1908])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    assert str(circuit) == ref


def test_circuit_draw_line_wrap(capsys):
    """Test circuit text draw with line wrap."""
    ref_line_wrap_50 = (
        "q0:     ─H─U1─U1─U1─U1───────────────────────────x───I───f ...\n"
        + "q1:     ───o──|──|──|──H─U1─U1─U1────────────────|─x─I───| ...\n"
        + "q2:     ──────o──|──|────o──|──|──H─U1─U1────────|─|─────| ...\n"
        + "q3:     ─────────o──|───────o──|────o──|──H─U1───|─x───M─| ...\n"
        + "q4:     ────────────o──────────o───────o────o──H─x───────f ...\n"
        + "\n"
        + "q0: ... ─o────gf───M─\n"
        + "q1: ... ─U3───|──o─M─\n"
        + "q2: ... ────X─gf─o─M─\n"
        + "q3: ... ────o────o───\n"
        + "q4: ... ────o────X───"
    )

    ref_line_wrap_30 = (
        "q0:     ─H─U1─U1─U1─U1──────────────── ...\n"
        + "q1:     ───o──|──|──|──H─U1─U1─U1───── ...\n"
        + "q2:     ──────o──|──|────o──|──|──H─U1 ...\n"
        + "q3:     ─────────o──|───────o──|────o─ ...\n"
        + "q4:     ────────────o──────────o────── ...\n"
        + "\n"
        + "q0: ... ───────────x───I───f─o────gf── ...\n"
        + "q1: ... ───────────|─x─I───|─U3───|──o ...\n"
        + "q2: ... ─U1────────|─|─────|────X─gf─o ...\n"
        + "q3: ... ─|──H─U1───|─x───M─|────o────o ...\n"
        + "q4: ... ─o────o──H─x───────f────o────X ...\n"
        + "\n"
        + "q0: ... ─M─\n"
        + "q1: ... ─M─\n"
        + "q2: ... ─M─\n"
        + "q3: ... ───\n"
        + "q4: ... ───"
    )

    circuit = Circuit(5, wire_names=["q0", "q1", "q2", "q3", "q4"])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    circuit.add(gates.I(*range(2)))
    circuit.add(gates.M(3, collapse=True))
    circuit.add(gates.fSim(0, 4, 0, 0))
    circuit.add(gates.CU3(0, 1, 0, 0, 0))
    circuit.add(gates.TOFFOLI(4, 3, 2))
    circuit.add(gates.GeneralizedfSim(0, 2, np.eye(2), 0))
    circuit.add(gates.X(4).controlled_by(1, 2, 3))
    circuit.add(gates.M(*range(3)))

    circuit.draw(line_wrap=50)
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == ref_line_wrap_50

    circuit.draw(line_wrap=30)
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == ref_line_wrap_30


def test_circuit_draw_line_wrap_names(capsys):
    """Test circuit text draw with line wrap."""
    ref_line_wrap_50 = (
        "q0:     ─H─U1─U1─U1─U1───────────────────────────x───I───f ...\n"
        + "a :     ───o──|──|──|──H─U1─U1─U1────────────────|─x─I───| ...\n"
        + "q2:     ──────o──|──|────o──|──|──H─U1─U1────────|─|─────| ...\n"
        + "q3:     ─────────o──|───────o──|────o──|──H─U1───|─x───M─| ...\n"
        + "q4:     ────────────o──────────o───────o────o──H─x───────f ...\n"
        + "\n"
        + "q0: ... ─o────gf───M─\n"
        + "a : ... ─U3───|──o─M─\n"
        + "q2: ... ────X─gf─o─M─\n"
        + "q3: ... ────o────o───\n"
        + "q4: ... ────o────X───"
    )

    ref_line_wrap_30 = (
        "q0:     ─H─U1─U1─U1─U1──────────────── ...\n"
        + "a :     ───o──|──|──|──H─U1─U1─U1───── ...\n"
        + "q2:     ──────o──|──|────o──|──|──H─U1 ...\n"
        + "q3:     ─────────o──|───────o──|────o─ ...\n"
        + "q4:     ────────────o──────────o────── ...\n"
        + "\n"
        + "q0: ... ───────────x───I───f─o────gf── ...\n"
        + "a : ... ───────────|─x─I───|─U3───|──o ...\n"
        + "q2: ... ─U1────────|─|─────|────X─gf─o ...\n"
        + "q3: ... ─|──H─U1───|─x───M─|────o────o ...\n"
        + "q4: ... ─o────o──H─x───────f────o────X ...\n"
        + "\n"
        + "q0: ... ─M─\n"
        + "a : ... ─M─\n"
        + "q2: ... ─M─\n"
        + "q3: ... ───\n"
        + "q4: ... ───"
    )

    circuit = Circuit(5, wire_names=["q0", "a", "q2", "q3", "q4"])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    circuit.add(gates.I(*range(2)))
    circuit.add(gates.M(3, collapse=True))
    circuit.add(gates.fSim(0, 4, 0, 0))
    circuit.add(gates.CU3(0, 1, 0, 0, 0))
    circuit.add(gates.TOFFOLI(4, 3, 2))
    circuit.add(gates.GeneralizedfSim(0, 2, np.eye(2), 0))
    circuit.add(gates.X(4).controlled_by(1, 2, 3))
    circuit.add(gates.M(*range(3)))

    circuit.draw(line_wrap=50)
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == ref_line_wrap_50

    circuit.draw(line_wrap=30)
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == ref_line_wrap_30


@pytest.mark.parametrize("legend", [True, False])
def test_circuit_draw_channels(capsys, legend):
    """Check that channels are drawn correctly."""

    circuit = Circuit(2, density_matrix=True, wire_names=["q0", "q1"])
    circuit.add(gates.H(0))
    circuit.add(gates.PauliNoiseChannel(0, list(zip(["X", "Z"], [0.1, 0.2]))))
    circuit.add(gates.H(1))
    circuit.add(gates.PauliNoiseChannel(1, list(zip(["Y", "Z"], [0.2, 0.1]))))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.PauliNoiseChannel(0, list(zip(["X", "Z"], [0.1, 0.2]))))
    circuit.add(gates.PauliNoiseChannel(1, list(zip(["Y", "Z"], [0.2, 0.1]))))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.DepolarizingChannel((0, 1), 0.1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.DepolarizingChannel((0,), 0.1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.DepolarizingChannel((1,), 0.1))

    ref = "q0: ─H─PN─o─PN─o─D─o─D─o───\n" + "q1: ─H─PN─X─PN─X─D─X───X─D─"

    if legend:
        ref += (
            "\n\n Legend for callbacks and channels: \n"
            + "| Gate                | Symbol   |\n"
            + "|---------------------+----------|\n"
            + "| DepolarizingChannel | D        |\n"
            + "| PauliNoiseChannel   | PN       |"
        )

    circuit.draw(legend=legend)
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == ref


@pytest.mark.parametrize("legend", [True, False])
def test_circuit_draw_callbacks(capsys, legend):
    """Check that callbacks are drawn correcly."""
    from qibo.callbacks import EntanglementEntropy

    entropy = EntanglementEntropy([0])
    c = Circuit(2, wire_names=["q0", "q1"])
    c.add(gates.CallbackGate(entropy))
    c.add(gates.H(0))
    c.add(gates.CallbackGate(entropy))
    c.add(gates.CNOT(0, 1))
    c.add(gates.CallbackGate(entropy))

    ref = "q0: ─EE─H─EE─o─EE─\n" + "q1: ─EE───EE─X─EE─"

    if legend:
        ref += (
            "\n\n Legend for callbacks and channels: \n"
            + "| Gate                | Symbol   |\n"
            + "|---------------------+----------|\n"
            + "| EntanglementEntropy | EE       |"
        )

    c.draw(legend=legend)
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == ref


def test_circuit_draw_labels():
    """Test circuit text draw."""
    ref = (
        "q0: ─H─G1─G2─G3─G4───────────────────────────x───\n"
        + "q1: ───o──|──|──|──H─G2─G3─G4────────────────|─x─\n"
        + "q2: ──────o──|──|────o──|──|──H─G3─G4────────|─|─\n"
        + "q3: ─────────o──|───────o──|────o──|──H─G4───|─x─\n"
        + "q4: ────────────o──────────o───────o────o──H─x───"
    )
    circuit = Circuit(5, wire_names=["q0", "q1", "q2", "q3", "q4"])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            gate = gates.CNOT(i2, i1)
            gate.draw_label = f"G{i2}"
            circuit.add(gate)
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    assert str(circuit).rstrip("\n") == ref


def test_circuit_draw_names(capsys):
    """Test circuit text draw."""
    ref = (
        "q0: ─H─cx─cx─cx─cx───────────────────────────x───\n"
        + "q1: ───o──|──|──|──H─cx─cx─cx────────────────|─x─\n"
        + "q2: ──────o──|──|────o──|──|──H─cx─cx────────|─|─\n"
        + "q3: ─────────o──|───────o──|────o──|──H─cx───|─x─\n"
        + "q4: ────────────o──────────o───────o────o──H─x───"
    )
    circuit = Circuit(5, wire_names=["q0", "q1", "q2", "q3", "q4"])
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            gate = gates.CNOT(i2, i1)
            gate.draw_label = ""
            circuit.add(gate)
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    circuit.draw()
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == ref


def test_circuit_draw_error():
    """Test NotImplementedError in circuit draw."""
    circuit = Circuit(1)
    error_gate = gates.X(0)
    error_gate.name = ""
    error_gate.draw_label = ""
    circuit.add(error_gate)

    with pytest.raises(NotImplementedError):
        circuit.draw()
