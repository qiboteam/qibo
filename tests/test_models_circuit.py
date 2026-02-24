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
    circuit = Circuit(2)
    assert circuit.nqubits == 2


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
    circuit = Circuit(["a", "b", "c"])
    assert circuit.nqubits == 3 and circuit.wire_names == ["a", "b", "c"]
    d = Circuit(wire_names=["x", "y", "z"])
    assert d.nqubits == 3 and d.wire_names == ["x", "y", "z"]


def test_circuit_init_resolve_qubits_err():
    with pytest.raises(ValueError):
        a = Circuit()
    with pytest.raises(ValueError):
        b = Circuit(3, wire_names=["a", "b"])
    with pytest.raises(ValueError):
        circuit = Circuit(["a", "b", "c"], wire_names=["x", "y"])


def test_eigenstate(backend):
    nqubits = 3
    circuit = Circuit(nqubits)
    circuit.add(gates.M(*list(range(nqubits))))
    c2 = initialize(nqubits, eigenstate="-")
    assert backend.execute_circuit(
        circuit, nshots=100, initial_state=c2
    ).frequencies() == {"111": 100}
    c2 = initialize(nqubits, eigenstate="+")
    assert backend.execute_circuit(
        circuit, nshots=100, initial_state=c2
    ).frequencies() == {"000": 100}

    with pytest.raises(NotImplementedError):
        c2 = initialize(nqubits, eigenstate="x")


def test_initialize(backend):
    nqubits = 3
    for gate in [gates.X, gates.Y, gates.Z]:
        circuit = Circuit(nqubits)
        circuit.add(gates.M(*list(range(nqubits)), basis=gate))
        c2 = initialize(nqubits, basis=gate)
        assert backend.execute_circuit(
            circuit, nshots=100, initial_state=c2
        ).frequencies() == {"000": 100}


@pytest.mark.parametrize("nqubits", [0, -10, 2.5])
def test_circuit_init_errors(nqubits):
    with pytest.raises((ValueError, TypeError)):
        Circuit(nqubits)


def test_circuit_constructor():
    circuit = Circuit(5)
    assert isinstance(circuit, Circuit)
    assert not circuit.density_matrix
    circuit = Circuit(5, density_matrix=True)
    assert isinstance(circuit, Circuit)
    assert circuit.density_matrix
    circuit = Circuit(5, accelerators={"/GPU:0": 2})
    with pytest.raises(NotImplementedError):
        Circuit(5, accelerators={"/GPU:0": 2}, density_matrix=True)


def test_circuit_add():
    circuit = Circuit(2)
    g1, g2, g3 = gates.H(0), gates.H(1), gates.CNOT(0, 1)
    circuit.add(g1)
    circuit.add(g2)
    circuit.add(g3)
    assert circuit.depth == 2
    assert circuit.ngates == 3
    assert list(circuit.queue) == [g1, g2, g3]


def test_circuit_add_errors():
    circuit = Circuit(2)
    with pytest.raises(TypeError):
        circuit.add(0)
    with pytest.raises(ValueError):
        circuit.add(gates.H(2))
    circuit._final_state = 0
    with pytest.raises(RuntimeError):
        circuit.add(gates.H(1))


def test_circuit_add_iterable():
    circuit = Circuit(2)
    # adding list
    gatelist = [gates.H(0), gates.H(1), gates.CNOT(0, 1)]
    circuit.add(gatelist)
    assert circuit.depth == 2
    assert circuit.ngates == 3
    assert list(circuit.queue) == gatelist
    # adding tuple
    gatetuple = (gates.H(0), gates.H(1), gates.CNOT(0, 1))
    circuit.add(gatetuple)
    assert circuit.depth == 4
    assert circuit.ngates == 6
    assert isinstance(circuit.queue[-1], gates.CNOT)


def test_circuit_add_generator():
    """Check if `circuit.add` works with generators."""

    def gen():
        yield gates.H(0)
        yield gates.H(1)
        yield gates.CNOT(0, 1)

    circuit = Circuit(2)
    circuit.add(gen())
    assert circuit.depth == 2
    assert circuit.ngates == 3
    assert isinstance(circuit.queue[-1], gates.CNOT)


def test_circuit_add_nested_generator():
    def gen():
        yield gates.H(0)
        yield gates.H(1)
        yield gates.CNOT(0, 1)

    circuit = Circuit(2)
    circuit.add(gen() for _ in range(3))
    assert circuit.depth == 6
    assert circuit.ngates == 9
    assert isinstance(circuit.queue[2], gates.CNOT)
    assert isinstance(circuit.queue[5], gates.CNOT)
    assert isinstance(circuit.queue[7], gates.H)


def test_add_measurement():
    circuit = Circuit(5)
    g1 = gates.M(0, 2, register_name="a")
    g2 = gates.M(3, register_name="b")
    circuit.add([g1, g2])
    assert len(circuit.queue) == 2
    assert circuit.measurement_tuples == {"a": (0, 2), "b": (3,)}
    assert g1.target_qubits == (0, 2)
    assert g2.target_qubits == (3,)


def test_add_measurement_collapse():
    circuit = Circuit(3)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0, 1))
    circuit.add(gates.X(1))
    circuit.add(gates.M(1))
    circuit.add(gates.X(2))
    circuit.add(gates.M(2))
    assert len(circuit.queue) == 6
    # assert that the first measurement was switched to collapse automatically
    assert circuit.queue[1].collapse
    assert len(circuit.measurements) == 2


# :meth:`qibo.core.circuit.Circuit.fuse` is tested in `test_core_fusion.py`


def test_gate_types():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.X(2))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    target_counter = Counter({gates.H: 2, gates.X: 1, gates.CNOT: 2, gates.TOFFOLI: 1})
    assert circuit.ngates == 6
    assert circuit.gate_types == target_counter


def test_gate_names():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.X(2))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    target_counter = Counter({"h": 2, "x": 1, "cx": 2, "ccx": 1})
    assert circuit.ngates == 6
    assert circuit.gate_names == target_counter


def test_gates_of_type():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.X(1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.H(2))
    h_gates = circuit.gates_of_type(gates.H)
    cx_gates = circuit.gates_of_type("cx")
    assert h_gates == [
        (0, circuit.queue[0]),
        (1, circuit.queue[1]),
        (6, circuit.queue[6]),
    ]
    assert cx_gates == [(2, circuit.queue[2]), (4, circuit.queue[4])]
    with pytest.raises(TypeError):
        circuit.gates_of_type(5)


def test_summary(capsys):
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.H(2))
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
    circuit.summary()
    out, _ = capsys.readouterr()
    assert out.rstrip("\n") == target_summary


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
    circuit = Circuit(3)
    circuit.add([gates.H(0), gates.X(1), gates.Y(2)])
    circuit.add([gates.CNOT(0, 1), gates.CZ(1, 2)])
    circuit.add(gates.H(1).controlled_by(0, 2))
    new_gates = list(circuit.on_qubits(2, 5, 4))
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
    small_circuit = Circuit(2)
    small_circuit.add(gates.H(i) for i in range(2))
    with pytest.raises(ValueError):
        next(small_circuit.on_qubits(0, 1, 2))

    from qibo.callbacks import Callback

    small_circuit = Circuit(4)
    small_circuit.add(gates.CallbackGate(Callback()))
    with pytest.raises(NotImplementedError):
        next(small_circuit.on_qubits(0, 1, 2, 3))


def test_circuit_serialization():
    circuit = Circuit(4)
    circuit.add(gates.RY(2, theta=0).controlled_by(0, 1))
    circuit.add(gates.RX(3, theta=0))
    circuit.add(gates.CNOT(1, 3))
    circuit.add(gates.RZ(2, theta=0).controlled_by(0, 3))

    raw = circuit.raw
    assert isinstance(raw, dict)
    assert Circuit.from_dict(raw).raw == raw

    circuit.add(gates.M(0))
    raw = circuit.raw
    assert isinstance(raw, dict)
    assert Circuit.from_dict(raw).raw == raw


def test_circuit_serialization_with_wire_names():

    wire_names = ["a", "b"]
    circuit = Circuit(2, wire_names=wire_names)
    raw = circuit.raw
    assert "wire_names" in raw
    new_circuit = Circuit.from_dict(raw)
    assert new_circuit.wire_names == circuit.wire_names

    transpiler = Passes(passes=[Sabre()], connectivity=Graph([wire_names]))

    circuit, _ = transpiler(circuit)
    new_circuit, _ = transpiler(new_circuit)
    assert new_circuit.wire_names == circuit.wire_names

    with pytest.raises(PlacementError):
        circuit.wire_names = ["c", "b"]
        transpiler(circuit)


def test_circuit_light_cone():
    from qibo import __version__

    nqubits = 10
    circuit = Circuit(nqubits)
    circuit.add(gates.RY(i, theta=0) for i in range(nqubits))
    circuit.add(gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2))
    circuit.add(gates.RY(i, theta=0) for i in range(nqubits))
    circuit.add(gates.CZ(i, i + 1) for i in range(1, nqubits - 1, 2))
    circuit.add(gates.CZ(0, nqubits - 1))
    sc, qubit_map = circuit.light_cone(4, 5)
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
    circuit = Circuit(4)
    circuit.add(gates.RY(2, theta=0).controlled_by(0, 1))
    circuit.add(gates.RX(3, theta=0))
    sc, qubit_map = circuit.light_cone(3)
    assert qubit_map == {3: 0}
    assert sc.nqubits == 1
    assert len(sc.queue) == 1
    assert isinstance(sc.queue[0], gates.RX)
    sc, qubit_map = circuit.light_cone(2)
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
    circuit = Circuit(3)
    gatelist = [gates.H(0), gates.X(1), gates.Y(2), gates.CNOT(0, 1), gates.CZ(1, 2)]
    circuit.add(gatelist)
    if measurements:
        circuit.add(gates.M(0, 2))
    inv_circuit = circuit.invert()
    for g1, g2 in zip(inv_circuit.queue, gatelist[::-1]):
        g2 = g2.dagger()
        assert isinstance(g1, g2.__class__)
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    if measurements:
        assert inv_circuit.measurement_tuples == {"register0": (0, 2)}


@pytest.mark.parametrize("measurements", [False, True])
def test_circuit_decompose(measurements):
    circuit = Circuit(4)
    circuit.add([gates.H(0), gates.X(1), gates.Y(2)])
    circuit.add([gates.CZ(0, 1), gates.CNOT(2, 3), gates.TOFFOLI(0, 1, 3)])
    if measurements:
        circuit.add(gates.M(0, 2))
    decomp_circuitircuit = circuit.decompose()

    dgates = []
    for gate in circuit.queue:
        dgates.extend(gate.decompose())
    for g1, g2 in zip(decomp_circuitircuit.queue, dgates):
        assert isinstance(g1, g2.__class__)
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    if measurements:
        assert decomp_circuitircuit.measurement_tuples == {"register0": (0, 2)}


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

    circuit = Circuit(2)
    circuit.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    if measurements:
        circuit.add(gates.M(0, 1))
    noisycircuit = circuit.with_pauli_noise(noise_map)

    if not isinstance(noise_map, dict):
        noise_map = {0: noise_map, 1: noise_map}
    targetcircuit = Circuit(2)
    targetcircuit.add(gates.H(0))
    targetcircuit.add(gates.PauliNoiseChannel(0, noise_map[0]))
    targetcircuit.add(gates.H(1))
    targetcircuit.add(gates.PauliNoiseChannel(1, noise_map[1]))
    targetcircuit.add(gates.CNOT(0, 1))
    targetcircuit.add(gates.PauliNoiseChannel(0, noise_map[0]))
    targetcircuit.add(gates.PauliNoiseChannel(1, noise_map[1]))
    for g1, g2 in zip(noisycircuit.queue, targetcircuit.queue):
        assert isinstance(g1, g2.__class__)
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    if measurements:
        assert noisycircuit.measurement_tuples == {"register0": (0, 1)}


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("include_not_trainable", [True, False])
@pytest.mark.parametrize("output_format", ["list", "dict", "flatlist"])
def test_get_parameters(trainable, include_not_trainable, output_format):
    matrix = np.random.random((2, 2))

    circuit = Circuit(3)
    circuit.add(gates.RX(0, theta=0.123))
    circuit.add(gates.RY(1, theta=0.456, trainable=trainable))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.Unitary(matrix, 2))
    circuit.add(gates.fSim(0, 2, theta=0.789, phi=0.987, trainable=trainable))
    circuit.add(gates.H(2))

    params = circuit.get_parameters(output_format, include_not_trainable)

    if trainable or include_not_trainable:
        target_params = {
            "list": [(0.123,), (0.456,), (0.789, 0.987)],
            "dict": {
                circuit.queue[0]: (0.123,),
                circuit.queue[1]: (0.456,),
                circuit.queue[4]: (0.789, 0.987),
            },
            "flatlist": [0.123, 0.456],
        }
        target_params["flatlist"].extend(list(matrix.ravel()))
        target_params["flatlist"].extend([0.789, 0.987])
    else:
        target_params = {
            "list": [(0.123,)],
            "dict": {circuit.queue[0]: (0.123,)},
            "flatlist": [0.123],
        }
        target_params["flatlist"].extend(list(matrix.ravel()))

    if output_format == "list":
        i = len(target_params["list"]) // 2 + 1
        np.testing.assert_allclose(params.pop(i)[0], matrix)
    elif output_format == "dict":
        np.testing.assert_allclose(params.pop(circuit.queue[3])[0], matrix)

    assert params == target_params[output_format]

    with pytest.raises(ValueError):
        circuit.get_parameters("test")


def test_get_parameters_0_dimensional_tensor():
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 0))
    circuit.set_parameters([np.array(1)])
    assert circuit.get_parameters(output_format="flatlist") == [np.array(1)]


@pytest.mark.parametrize("trainable", [True, False])
def test_circuit_set_parameters_with_list(trainable):
    """Check updating parameters of circuit with list."""
    params = [0.123, 0.456, (0.789, 0.321)]

    circuit = Circuit(3)
    if trainable:
        circuit.add(gates.RX(0, theta=0, trainable=trainable))
    else:
        circuit.add(gates.RX(0, theta=params[0], trainable=trainable))
    circuit.add(gates.RY(1, theta=0))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.fSim(0, 2, theta=0, phi=0))
    circuit.add(gates.H(2))
    if trainable:
        circuit.set_parameters(params)
        assert circuit.queue[0].parameters == (params[0],)
    else:
        circuit.set_parameters(params[1:])
    assert circuit.queue[1].parameters == (params[1],)
    assert circuit.queue[3].parameters == params[2]


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
    circuit = c1 + c2

    if trainable:
        params_dict = {circuit.queue[i]: p for i, p in zip([2, 3, 5], params)}
        circuit.set_parameters(params_dict)
        assert circuit.queue[2].parameters == (params[0],)
    else:
        params_dict = {circuit.queue[3]: params[1], circuit.queue[5]: params[2]}
        circuit.set_parameters(params_dict)
    assert circuit.queue[3].parameters == (params[1],)
    assert circuit.queue[5].parameters == (params[2],)

    # test not passing all parametrized gates
    circuit.set_parameters({circuit.queue[5]: 0.7891})
    if trainable:
        assert circuit.queue[2].parameters == (params[0],)
    assert circuit.queue[3].parameters == (params[1],)
    assert circuit.queue[5].parameters == (0.7891,)


def test_circuit_set_parameters_errors():
    """Check updating parameters errors."""
    circuit = Circuit(2)
    circuit.add(gates.RX(0, theta=0.789))
    circuit.add(gates.RX(1, theta=0.789))
    circuit.add(gates.fSim(0, 1, theta=0.123, phi=0.456))

    with pytest.raises(KeyError):
        circuit.set_parameters({gates.RX(0, theta=1.0): 0.568})
    with pytest.raises(ValueError):
        circuit.set_parameters([0.12586])
    with pytest.raises(TypeError):
        circuit.set_parameters({0.3568})
    with pytest.raises(ValueError):
        circuit.queue[2].parameters = [0.1234, 0.4321, 0.156]


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
    circuit = Circuit(2, wire_names=["q0", "q1"])
    circuit.add(gates.CallbackGate(entropy))
    circuit.add(gates.H(0))
    circuit.add(gates.CallbackGate(entropy))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CallbackGate(entropy))

    ref = "q0: ─EE─H─EE─o─EE─\n" + "q1: ─EE───EE─X─EE─"

    if legend:
        ref += (
            "\n\n Legend for callbacks and channels: \n"
            + "| Gate                | Symbol   |\n"
            + "|---------------------+----------|\n"
            + "| EntanglementEntropy | EE       |"
        )

    circuit.draw(legend=legend)
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
