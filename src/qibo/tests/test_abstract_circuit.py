"""Test all methods defined in `qibo/abstractions/circuit.py`."""
import pytest
from qibo.abstractions import gates
from qibo.config import raise_error
from qibo.abstractions.circuit import AbstractCircuit


class Circuit(AbstractCircuit): # pragma: no cover
    """``BaseCircuit`` implementation without abstract methods for testing."""

    def _add_layer(self, gate):
        raise_error(NotImplementedError)

    def fuse(self):
        raise_error(NotImplementedError)

    def execute(self):
        raise_error(NotImplementedError)

    @property
    def final_state(self):
        raise_error(NotImplementedError)


def test_parametrizedgates_class():
    from qibo.abstractions.circuit import _ParametrizedGates
    paramgates = _ParametrizedGates()
    paramgates.append(gates.RX(0, 0.1234))
    paramgates.append(gates.fSim(0, 1, 0.1234, 0.4321))
    assert len(paramgates.set) == 2
    assert paramgates.nparams == 3


def test_queue_class():
    from qibo.abstractions.circuit import _Queue
    queue = _Queue(4)
    gatelist = [gates.H(0), gates.H(1), gates.X(0),
                gates.H(2), gates.CNOT(1, 2), gates.Y(3)]
    for g in gatelist:
        queue.append(g)
    assert queue.moments == [[gatelist[0], gatelist[1], gatelist[3], gatelist[5]],
                             [gatelist[2], gatelist[4], gatelist[4], None]]


def test_circuit_init():
    c = Circuit(2)
    assert c.nqubits == 2


@pytest.mark.parametrize("nqubits", [0, -10, 2.5])
def test_circuit_init_errors(nqubits):
    with pytest.raises((ValueError, TypeError)):
        c = Circuit(nqubits)


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
    c.add((gen() for _ in range(3)))
    assert c.depth == 6
    assert c.ngates == 9
    assert isinstance(c.queue[2], gates.CNOT)
    assert isinstance(c.queue[5], gates.CNOT)
    assert isinstance(c.queue[7], gates.H)


def test_set_nqubits():
    c = Circuit(2)
    gate = gates.H(0)
    gate.nqubits = 3
    gate.is_prepared = True
    with pytest.raises(RuntimeError):
        c.add(gate)


def test_add_measurement():
    c = Circuit(5)
    g1 = gates.M(0, 2, register_name="a")
    g2 = gates.M(3, register_name="b")
    c.add([g1, g2])
    assert c.measurement_gate is g1
    assert c.measurement_tuples == {"a": (0, 2), "b": (3,)}
    assert g1.target_qubits == (0, 2, 3)
    assert g2.target_qubits == (3,)
    with pytest.raises(KeyError):
        c.add(gates.M(4, register_name="b"))


def test_gate_types():
    import collections
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.X(2))
    c.add(gates.CNOT(0, 2))
    c.add(gates.CNOT(1, 2))
    c.add(gates.TOFFOLI(0, 1, 2))
    target_counter = collections.Counter({"h": 2, "x": 1, "cx": 2, "ccx": 1})
    assert c.ngates == 6
    assert c.gate_types == target_counter


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
    target_summary = "\n".join(["Circuit depth = 5",
                                "Total number of gates = 6",
                                "Number of qubits = 3",
                                "Most common gates:",
                                "h: 3", "cx: 2", "ccx: 1"])
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
    assert c3.depth == 3
    assert list(c3.queue) == [g1, g2, g3]
    if measurements:
        assert c3.measurement_tuples == {"a": (0,), "b": (1,)}
        assert c3.measurement_gate.target_qubits == (0, 1)


def test_circuit_addition_errors():
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.H(1))

    c2 = Circuit(1)
    c2.add(gates.X(0))

    with pytest.raises(ValueError):
        c3 = c1 + c2


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
    smallc.add((gates.H(i) for i in range(2)))
    with pytest.raises(ValueError):
        next(smallc.on_qubits(0, 1, 2))

    smallc = Circuit(2)
    smallc.add(gates.Flatten(4 * [0.5]))
    with pytest.raises(NotImplementedError):
        next(smallc.on_qubits(0, 1))

    from qibo.abstractions.callbacks import Callback
    smallc = Circuit(4)
    smallc.add(gates.CallbackGate(Callback()))
    with pytest.raises(NotImplementedError):
        next(smallc.on_qubits(0, 1, 2, 3))


@pytest.mark.parametrize("deep", [False, True])
def test_circuit_copy(deep):
    c1 = Circuit(2)
    c1.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    c2 = c1.copy(deep)
    assert c2.depth == c1.depth
    assert c2.ngates == c1.ngates
    assert c2.nqubits == c1.nqubits
    for g1, g2 in zip(c1.queue, c2.queue):
        if deep:
            assert g1.__class__ == g2.__class__
            assert g1.target_qubits == g2.target_qubits
            assert g1.control_qubits == g2.control_qubits
        else:
            assert g1 is g2


def test_circuit_copy_with_measurements():
    c1 = Circuit(4)
    c1.add([gates.H(0), gates.H(3), gates.CNOT(0, 2)])
    c1.add(gates.M(0, 1, register_name="a"))
    c1.add(gates.M(3, register_name="b"))
    c2 = c1.copy()
    assert c2.measurement_gate is c1.measurement_gate
    assert c2.measurement_tuples == {"a": (0, 1), "b": (3,)}


@pytest.mark.parametrize("measurements", [False, True])
def test_circuit_invert(measurements):
    c = Circuit(3)
    gatelist = [gates.H(0), gates.X(1), gates.Y(2),
                gates.CNOT(0, 1), gates.CZ(1, 2)]
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
        assert invc.measurement_gate.target_qubits == (0, 2)
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
        assert decompc.measurement_gate.target_qubits == (0, 2)
        assert decompc.measurement_tuples == {"register0": (0, 2)}


@pytest.mark.parametrize("measurements", [False, True])
@pytest.mark.parametrize("noise_map",
                         [(0.1, 0.2, 0.3),
                          {0: (0.1, 0.0, 0.2), 1: (0.0, 0.2, 0.1)}])
def test_circuit_with_noise(measurements, noise_map):
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1), gates.CNOT(0, 1)])
    if measurements:
        c.add(gates.M(0, 1))
    noisyc = c.with_noise(noise_map)

    if not isinstance(noise_map, dict):
        noise_map = {0: noise_map, 1: noise_map}
    targetc = Circuit(2)
    targetc.add(gates.H(0))
    targetc.add(gates.PauliNoiseChannel(0, *noise_map[0]))
    targetc.add(gates.H(1))
    targetc.add(gates.PauliNoiseChannel(1, *noise_map[1]))
    targetc.add(gates.CNOT(0, 1))
    targetc.add(gates.PauliNoiseChannel(0, *noise_map[0]))
    targetc.add(gates.PauliNoiseChannel(1, *noise_map[1]))
    for g1, g2 in zip(noisyc.queue, targetc.queue):
        assert isinstance(g1, g2.__class__)
        assert g1.target_qubits == g2.target_qubits
        assert g1.control_qubits == g2.control_qubits
    if measurements:
        assert noisyc.measurement_gate.target_qubits == (0, 1)
        assert noisyc.measurement_tuples == {"register0": (0, 1)}


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("include_not_trainable", [True, False])
@pytest.mark.parametrize("format", ["list", "dict", "flatlist"])
def test_get_parameters(trainable, include_not_trainable, format):
    import numpy as np
    matrix = np.random.random((2, 2))
    c = Circuit(3)
    c.add(gates.RX(0, theta=0.123))
    c.add(gates.RY(1, theta=0.456, trainable=trainable))
    c.add(gates.CZ(1, 2))
    c.add(gates.Unitary(matrix, 2))
    c.add(gates.fSim(0, 2, theta=0.789, phi=0.987, trainable=trainable))
    c.add(gates.H(2))
    c.param_tensor_types = (np.ndarray,)
    params = c.get_parameters(format, include_not_trainable)
    if trainable or include_not_trainable:
        target_params = {
            "list": [0.123, 0.456, (0.789, 0.987)],
            "dict": {c.queue[0]: 0.123, c.queue[1]: 0.456,
                     c.queue[4]: (0.789, 0.987)},
            "flatlist": [0.123, 0.456]
            }
        target_params["flatlist"].extend(list(matrix.ravel()))
        target_params["flatlist"].extend([0.789, 0.987])
    else:
        target_params = {
            "list": [0.123],
            "dict": {c.queue[0]: 0.123},
            "flatlist": [0.123]
            }
        target_params["flatlist"].extend(list(matrix.ravel()))
    if format == "list":
        i = len(target_params["list"]) // 2 + 1
        np.testing.assert_allclose(params.pop(i), matrix)
    elif format == "dict":
        np.testing.assert_allclose(params.pop(c.queue[3]), matrix)
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
        assert c.queue[0].parameters == params[0]
    else:
        c.set_parameters(params[1:])
    assert c.queue[1].parameters == params[1]
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
        assert c.queue[2].parameters == params[0]
    else:
        params_dict = {c.queue[3]: params[1], c.queue[5]: params[2]}
        c.set_parameters(params_dict)
    assert c.queue[3].parameters == params[1]
    assert c.queue[5].parameters == params[2]

    # test not passing all parametrized gates
    c.set_parameters({c.queue[5]: 0.7891})
    if trainable:
        assert c.queue[2].parameters == params[0]
    assert c.queue[3].parameters == params[1]
    assert c.queue[5].parameters == 0.7891


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
    c.fusion_groups = ["test"]
    with pytest.raises(TypeError):
        c.set_parameters({gates.RX(0, theta=1.0): 0.568})


def test_circuit_draw():
    """Test circuit text draw."""
    ref = 'q0: ─H─U1─U1─U1─U1───────────────────────────x───\n' \
          'q1: ───o──|──|──|──H─U1─U1─U1────────────────|─x─\n' \
          'q2: ──────o──|──|────o──|──|──H─U1─U1────────|─|─\n' \
          'q3: ─────────o──|───────o──|────o──|──H─U1───|─x─\n' \
          'q4: ────────────o──────────o───────o────o──H─x───'
    circuit = Circuit(5)
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    assert circuit.draw() == ref


def test_circuit_draw_line_wrap():
    """Test circuit text draw with line wrap."""
    ref_line_wrap_50 = \
        'q0:     ─H─U1─U1─U1─U1───────────────────────────x───I───f ...\n' \
        'q1:     ───o──|──|──|──H─U1─U1─U1────────────────|─x─I───| ...\n' \
        'q2:     ──────o──|──|────o──|──|──H─U1─U1────────|─|─────| ...\n' \
        'q3:     ─────────o──|───────o──|────o──|──H─U1───|─x───M─| ...\n' \
        'q4:     ────────────o──────────o───────o────o──H─x───────f ...\n' \
        '\n' \
        'q0: ... ─o────gf───M─\n' \
        'q1: ... ─U3───|──o─M─\n' \
        'q2: ... ────X─gf─o─M─\n' \
        'q3: ... ────o────o───\n' \
        'q4: ... ────o────X───'

    ref_line_wrap_30 = \
        'q0:     ─H─U1─U1─U1─U1──────────────── ...\n' \
        'q1:     ───o──|──|──|──H─U1─U1─U1───── ...\n' \
        'q2:     ──────o──|──|────o──|──|──H─U1 ...\n' \
        'q3:     ─────────o──|───────o──|────o─ ...\n' \
        'q4:     ────────────o──────────o────── ...\n' \
        '\n' \
        'q0: ... ───────────x───I───f─o────gf── ...\n' \
        'q1: ... ───────────|─x─I───|─U3───|──o ...\n' \
        'q2: ... ─U1────────|─|─────|────X─gf─o ...\n' \
        'q3: ... ─|──H─U1───|─x───M─|────o────o ...\n' \
        'q4: ... ─o────o──H─x───────f────o────X ...\n' \
        '\n' \
        'q0: ... ─M─\n' \
        'q1: ... ─M─\n' \
        'q2: ... ─M─\n' \
        'q3: ... ───\n' \
        'q4: ... ───'

    import numpy as np
    circuit = Circuit(5)
    for i1 in range(5):
        circuit.add(gates.H(i1))
        for i2 in range(i1 + 1, 5):
            circuit.add(gates.CU1(i2, i1, theta=0))
    circuit.add(gates.SWAP(0, 4))
    circuit.add(gates.SWAP(1, 3))
    circuit.add(gates.I(*range(2)))
    circuit.add(gates.M(3, collapse=True))
    circuit.add(gates.fSim(0,4,0,0))
    circuit.add(gates.CU3(0,1,0,0,0))
    circuit.add(gates.TOFFOLI(4,3,2))
    circuit.add(gates.GeneralizedfSim(0, 2, np.eye(2), 0))
    circuit.add(gates.X(4).controlled_by(1,2,3))
    circuit.add(gates.M(*range(3)))
    assert circuit.draw(line_wrap=50) == ref_line_wrap_50
    assert circuit.draw(line_wrap=30) == ref_line_wrap_30


def test_circuit_draw_not_supported_gates():
    """Check that ``NotImplementedError`` is raised if gate is not supported."""
    c = Circuit(2)
    c.add(gates.Flatten(1))
    with pytest.raises(NotImplementedError):
        c.draw()
