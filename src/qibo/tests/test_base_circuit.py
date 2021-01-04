import pytest
from qibo.base import gates
from qibo.config import raise_error
from qibo.base.circuit import BaseCircuit


class Circuit(BaseCircuit): # pragma: no-cover
    """``BaseCircuit`` implementation without abstract methods for testing."""

    def fuse(self):
        raise_error(NotImplementedError)

    def _get_parameters_flatlist(self):
        raise_error(NotImplementedError)

    def execute(self):
        raise_error(NotImplementedError)

    @property
    def final_state(self):
        raise_error(NotImplementedError)


def test_parametrizedgates_class():
    from qibo.base.circuit import _ParametrizedGates
    paramgates = _ParametrizedGates()
    paramgates.append(gates.RX(0, 0.1234))
    paramgates.append(gates.fSim(0, 1, 0.1234, 0.4321))
    assert len(paramgates.set) == 2
    assert paramgates.nparams == 3


def test_queue_class():
    from qibo.base.circuit import _Queue
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

# TODO: Test `_add_measurement`
# TODO: Test `_add_layer`

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
    new_gates = list(c.on_qubits(2, 5, 4))
    assert new_gates[0].target_qubits == (2,)
    assert new_gates[1].target_qubits == (5,)
    assert new_gates[2].target_qubits == (4,)
    assert new_gates[3].target_qubits == (5,)
    assert new_gates[3].control_qubits == (2,)
    assert new_gates[4].target_qubits == (4,)
    assert new_gates[4].control_qubits == (5,)


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
