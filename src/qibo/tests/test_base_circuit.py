import pytest
from qibo.base import gates
from qibo.config import raise_error
from qibo.base.circuit import BaseCircuit


class Circuit(BaseCircuit): # pragma: no-cover
    """``BaseCircuit`` implementation without abstract methods for testing."""

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
    assert c.size == 2


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


def test_circuit_addition():
    c1 = Circuit(2)
    g1, g2 = gates.H(0), gates.H(1)
    c1.add(g1)
    c1.add(g2)

    c2 = Circuit(2)
    g3 = gates.CNOT(0, 1)
    c2.add(g3)

    c3 = c1 + c2
    assert c3.depth == 2
    assert list(c3.queue) == [g1, g2, g3]


def test_circuit_addition_errors():
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.H(1))

    c2 = Circuit(1)
    c2.add(gates.X(0))

    with pytest.raises(ValueError):
        c3 = c1 + c2
