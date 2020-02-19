from qibo import models
from qibo import gates


def test_circuit_sanity():
    c = models.Circuit(2)
    assert c.nqubits == 2
    assert c.size() == 2


def test_circuit_add():
    c = models.Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))
    assert c.depth() == 3