"""
Test imports and basic functionality that is indepedent of calculation backend.
"""
import numpy as np
import pytest
from qibo.models import *
from qibo.gates import *


def test_circuit_sanity():
    """Check if the number of qbits is preserved."""
    c = Circuit(2)
    assert c.nqubits == 2
    assert c.size == 2


def test_importing_full_qibo():
    """Checks accessing `models` and `gates` from `qibo`."""
    import qibo
    c = qibo.models.Circuit(2)
    c.add(qibo.gates.H(0))
    assert c.nqubits == 2
    assert c.depth == 1


def test_importing_qibo_modules():
    """Checks importing `models` and `gates` from qibo."""
    from qibo import models, gates
    c = models.Circuit(2)
    c.add(gates.H(0))
    assert c.nqubits == 2
    assert c.depth == 1


def test_circuit_add():
    """Check if circuit depth increases with the add method."""
    c = Circuit(2)
    c.add(H(0))
    c.add(H(1))
    c.add(CNOT(0, 1))
    assert c.depth == 3


def test_circuit_add_bad_gate():
    """Check ``circuit.add()`` exceptions."""
    c = Circuit(2)
    with pytest.raises(TypeError):
        c.add(0)
    with pytest.raises(ValueError):
        c.add(H(2))
    with pytest.raises(ValueError):
        gate = H(1)
        gate.nqubits = 3
        c.add(gate)

    final_state = c()
    with pytest.raises(RuntimeError):
        c.add(H(0))


def test_circuit_add_iterable():
    """Check if `circuit.add` works with iterables."""
    c = Circuit(2)
    # Try adding list
    c.add([H(0), H(1), CNOT(0, 1)])
    assert c.depth == 3
    assert isinstance(c.queue[-1], CNOT)
    # Try adding tuple
    c.add((H(0), H(1), CNOT(0, 1)))
    assert c.depth == 6
    assert isinstance(c.queue[-1], CNOT)


def test_circuit_add_generator():
    """Check if `circuit.add` works with generators."""
    def gen():
        yield H(0)
        yield H(1)
        yield CNOT(0, 1)
    c = Circuit(2)
    c.add(gen())
    assert c.depth == 3
    assert isinstance(c.queue[-1], CNOT)


def test_circuit_addition():
    """Check if circuit addition increases depth."""
    c1 = Circuit(2)
    c1.add(H(0))
    c1.add(H(1))
    assert c1.depth == 2

    c2 = Circuit(2)
    c2.add(CNOT(0, 1))
    assert c2.depth == 1

    c3 = c1 + c2
    assert c3.depth == 3


def test_bad_circuit_addition():
    """Check that it is not possible to add circuits with different number of qubits."""
    c1 = Circuit(2)
    c1.add(H(0))
    c1.add(H(1))

    c2 = Circuit(1)
    c2.add(X(0))

    with pytest.raises(ValueError):
        c3 = c1 + c2


def test_circuit_copy():
    """Check that ``circuit.copy()`` copies gates properly."""
    c1 = Circuit(2)
    c1.add([H(0), H(1), CNOT(0, 1)])
    c2 = c1.copy()
    assert c2.depth == c1.depth
    assert c2.nqubits == c1.nqubits
    for g1, g2 in zip(c1.queue, c2.queue):
        assert g1 is g2


def test_circuit_deep_copy():
    c1 = Circuit(2)
    c1.add([H(0), H(1), CNOT(0, 1)])
    with pytest.raises(NotImplementedError):
        c2 = c1.copy(deep=True)


def test_circuit_copy_with_measurements():
    """Check that ``circuit.copy()`` copies measurements properly."""
    c1 = Circuit(4)
    c1.add([H(0), H(3), CNOT(0, 2)])
    c1.add(M(0, 1, register_name="a"))
    c1.add(M(3, register_name="b"))
    c2 = c1.copy()

    assert c2.measurement_gate is c1.measurement_gate
    assert c2.measurement_tuples == {"a": (0, 1), "b": (3,)}
