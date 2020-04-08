"""
Test imports and basic functionality that is indepedent of calculation backend.
"""
import numpy as np
from qibo.models import *
from qibo.gates import *


def test_circuit_sanity():
    """Check if the number of qbits is preserved."""
    c = Circuit(2)
    assert c.nqubits == 2
    assert c.size == 2


def test_importing_qibo():
    """Checks accessing `models` and `gates` from `qibo`."""
    import qibo
    c = qibo.models.Circuit(2)
    c.add(qibo.gates.H(0))
    assert c.nqubits == 2
    assert c.depth == 1


def test_importing_qibo():
    """Checks importing `models` and `gates`."""
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


def test_measured_qubits_property():
    """Check that `measured_qubits` property returns the correct qubit ids."""
    c = Circuit(4)
    c.add(M(0, 1))
    c.add(M(3))
    assert c.measured_qubits == {0, 1, 3}


def test_circuit_addition_with_measurements():
    """Check if measurements are transferred during circuit addition."""
    c = Circuit(2)
    c.add(H(0))
    c.add(H(1))

    meas_c = Circuit(2)
    c.add(M(0, 1))

    c += meas_c
    assert len(c.measurement_gate.target_qubits) == 2
    assert c.measurement_sets == {"register0": {0, 1}}


def test_circuit_addition_with_measurements_in_both_circuits():
    """Check if measurements of two circuits are added during circuit addition."""
    c1 = Circuit(2)
    c1.add(H(0))
    c1.add(H(1))
    c1.add(M(1, register_name="a"))

    c2 = Circuit(2)
    c2.add(X(0))
    c2.add(M(0, register_name="b"))

    c = c1 + c2
    assert len(c.measurement_gate.target_qubits) == 2
    assert c.measurement_sets == {"a": {1}, "b": {0}}
