"""
Test imports and basic functionality that is indepedent of calculation backend.
"""
import numpy as np
import pytest
import qibo
from qibo.models import *
from qibo.gates import *


def test_importing_full_qibo():
    """Checks accessing `models` and `gates` from `qibo`."""
    import qibo
    c = qibo.models.Circuit(2)
    c.add(qibo.gates.H(0))
    assert c.nqubits == 2
    assert c.depth == 1
    assert c.ngates == 1


def test_importing_qibo_modules():
    """Checks importing `models` and `gates` from qibo."""
    from qibo import models, gates
    c = models.Circuit(2)
    c.add(gates.H(0))
    assert c.nqubits == 2
    assert c.depth == 1
    assert c.ngates == 1


def test_base_gate_errors():
    """Check errors in ``base.gates.Gate`` for coverage."""
    gate = H(0)
    with pytest.raises(ValueError):
        nqubits = gate.nqubits
    with pytest.raises(ValueError):
        nstates = gate.nstates
    # Access nstates
    gate2 = H(0)
    gate2.nqubits = 3
    _ = gate2.nstates

    with pytest.raises(ValueError):
        gate.nqubits = 2
        gate.nqubits = 3
    with pytest.raises(RuntimeError):
        cgate = gate.controlled_by(1)
    with pytest.raises(RuntimeError):
        gate = H(0).controlled_by(1).controlled_by(2)
