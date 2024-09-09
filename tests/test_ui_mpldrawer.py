"""Test that Qibo matplotlib drawer"""

import matplotlib.pyplot
import pytest

from qibo import Circuit, gates
from qibo.ui import plot_circuit
from qibo.ui.drawer_utils import FusedEndGateBarrier, FusedStartGateBarrier

# defining a dummy circuit
def circuit(nqubits=2):
    c = Circuit(nqubits)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.M(0))
    c.add(gates.M(1))
    return c


def test_plot_circuit():
    circ = circuit()
    ax, _ = plot_circuit(circ)
    assert ax.title == ax.title

def test_fused_gates():
    min_q = 0
    max_q = 1
    l_gates = 1
    equal_qbits = True
    start_barrier = FusedStartGateBarrier(min_q, max_q, l_gates, equal_qbits)
    end_barrier = FusedEndGateBarrier(min_q, max_q)
    assert start_barrier.unitary == start_barrier.unitary
    assert end_barrier.unitary == end_barrier.unitary
