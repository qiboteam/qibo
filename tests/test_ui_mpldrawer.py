"""Test that Qibo matplotlib drawer"""

import matplotlib.pyplot
import pytest

from qibo import Circuit, gates
from qibo.ui import plot_circuit


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


@pytest.mark.parametrize("style", ["default"])
def test_plot_params(style):
    dict_style = _plot_params(style)
    assert dict_style["facecolor"] == dict_style["facecolor"]
