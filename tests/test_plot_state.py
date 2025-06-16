from pathlib import Path

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest

import qibo
from qibo import Circuit, construct_backend, gates
from qibo.ui.plot_state import plot_density_hist

from .utils import match_figure_image

qibo.set_backend("numpy")

matplotlib.use("agg")

BASEPATH = str(Path(__file__).parent / "test_plot_state_ui_images")


def test_complex_circuit_state():
    """Test for simple circuit plot state"""
    nqubits = 2
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RY(0, theta=np.pi / 3))
    circuit.add(gates.RX(1, theta=np.pi / 5))
    fig, _, _ = plot_density_hist(circuit)
    assert (
        match_figure_image(
            fig,
            BASEPATH + "/test_complex_circuit_state_nqubits_" + str(nqubits) + ".npy",
        )
        == True
    )


def test_simple_circuit_state():
    """Test for simple circuit plot state"""
    nqubits = 2
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    fig, _, _ = plot_density_hist(circuit)
    assert (
        match_figure_image(
            fig,
            BASEPATH + "/test_simple_circuit_state_nqubits_" + str(nqubits) + ".npy",
        )
        == True
    )
