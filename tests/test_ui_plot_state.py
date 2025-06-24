from pathlib import Path

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images

import qibo
from qibo import Circuit, gates
from qibo.ui.plot_state import plot_density_hist

from .utils import fig2png

qibo.set_backend("numpy")

matplotlib.use("agg")

BASEPATH = str(Path(__file__).parent / "test_plot_state_ui_images")

IMAGE_TOLERANCE = 255


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

    temp_file_path = fig2png(fig)

    assert (
        compare_images(
            BASEPATH + "/test_complex_circuit_state_nqubits_" + str(nqubits) + ".png",
            temp_file_path,
            tol=IMAGE_TOLERANCE,
        )
        == None
    )


def test_simple_circuit_state():
    """Test for simple circuit plot state"""
    nqubits = 2
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    fig, _, _ = plot_density_hist(circuit)

    temp_file_path = fig2png(fig)

    assert (
        compare_images(
            BASEPATH + "/test_simple_circuit_state_nqubits_" + str(nqubits) + ".png",
            temp_file_path,
            tol=IMAGE_TOLERANCE,
        )
        == None
    )


def test_simple_circuit_state_hadamard():
    """Test for simple circuit plot state with Hadamard gates"""
    nqubits = 2
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.H(1))
    fig, _, _ = plot_density_hist(circuit)

    temp_file_path = fig2png(fig)

    assert (
        compare_images(
            BASEPATH
            + "/test_simple_circuit_state_hadamard_nqubits_"
            + str(nqubits)
            + ".png",
            temp_file_path,
            tol=IMAGE_TOLERANCE,
        )
        == None
    )


def test_simple_title_circuit_colors_state():
    """Test for simple circuit plot state with title, alpha and custom colors"""
    nqubits = 2
    circuit = Circuit(nqubits)

    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    fig, _, _ = plot_density_hist(
        circuit, title="Density plot", alpha=0.5, colors=["green", "purple"]
    )

    temp_file_path = fig2png(fig)

    assert (
        compare_images(
            BASEPATH
            + "/test_simple_circuit_state_colors_nqubits_"
            + str(nqubits)
            + ".png",
            temp_file_path,
            tol=IMAGE_TOLERANCE,
        )
        == None
    )


def test_title_circuit_state():
    """Test for simple circuit plot state with title"""
    nqubits = 3
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(0, 2))
    fig, _, _ = plot_density_hist(circuit, title="Test Circuit State")

    temp_file_path = fig2png(fig)

    assert (
        compare_images(
            BASEPATH + "/test_title_circuit_state_nqubits_" + str(nqubits) + ".png",
            temp_file_path,
            tol=IMAGE_TOLERANCE,
        )
        == None
    )


def test_simple_raise_error_measure_state():
    """Test for simple circuit plot state with error raising if measurement gates are present"""
    nqubits = 3
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(*range(nqubits)))

    with pytest.raises(Exception) as excinfo:
        plot_density_hist(circuit, title="Test Circuit State")
        assert (
            str(excinfo.value)
            == "Circuit must not contain measurement gates for density matrix visualization"
        )


def test_simple_raise_error_colors_state():
    """Test for simple circuit plot state with error raising if colors are not a list of length 2"""
    nqubits = 2
    circuit = Circuit(nqubits)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    with pytest.raises(Exception) as excinfo:
        colors = ["red", "blue", "green"]
        plot_density_hist(circuit, title="Test Circuit State", colors=colors)
        assert str(
            excinfo.value
        ) == "Colors must be a list of len=2, got {} instead".format(len(colors))
