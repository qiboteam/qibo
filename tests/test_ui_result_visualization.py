"""Tests for Qibo matplotlib drawer"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.testing.compare import compare_images

from qibo import Circuit, construct_backend, gates
from qibo.ui.result_visualization import visualize_state

from .utils import fig2png

matplotlib.use("agg")

BASEPATH = str(Path(__file__).parent / "test_ui_array_images")

IMAGE_TOLERANCE = 0


def build_circuit(nqubits, measurements=True):
    """Helper function building a circuit."""
    circuit = Circuit(nqubits)
    for q in range(nqubits):
        circuit.add(gates.RY(q, theta=np.random.randn()))
        circuit.add(gates.RZ(q, theta=np.random.randn()))
    if measurements:
        circuit.add(gates.M(*range(nqubits)))
    return circuit


@pytest.mark.parametrize("mode", ["frequencies", "amplitudes", "probabilities"])
def test_visualize_state(mode):
    """Testing the plotting function."""

    backend = construct_backend("numpy")
    backend.set_seed(42)
    np.random.seed(42)

    circuit = build_circuit(nqubits=3)
    outcome = backend.execute_circuit(circuit, nshots=1000)

    _, fig = visualize_state(
        execution_outcome=outcome,
        mode=mode,
    )

    temp_file_path = fig2png(fig)
    base_image_path = f"{BASEPATH}/state_visualization_{mode}_3q.png"
    assert (
        compare_images(
            base_image_path,
            temp_file_path,
            tol=IMAGE_TOLERANCE,
        )
        == None
    )


@pytest.mark.parametrize("mode", ["frequencies", "non_valid"])
def test_error_raises(mode):
    """Testing the non valid options."""

    circuit = build_circuit(nqubits=3, measurements=False)
    outcome = circuit()

    with pytest.raises(ValueError):
        _, _ = visualize_state(
            execution_outcome=outcome,
            mode=mode,
        )


def test_n_most_relevant_components():
    """
    Testing the plotting function when the number of relevant components is
    less than 2**nqubits.
    """

    backend = construct_backend("numpy")
    backend.set_seed(42)
    np.random.seed(42)

    circuit = build_circuit(nqubits=3)
    outcome = backend.execute_circuit(circuit, nshots=1000)

    _, fig = visualize_state(
        execution_outcome=outcome,
        mode="probabilities",
        n_most_relevant_components=4,
    )
    temp_file_path = fig2png(fig)
    base_image_path = f"{BASEPATH}/state_visualization_probabilities_3q_lim.png"
    assert (
        compare_images(
            base_image_path,
            temp_file_path,
            tol=IMAGE_TOLERANCE,
        )
        == None
    )
