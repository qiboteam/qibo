"""Tests for Qibo matplotlib drawer"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest

from qibo import Circuit, construct_backend, gates
from qibo.ui.result_visualization import visualize_state

from .utils import match_figure_image

matplotlib.use("agg")

BASEPATH = str(Path(__file__).parent / "test_ui_array_images")


def build_circuit(nqubits):
    """Helper function building a circuit."""
    circuit = Circuit(nqubits)
    for q in range(nqubits):
        circuit.add(gates.RY(q, theta=np.random.randn()))
        circuit.add(gates.RZ(q, theta=np.random.randn()))
    circuit.add(gates.M(*range(nqubits)))
    return circuit


@pytest.mark.parametrize("mode", ["frequencies", "amplitudes", "probabilities"])
@pytest.mark.parametrize("nqubits", [3])
def test_visualize_state(mode, nqubits):
    # Fixing seed for reproducibility
    backend = construct_backend("numpy")
    backend.set_seed(42)
    np.random.seed(42)

    circuit = build_circuit(nqubits)
    outcome = backend.execute_circuit(circuit, nshots=1000)

    _, fig = visualize_state(
        outcome,
        mode=mode,
        n_most_relevant_components=10,
    )

    assert (
        match_figure_image(fig, f"{BASEPATH}/state_visualization_{mode}_{nqubits}q.npy")
        == True
    )
