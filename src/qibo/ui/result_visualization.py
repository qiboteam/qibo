"""Plotscripts to visualize circuit execution's result."""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from qibo.config import raise_error
from qibo.result import CircuitResult, MeasurementOutcomes, QuantumState
from qibo.ui.drawing_utils import (
    QIBO_COMPLEMENTARY_COLOR,
    QIBO_DEFAULT_COLOR,
    generate_bitstring_combinations,
)


def visualize_state(
    execution_outcome: Union[QuantumState, MeasurementOutcomes, CircuitResult],
    mode: str = "probabilities",
    n_most_relevant_components=None,
):
    """
    Plot circuit execution's result data according to the chosen ``mode``.

    Args:
        execution_outcome: qibo circuit's result. Depending on the simulation
            preferences, some of the visualizations can be accessed and some of them
            not. In particular:
                - if ``execution_outcome`` is a `QuantumState`, only probabilities and
                  amplitudes can be visualized;
                - if ``execution_outcome`` is a ``MeasurementOutcomes``, then all
                  the ``mode`` options are available.
        mode: visualization mode can be "amplitudes", "probabilities" or "frequencies".
            Default is "probabilities".
        n_most_relevant_components (int): in case the system is big (more than a few
            qubits), it can be helpful to reduce the number of ticks in the x-axis.
            To do so, this argument can be set, reducing the number of plotted ticks
            to `n_most_relevant_components`. Default is None.
    """

    # Collect amplitude
    amplitudes = execution_outcome.state()
    nqubits = int(np.log2(len(amplitudes)))

    bitstrings = generate_bitstring_combinations(nqubits)

    num_bitstrings = len(bitstrings)

    fig_width = max(6, nqubits * 1.5)
    fig_height = 5

    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(num_bitstrings)

    ax.set_xlabel("States")
    ax.set_ylabel(mode.capitalize())

    if mode == "frequencies":
        if not isinstance(execution_outcome, MeasurementOutcomes):
            raise_error(
                ValueError,
                "To visualize frequencies, ensure the circuit is executed with shots.",
            )
        frequencies = execution_outcome.frequencies()
        y_values = [frequencies.get(b, 0) for b in bitstrings]
        ax.bar(x, y_values, color=QIBO_DEFAULT_COLOR, edgecolor="black")

    elif mode == "amplitudes":
        real_parts = np.real(amplitudes)
        imag_parts = np.imag(amplitudes)
        width = 0.3
        ax.bar(
            x - width / 2,
            real_parts,
            width,
            color=QIBO_DEFAULT_COLOR,
            edgecolor="black",
            label="Real part",
        )
        ax.bar(
            x + width / 2,
            imag_parts,
            width,
            color=QIBO_COMPLEMENTARY_COLOR,
            edgecolor="black",
            label="Imag part",
        )
        ax.hlines(0, -1, num_bitstrings, color="black", lw=1)
        ax.legend()
        y_values = np.abs(amplitudes)

    elif mode == "probabilities":
        probabilities = np.abs(amplitudes) ** 2
        ax.bar(x, probabilities, color=QIBO_DEFAULT_COLOR, edgecolor="black")
        y_values = probabilities

    else:
        raise_error(
            ValueError,
            f"Unsupported mode '{mode}'. Choose 'amplitudes', 'probabilities' or 'frequencies'.",
        )

    # Adjust x-axis labels based on number of qubits
    if n_most_relevant_components is not None:
        top_indices = np.argsort(y_values)[-n_most_relevant_components:]
        tick_labels = [
            bitstrings[i] if i in top_indices else "" for i in range(num_bitstrings)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=90)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(bitstrings, rotation=90)

    plt.tight_layout()

    return ax, ax.figure
