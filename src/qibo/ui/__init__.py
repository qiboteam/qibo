"""User-interface module."""

from importlib.util import find_spec

from qibo.config import raise_error

if find_spec("matplotlib") is None:  # pragma: no cover
    raise_error(
        ModuleNotFoundError,
        "``qibo.ui`` requires ``matplotlib``. Install it with ``pip install qibo[ui]``.",
    )

from qibo.ui.mpldrawer import plot_circuit
from qibo.ui.result_visualization import plot_density_hist, visualize_state

__all__ = [
    "plot_circuit",
    "plot_density_hist",
    "visualize_state",
]
