"""User-interface module."""

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    raise ModuleNotFoundError(
        "``qibo.ui`` requires ``matplotlib``. Install it with ``pip install qibo[ui]``."
    ) from None

from qibo.ui.mpldrawer import plot_circuit
from qibo.ui.result_visualization import plot_density_hist, visualize_state

__all__ = [
    "plot_circuit",
    "plot_density_hist",
    "visualize_state",
]
