from abc import ABC, abstractmethod

import networkx as nx

from qibo import Circuit
from qibo.backends import Backend


class Placer(ABC):
    """Maps logical qubits to physical qubits."""

    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """Initializes the placer.

        Args:
            connectivity (nx.Graph): Hardware topology.
        """

    @abstractmethod
    def __call__(self, circuit: Circuit, backend: Backend | None = None) -> None:
        """Find initial qubit mapping.

        Method works in-place.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be placed.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend to use.
                Placers that do not need a backend accept it to be compatible with
                :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        """


class Router(ABC):
    """Makes the circuit executable on the given topology."""

    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """Initializes the router.

        Args:
            connectivity (nx.Graph): Hardware topology.
        """

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> tuple[Circuit, dict]:
        """Match circuit to hardware connectivity.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be routed.

        Returns:
            tuple(:class:`qibo.models.circuit.Circuit`, dict): Routed circuit and final
            {logical: physical} qubit mapping.
        """


class Optimizer(ABC):
    """Reduces the number of gates in the circuit."""

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> Circuit:
        """Optimize transpiled circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be optimized.

        Returns:
            :class:`qibo.models.circuit.Circuit`: Optimized circuit.
        """
