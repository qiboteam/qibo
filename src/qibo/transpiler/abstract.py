from abc import ABC, abstractmethod
from typing import Tuple

import networkx as nx

from qibo.models import Circuit


class Placer(ABC):
    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """A placer implements the initial logical-physical qubit mapping"""

    @abstractmethod
    def __call__(self, circuit: Circuit, *args):
        """Find initial qubit mapping. Mapping is saved in the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be mapped.
        """


class Router(ABC):
    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """A router implements the mapping of a circuit on a specific hardware."""

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> Tuple[Circuit, dict]:
        """Match circuit to hardware connectivity.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be routed.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): routed circuit.
        """


class Optimizer(ABC):
    """An optimizer tries to reduce the number of gates during transpilation."""

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> Circuit:
        """Optimize transpiled circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be optimized

        Returns:
            (:class:`qibo.models.circuit.Circuit`): circuit with optimized number of gates.
        """
