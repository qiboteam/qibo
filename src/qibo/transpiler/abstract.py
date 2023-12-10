from abc import ABC, abstractmethod
from typing import Tuple

import networkx as nx

from qibo.models import Circuit


class Placer(ABC):
    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """A placer implements the initial logical-physical qubit mapping"""

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> dict:
        """Find initial qubit mapping

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be mapped.

        Returns:
            (dict): dictionary containing the initial logical to physical qubit mapping.
        """


class Router(ABC):
    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """A router implements the mapping of a circuit on a specific hardware."""

    @abstractmethod
    def __call__(
        self, circuit: Circuit, initial_layout: dict, *args
    ) -> Tuple[Circuit, dict]:
        """Match circuit to hardware connectivity.

        Args:
            circuit (qibo.models.Circuit): circuit to be routed.
            initial_layout (dict): dictionary containing the initial logical to physical qubit mapping.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): routed circuit and dictionary containing the final logical to physical qubit mapping.
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
