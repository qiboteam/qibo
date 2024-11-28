from abc import ABC, abstractmethod
from typing import Tuple

import networkx as nx

from qibo.models import Circuit


class Placer(ABC):
    """Maps logical qubits to physical qubits."""

    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """Initializes the placer.

        Args:
            connectivity (nx.Graph): hardware topology.
        """

    @abstractmethod
    def __call__(self, circuit: Circuit, *args):
        """Find initial qubit mapping. Mapping is saved in the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be mapped.
        """


class Router(ABC):
    """Makes the circuit executable on the given topology."""

    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """Initializes the router.

        Args:
            connectivity (nx.Graph): hardware topology.
        """

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> Tuple[Circuit, dict]:
        """Match circuit to hardware connectivity.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be routed.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): routed circuit and dictionary containing the final wire_names mapping.
        """


class Optimizer(ABC):
    """Reduces the number of gates in the circuit."""

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> Circuit:
        """Optimize transpiled circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be optimized.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): circuit with optimized number of gates.
        """
