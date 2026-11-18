from abc import ABC, abstractmethod

import networkx as nx

from qibo import Circuit
from qibo.backends import Backend
from qibo.transpiler.layout import TranspileLayout


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

    @property
    def connectivity(self) -> nx.Graph | None:
        """Hardware topology. A directed graph is stored as an undirected one, because the
        direction of the connections does not matter to place the qubits."""
        return self._connectivity

    @connectivity.setter
    def connectivity(self, connectivity: nx.Graph | None) -> None:
        if connectivity is not None and connectivity.is_directed():
            connectivity = connectivity.to_undirected()
        self._connectivity = connectivity


class PostLayout(ABC):
    """Chooses other physical qubits for a circuit that is already routed.

    It only relabels the physical qubits, so the gates of the circuit and their order do
    not change, and neither do the SWAP gates of the router. The new qubits must be
    connected as the old ones, *i.e.* every two-qubit gate must still act on connected
    qubits."""

    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        """Initializes the post layout.

        Args:
            connectivity (nx.Graph): Hardware topology.
        """

    @abstractmethod
    def __call__(self, circuit: Circuit, layout: TranspileLayout) -> TranspileLayout:
        """Find a better layout for a routed circuit.

        The method changes the ``wire_names`` of the circuit in place.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Routed circuit.
            layout (:class:`qibo.transpiler.layout.TranspileLayout`): Layout of the
                circuit, which the router returns.

        Returns:
            :class:`qibo.transpiler.layout.TranspileLayout`: Layout with the labels of the
            new physical qubits. If the layout of the circuit is not changed, it is
            ``layout``.
        """

    @property
    def connectivity(self) -> nx.Graph | None:
        """Hardware topology. A directed graph is stored as an undirected one, because the
        direction of the connections does not matter to choose the qubits."""
        return self._connectivity

    @connectivity.setter
    def connectivity(self, connectivity: nx.Graph | None) -> None:
        if connectivity is not None and connectivity.is_directed():
            connectivity = connectivity.to_undirected()
        self._connectivity = connectivity


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

    @property
    def connectivity(self) -> nx.Graph | None:
        """Hardware topology. A directed graph is stored as an undirected one, because the
        direction of the connections does not matter to route the qubits."""
        return self._connectivity

    @connectivity.setter
    def connectivity(self, connectivity: nx.Graph | None) -> None:
        if connectivity is not None and connectivity.is_directed():
            connectivity = connectivity.to_undirected()
        self._connectivity = connectivity


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
