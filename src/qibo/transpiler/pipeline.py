import networkx as nx

from qibo.config import raise_error
from qibo.models import Circuit
from qibo.transpiler._exceptions import (
    ConnectivityError,
    PlacementError,
    TranspilerPipelineError,
)
from qibo.transpiler.abstract import Optimizer, Placer, Router
from qibo.transpiler.asserts import (
    assert_connectivity,
    assert_decomposition,
    assert_placement,
)
from qibo.transpiler.unroller import DecompositionError, NativeGates, Unroller


def restrict_connectivity_qubits(connectivity: nx.Graph, qubits: list[str]):
    """Restrict the connectivity to selected qubits.

    Args:
        connectivity (:class:`networkx.Graph`): chip connectivity.
        qubits (list): list of physical qubits to be used.

    Returns:
        (:class:`networkx.Graph`): restricted connectivity.
    """
    if not set(qubits).issubset(set(connectivity.nodes)):
        raise_error(
            ConnectivityError, "Some qubits are not in the original connectivity."
        )

    new_connectivity = nx.Graph()
    new_connectivity.add_nodes_from(qubits)
    new_edges = [
        edge for edge in connectivity.edges if edge[0] in qubits and edge[1] in qubits
    ]
    new_connectivity.add_edges_from(new_edges)

    if not nx.is_connected(new_connectivity):
        raise_error(ConnectivityError, "New connectivity graph is not connected.")

    return new_connectivity


class Passes:
    """Define a transpiler pipeline consisting of smaller transpiler steps that are applied sequentially:

    Args:
        passes (list, optional): list of passes to be applied sequentially.
            If ``None``, default transpiler will be used.
            Defaults to ``None``.
        connectivity (:class:`networkx.Graph`, optional): physical qubits connectivity.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): native gates.
            Defaults to :math:`qibo.transpiler.unroller.NativeGates.default`.
        on_qubits (list, optional): list of physical qubits to be used.
            If "None" all qubits are used. Defaults to ``None``.
    """

    def __init__(
        self,
        passes: list = None,
        connectivity: nx.Graph = None,
        native_gates: NativeGates = NativeGates.default(),
        on_qubits: list = None,
    ):
        if on_qubits is not None:
            connectivity = restrict_connectivity_qubits(connectivity, on_qubits)
        self.connectivity = connectivity
        self.native_gates = native_gates
        self.passes = [] if passes is None else passes

    def __call__(self, circuit):
        """
        This function returns the compiled circuits and the dictionary mapping
        physical (keys) to logical (values) qubit.
        """

        final_layout = None
        for transpiler_pass in self.passes:
            if isinstance(transpiler_pass, Optimizer):
                transpiler_pass.connectivity = self.connectivity
                circuit = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Placer):
                transpiler_pass.connectivity = self.connectivity
                final_layout = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Router):
                transpiler_pass.connectivity = self.connectivity
                circuit, final_layout = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Unroller):
                circuit = transpiler_pass(circuit)
            else:
                raise_error(
                    TranspilerPipelineError,
                    f"Unrecognised transpiler pass: {transpiler_pass}",
                )
        return circuit, final_layout

    def is_satisfied(self, circuit: Circuit):
        """Returns ``True`` if the circuit respects the hardware connectivity and native gates, ``False`` otherwise.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be checked.

        Returns:
            (bool): satisfiability condition.
        """
        try:
            assert_placement(circuit=circuit, connectivity=self.connectivity)
            assert_connectivity(circuit=circuit, connectivity=self.connectivity)
            assert_decomposition(circuit=circuit, native_gates=self.native_gates)
            return True
        except (ConnectivityError, DecompositionError, PlacementError, ValueError):
            return False
