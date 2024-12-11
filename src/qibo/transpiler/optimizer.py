from typing import Optional

import networkx as nx

from qibo import gates
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.transpiler.abstract import Optimizer


class Preprocessing(Optimizer):
    """Pad the circuit with unused qubits to match the number of physical qubits.

    Args:
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
    """

    def __init__(self, connectivity: Optional[nx.Graph] = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit) -> Circuit:
        if not all(qubit in self.connectivity.nodes for qubit in circuit.wire_names):
            raise_error(
                ValueError,
                "Some wire_names in the circuit are not in the connectivity graph.",
            )

        physical_qubits = self.connectivity.number_of_nodes()
        logical_qubits = circuit.nqubits
        if logical_qubits > physical_qubits:
            raise_error(
                ValueError,
                f"The number of qubits in the circuit ({logical_qubits}) "
                + f"can't be greater than the number of physical qubits ({physical_qubits}).",
            )
        if logical_qubits == physical_qubits:
            return circuit
        new_wire_names = circuit.wire_names + list(
            self.connectivity.nodes - circuit.wire_names
        )
        new_circuit = Circuit(nqubits=physical_qubits, wire_names=new_wire_names)
        for gate in circuit.queue:
            new_circuit.add(gate)
        return new_circuit


class Rearrange(Optimizer):
    """Rearranges gates using qibo's fusion algorithm.
    May reduce number of SWAPs when fixing for connectivity
    but this has not been tested.

    Args:
        max_qubits (int, optional): Maximum number of qubits to fuse.
            Defaults to :math:`1`.
    """

    def __init__(self, max_qubits: int = 1):
        self.max_qubits = max_qubits

    def __call__(self, circuit: Circuit):
        fused_circuit = circuit.fuse(max_qubits=self.max_qubits)
        new = circuit.__class__(nqubits=circuit.nqubits, wire_names=circuit.wire_names)
        for fgate in fused_circuit.queue:
            if isinstance(fgate, gates.FusedGate):
                new.add(gates.Unitary(fgate.matrix(), *fgate.qubits))
            else:
                new.add(fgate)
        return new
