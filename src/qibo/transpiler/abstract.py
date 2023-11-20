from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import Tuple

import networkx as nx

from qibo import gates
from qibo.config import raise_error
from qibo.models import Circuit


class NativeGates(Flag):
    """Define native gates supported by the unroller.
    A native gate set should contain at least one two-qubit gate (CZ or iSWAP)
    and at least one single qubit gate (GPI2 or U3).

    Should have the same names with qibo gates.
    """

    GPI2 = auto()
    U3 = auto()
    CZ = auto()
    iSWAP = auto()

    @classmethod
    def from_gate(cls, gate: gates.Gate):
        try:
            return getattr(cls, gate.__class__.__name__)
        except AttributeError:
            raise ValueError(f"Gate {gate} cannot be used as native.")

    def single_qubit_natives(self):
        return (self.GPI2, self.U3)

    def two_qubit_natives(self):
        return (self.CZ, self.iSWAP)


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


class Unroller(ABC):
    @abstractmethod
    def __init__(self, native_gates: NativeGates, *args):
        """An unroller decomposes gates into native gates."""

    @abstractmethod
    def __call__(self, circuit: Circuit, *args) -> Circuit:
        """Find initial qubit mapping

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be optimized

        Returns:
            (:class:`qibo.models.circuit.Circuit`): circuit with native gates.
        """


def _find_gates_qubits_pairs(circuit: Circuit):
    """Translate qibo circuit into a list of pairs of qubits to be used by the router and placer.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

    Returns:
        (list): list containing qubits targeted by two qubit gates.
    """
    translated_circuit = []
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            pass
        elif len(gate.qubits) == 2:
            translated_circuit.append(sorted(gate.qubits))
        elif len(gate.qubits) >= 3:
            raise_error(
                ValueError, "Gates targeting more than 2 qubits are not supported"
            )

    return translated_circuit
