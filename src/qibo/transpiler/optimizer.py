from typing import Optional

import networkx as nx
import numpy as np

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


class InverseCancellation(Optimizer):
    """
        Identifies and removes pairs of adjacent gates from a quantum circuit.
    """

    def __init__(self):
        self.inverse_gates = gates.H|gates.Y|gates.Z|gates.X|gates.CNOT|gates.CZ|gates.SWAP
        self.pos_rm_inv_gate = []

    def __call__(self, circuit: Circuit) -> Circuit:
        self.__find_pos_rm(circuit)

        if (0 == len(self.pos_rm_inv_gate) or circuit.gates_of_type(gates.FusedGate)):
            return circuit

        tmp_new_circuit = circuit.copy(True)

        new_circuit = circuit.__class__(**tmp_new_circuit.init_kwargs)
        for i, gate in enumerate(tmp_new_circuit.queue):
            if not (i in self.pos_rm_inv_gate):
                new_circuit.add(gate)

        return new_circuit

    def __find_pos_rm(self, newCircuit: Circuit):
        """
            Identifies and marks pairs of inverse gates that can be removed from the circuit.

            Args:
                 newCircuit: The Circuit.

        """

        previous_gates = [None] * newCircuit.nqubits

        def update_previous_gates(qubits, idx, gate):
            """Helper function to update previous gate tracking."""
            for q in qubits:
                previous_gates[q] = (idx, gate)

        def clear_previous_gates(qubits):
            """Helper to clear tracking for given qubits."""
            for q in qubits:
                previous_gates[q] = None

        for i, gate in enumerate(newCircuit.queue):

            if not gate.init_args:
                continue

            primary_qubit, *other_qubits = gate.init_args

            if previous_gates[primary_qubit] is None:
                 update_previous_gates(gate.init_args, i, gate)
            else:
                if isinstance(gate, self.inverse_gates):
                    same_gate = self.__sameGates(gate, previous_gates[primary_qubit][1])

                    secondary_match = all(
                      previous_gates[q] is not None and self.__sameGates(gate, previous_gates[q][1])
                      for q in other_qubits
                    ) if other_qubits else True

                    if same_gate and secondary_match:
                        self.pos_rm_inv_gate.extend([previous_gates[primary_qubit][0], i])
                        clear_previous_gates(gate.init_args)
                    else:
                        update_previous_gates(gate.init_args, i, gate)
                else:
                   update_previous_gates(gate.init_args, i, gate)


    @staticmethod
    def __sameGates(gate1, gate2)-> bool:
        """
        Determines whether two gates are considered the same.

        Args:
            gate1: The first gate.
            gate2: The second gate.

        Returns:
            True if the gates are the same, otherwise False.
        """
        if gate1 is None or gate2 is None:
            return False

        return (gate1.name, gate1.init_args, gate1.init_kwargs) == (gate2.name, gate2.init_args, gate2.init_kwargs)


class RotationGateFusion(Optimizer):
    """
        Identifies and fuse rotated gates (RX, RY, RZ) from a quantum circuit.
    """

    def __init__(self):
        self.rotated_gates = gates.RX|gates.RY|gates.RZ
        self.gates = []

    def __call__(self, circuit: Circuit) -> Circuit:

        if (0 == circuit.ngates or circuit.gates_of_type(gates.FusedGate)):
            return circuit

        tmp_new_circuit = circuit.copy(True)
        self.__find_gates(tmp_new_circuit)

        new_circuit = circuit.__class__(**tmp_new_circuit.init_kwargs)
        for gate in self.gates:
            new_circuit.add(gate)

        return new_circuit

    def __find_gates(self, newCircuit: Circuit):
        """
            Identifies and accumulates rotation angles for consecutive rotation gates of the same type.
        """

        previous_gates = [None] * newCircuit.nqubits

        for gate in newCircuit.queue:

            if not gate.qubits:
                continue

            primary_qubit = gate.init_args[0]

            if isinstance(gate, self.rotated_gates):
                prev_gate = previous_gates[primary_qubit]
                if isinstance(prev_gate, type(gate)):
                    tmp_gate = type(gate)(primary_qubit, prev_gate.parameters[0] + gate.parameters[0])
                    previous_gates[primary_qubit] = tmp_gate
                else:
                    if isinstance(prev_gate, self.rotated_gates):
                        self.gates.append(prev_gate)
                    previous_gates[primary_qubit] = gate
            else:
                # Flush stored rotations before adding new gate
                for q in gate.init_args:
                    if isinstance(previous_gates[q], self.rotated_gates):
                        self.gates.append(previous_gates[q])
                        previous_gates[q] = None

                self.gates.append(gate)
                for q in gate.qubits:
                    previous_gates[q] = gate

        # Append any remaining rotation gates
        self.gates.extend(g for g in previous_gates if isinstance(g, self.rotated_gates))


class U3GateFusion(Optimizer):
    """ Merge pairs of U3 gates in the circuit"""

    def __init__(self):
        self.gates = []

    def __call__(self, circuit: Circuit) -> Circuit:

        if (0 == circuit.ngates or circuit.gates_of_type(gates.FusedGate)):
            return circuit

        tmp_new_circuit = circuit.copy(True)
        self.__find_gates(tmp_new_circuit)

        new_circuit = circuit.__class__(**tmp_new_circuit.init_kwargs)

        for gate in self.gates:
            new_circuit.add(gate)

        return new_circuit

    def __find_gates(self, newCircuit: Circuit) -> Circuit:
        """
            Identifies pairs of U3 gates that can be merged
        """

        previous_gates = [None] * newCircuit.nqubits

        for gate in newCircuit.queue:

            if not gate.qubits:
                continue

            primary_qubit = gate.init_args[0]  # Extract primary qubit

            if isinstance(gate, gates.U3):
                prev_gate = previous_gates[primary_qubit]
                if isinstance(prev_gate, gates.U3):
                    previous_gates[primary_qubit] = self.__U3Fusion(primary_qubit, prev_gate, gate)
                else:
                    previous_gates[primary_qubit] = gate
            else:
                # Flush stored U3 before adding new gate
                for q in gate.init_args:
                    if isinstance(previous_gates[q], gates.U3):
                        self.gates.append(previous_gates[q])
                        previous_gates[q] = None

                self.gates.append(gate)
                for q in gate.qubits:
                    previous_gates[q] = gate  # Track current gate

        # Append any remaining U3 gates in one pass
        self.gates.extend(g for g in previous_gates if isinstance(g, gates.U3))

    @staticmethod
    def __extract_u3_params(U):
        """ Extracts the theta, phi, and lambda parameters from a fused U3 unitary matrix. """

        theta_f = 2 * np.arccos(np.sqrt(np.abs(U[0, 0]*U[1, 1])))
        if  0 == np.cos(theta_f / 2) :
            lambda_f = np.real( - 1j * np.log( -1 * U[0, 1]))
            phi_f = np.real( - 1j * np.log(U[1, 0]))
        elif 0 == np.sin(theta_f / 2):
            lambda_f = np.real( - 1j * np.log( U[1, 1]))
            phi_f = np.real( 1j * np.log(U[0, 0]))
        else:
            phi_f = np.real( - 1j * np.log((U[1, 0] * U[1, 1]) / ( np.sin(theta_f / 2) * np.cos(theta_f / 2))))
            lambda_f = np.real( - 1j * np.log(( np.sin(theta_f / 2) * U[1, 1]) / (U[1, 0] * np.cos(theta_f / 2))))

        return theta_f, phi_f, lambda_f


    @staticmethod
    def __U3Fusion (qubit, gate1, gate2) -> gates:
        u_final = np.dot(gate2.matrix(), gate1.matrix())
        theta, phi, lam = U3GateFusion.__extract_u3_params(u_final)
        return gates.U3(qubit, theta, phi, lam)
