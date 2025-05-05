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

    def __init__(self):
        """Identify and remove pairs of adjacent gates from
        a quantum circuit.
        """

        self.inverse_gates = (
            gates.H | gates.Y | gates.Z | gates.X | gates.CNOT | gates.CZ | gates.SWAP
        )
        self.pos_rm_inv_gate = []

    def __call__(self, circuit: Circuit) -> Circuit:
        """Perform pattern recognition to detect and eliminate adjacent gate pairs
        in the quantum circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to have gates identified.

        Returns:
            circuit (:class:`qibo.models.circuit.Circuit`): A new circuit with the pairs of
            adjacent gates removed.
        """

        if 0 == circuit.ngates or circuit.gates_of_type(gates.FusedGate):
            return circuit

        self._find_pos_rm(circuit.nqubits, circuit.queue)

        if 0 == len(self.pos_rm_inv_gate):
            return circuit

        tmp_new_circuit = circuit.copy(True)
        new_circuit = circuit.__class__(**tmp_new_circuit.init_kwargs)
        for i, gate in enumerate(tmp_new_circuit.queue):
            if not i in self.pos_rm_inv_gate:
                new_circuit.add(gate)
        return new_circuit

    def _find_pos_rm(self, nqubits: int, list_gates: list):
        """Identify and mark pairs of inverse gates that
        can be removed from the circuit.

        Args:
            nqubits (int): number of qubits.
            list_gates (list): a list of gates (:class:`qibo.gates.abstract.Gate`).
        """

        previous_gates = [None] * nqubits

        for i, gate in enumerate(list_gates):

            primary_qubit = gate.init_args[0]
            other_qubits = gate.init_args[1:]

            if previous_gates[primary_qubit] is None:
                self._update_previous_gates(gate.init_args, i, gate, previous_gates)

            elif isinstance(gate, self.inverse_gates):
                same_gate = self._same_gates(gate, previous_gates[primary_qubit][1])

                secondary_match = True
                if other_qubits:  # for multi-qubits gates
                    conditions = []

                    for q in other_qubits:
                        has_previous_gate = previous_gates[q] is not None
                        is_same_gate = self._same_gates(gate, previous_gates[q][1])
                        conditions.append(has_previous_gate and is_same_gate)

                    secondary_match = all(conditions)

                if same_gate and secondary_match:
                    self.pos_rm_inv_gate.extend([previous_gates[primary_qubit][0], i])

                    for q in gate.init_args:
                        previous_gates[q] = None
                else:
                    self._update_previous_gates(
                        gate.init_args, i, gate, previous_gates
                    )
            else:
                self._update_previous_gates(gate.init_args, i, gate, previous_gates)

    @staticmethod
    def _update_previous_gates(qubits, idx, gate, previous_gates):
        """Helper function to update previous gate tracking."""

        for q in qubits:
            previous_gates[q] = (idx, gate)

    @staticmethod
    def _same_gates(gate1: gates.Gate, gate2: gates.Gate) -> bool:
        """Determine whether two gates are considered the same.

        Args:
            gate1: The first gate (:class:`qibo.gates.abstract.Gate`).
            gate2: The second gate (:class:`qibo.gates.abstract.Gate`).

        Returns:
            True if the gates are the same, otherwise False.
        """

        if gate1 is None or gate2 is None:
            return False

        paramaters_gate1 = (gate1.name, gate1.init_args, gate1.init_kwargs)
        paramaters_gate2 = (gate2.name, gate2.init_args, gate2.init_kwargs)

        return bool(paramaters_gate1 == paramaters_gate2)


class RotationGateFusion(Optimizer):

    def __init__(self):
        """Identify and fuse rotated gates (RX, RY, RZ) from
        a quantum circuit.
        """

        self.rotated_gates = gates.RX | gates.RY | gates.RZ
        self.gates = []

    def __call__(self, circuit: Circuit) -> Circuit:
        """Find and combine RX, RY, and RZ rotation gates in a quantum circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to have gates identified.

        Returns:
            circuit (:class:`qibo.models.circuit.Circuit`): A new circuit with the
            fuse-rotated gates removed
        """

        if 0 == circuit.ngates or circuit.gates_of_type(gates.FusedGate):
            return circuit

        tmp_new_circuit = circuit.copy(True)
        self._merge_rotation_gates(tmp_new_circuit.nqubits, tmp_new_circuit.queue)

        new_circuit = circuit.__class__(**tmp_new_circuit.init_kwargs)
        for gate in self.gates:
            new_circuit.add(gate)
        return new_circuit

    def _merge_rotation_gates(self, nqubits: int, list_gates: list):
        """Identify and accumulate rotation angles for
        consecutive rotation gates of the same type.

        Args:
            nqubits (int): number of qubits.
            list_gates (list): a list of gates (:class:`qibo.gates.abstract.Gate`).
        """

        previous_gates = [None] * nqubits

        for gate in list_gates:

            primary_qubit = gate.init_args[0]

            if isinstance(gate, self.rotated_gates):
                prev_gate = previous_gates[primary_qubit]
                if isinstance(prev_gate, gate.__class__):
                    tmp_gate = gate.__class__(
                        primary_qubit, prev_gate.parameters[0] + gate.parameters[0]
                    )
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
        self.gates.extend(
            g for g in previous_gates if isinstance(g, self.rotated_gates)
        )


class U3GateFusion(Optimizer):

    def __init__(self):
        """Merge pairs of U3 gates in the circuit"""

        self.gates = []

    def __call__(self, circuit: Circuit) -> Circuit:
        """Optimize the circuit by merging U3 gate pairs.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to have gates identified.

        Returns:
            circuit (:class:`qibo.models.circuit.Circuit`): A new circuit where pairs of
            U3 gates were merged.
        """

        if 0 == circuit.ngates or circuit.gates_of_type(gates.FusedGate):
            return circuit

        tmp_new_circuit = circuit.copy(True)
        self._merge_u3gates(tmp_new_circuit.nqubits, tmp_new_circuit.queue)

        new_circuit = circuit.__class__(**tmp_new_circuit.init_kwargs)

        for gate in self.gates:
            new_circuit.add(gate)
        return new_circuit

    def _merge_u3gates(self, nqubits: int, list_gates: list):
        """Identify pairs of U3 gates that can be merged.

        Args:
            nqubits (int): number of qubits.
            list_gates (list): a list of gates (:class:`qibo.gates.abstract.Gate`).
        """

        previous_gates = [None] * nqubits

        for gate in list_gates:

            primary_qubit = gate.init_args[0]  # Extract primary qubit

            if isinstance(gate, gates.U3):
                prev_gate = previous_gates[primary_qubit]
                if isinstance(prev_gate, gates.U3):
                    tmp_gate = self._create_u3_fusion(primary_qubit, prev_gate, gate)
                    previous_gates[primary_qubit] = tmp_gate
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
    def _extract_u3_params(unitary_matrix: np.ndarray):
        """Extract the theta, phi, and lambda parameters from a fused U3 unitary matrix.

        Args:
            unitary_matrix (np.ndarray): a unitary matrix.
        """

        theta_r = 2 * np.arccos(
            np.sqrt(np.abs(unitary_matrix[0, 0] * unitary_matrix[1, 1]))
        )
        sin_r = np.sin(theta_r / 2)
        cos_r = np.cos(theta_r / 2)

        if 0 == cos_r:
            lambda_r = -1j * np.log(-1 * unitary_matrix[0, 1])
            phi_r = -1j * np.log(unitary_matrix[1, 0])
        elif 0 == sin_r:
            lambda_r = -1j * np.log(unitary_matrix[1, 1])
            phi_r = 1j * np.log(unitary_matrix[0, 0])
        else:
            phi_r = -1j * np.log(
                (unitary_matrix[1, 0] * unitary_matrix[1, 1]) / (sin_r * cos_r)
            )
            lambda_r = -1j * np.log(
                (sin_r * unitary_matrix[1, 1]) / (unitary_matrix[1, 0] * cos_r)
            )

        return theta_r, np.real(phi_r), np.real(lambda_r)

    @staticmethod
    def _create_u3_fusion(
        qubit: int, gate1: gates.Gate, gate2: gates.Gate
    ) -> gates.Gate:
        """Create a U3 gate using two U3 gates.

        Args:
            qubit: Ids of the qubit to apply the gate U3.
            gate1: The first gate (:class:`qibo.gates.abstract.Gate`).
            gate2: The second gate (:class:`qibo.gates.abstract.Gate`).

        Returns:
            :class:`qibo.gates.Gate`: a gate representing a fused U3 unitary matrix.
        """

        if gate1 is None or gate2 is None:
            raise_error(ValueError, "_create_u3_fusion: a/two gate/s is/are None.")

        if not isinstance(gate1, gates.U3) or not isinstance(gate2, gates.U3):
            raise_error(ValueError, "_create_u3_fusion: a/two gate/s is/are not U3.")

        u_final = np.dot(gate2.matrix(), gate1.matrix())
        theta, phi, lam = U3GateFusion._extract_u3_params(u_final)
        return gates.U3(qubit, theta, phi, lam)
