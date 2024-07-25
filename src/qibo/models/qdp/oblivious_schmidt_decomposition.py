import numpy as np

from qibo import gates
from qibo.models.qdp.oblivious_schmidt_decomposition import *
from qibo.models.qdp.quantum_dynamic_programming import (
    AbstractQuantumDynamicProgramming,
    SequentialInstruction,
)
from qibo.transpiler.unitary_decompositions import two_qubit_decomposition

from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

sigma_Z_1q = SymbolicHamiltonian(Z(0),1)
#sigma_Z_2q = SymbolicHamiltonian(Z(0),2)

Heisenberg = SymbolicHamiltonian(X(0)*X(1) + Y(0)*Y(1) + Z(0)*Z(1),2)
def dSWAP(t):
    return Heisenberg.exp(t*np.pi/4)

def off_diagonal_norm(H):
    """
    Compute the Hilbert-Schmidt norm of the off-diagonal part of a Hermitian matrix H.

    Args:
        H (numpy.ndarray): The Hermitian matrix.

    Returns:
        float: The Hilbert-Schmidt norm of the off-diagonal part of H.
    """
    diagonal_h = np.diag(np.diag(H))
    off_diag_h = H - diagonal_h
    off_diag_h_dag = np.asarray(np.asmatrix(off_diag_h).H)
    return np.sqrt(np.real(np.trace(off_diag_h_dag @ off_diag_h)))


class ObliviousSchmidtDecompositionSingleQubit(SequentialInstruction):
    """
    Subclass of AbstractQuantumDynamicProgramming for Oblivious Schmidt Decomposition.

    Args:
        t (float): Total duration.
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int, optional): Number of memory units per call. Defaults to 1.
    """

    def __init__(
        self, t, num_work_qubits, num_instruction_qubits, number_muq_per_call=1
    ):
        super().__init__(
            num_work_qubits, num_instruction_qubits, number_muq_per_call, circuit=None
        )
        self.t = t
        self.id_current_work_reg = self.list_id_work_reg[0]

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits by applying the X gate."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))

    def memory_usage_query_circuit(self):
        """Defines the memory usage query circuit."""

        # Add e^-itZ
        self.c.add(gates.Unitary(sigma_Z_1q.exp(-self.t), self.id_current_work_reg))
        
        # Add e^it rho using DME
        delta_swap = dSWAP(self.t)
        for decomposed_gate in two_qubit_decomposition(
            self.id_current_work_reg,
            self.id_current_instruction_reg,
            unitary=delta_swap,
        ):
            self.c.add(decomposed_gate)

        # Add e^itZ
        self.c.add(gates.Unitary(sigma_Z_1q.exp(self.t), self.id_current_work_reg))

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits by applying the RX gate with a pi/2 rotation."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.RX(instruction_qubit, np.pi / 2))  # rho_A
            # self.c.add(gates.X(instruction_qubit + 1))  # rho_B


class TwoQubitsSequentialInstruction(AbstractQuantumDynamicProgramming):
    """
    Abstract class for sequential instructions involving two qubits.

    Args:
        num_instruction_qubits_per_query (int): Number of instruction qubits per query.
    """

    def memory_call_circuit(self, num_instruction_qubits_per_query):
        """
        Executes the memory call circuit. Every instruction qubit is used once then discarded.

        Args:
            num_instruction_qubits_per_query (int): Number of instruction qubits per query.
        """
        current_instruction_index = self.instruction_index(
            self.id_current_instruction_reg
        )
        self.list_id_current_instruction_reg = self.list_id_instruction_reg[
            current_instruction_index : self.M * num_instruction_qubits_per_query
            + current_instruction_index
        ]
        self.instruction_qubits_initialization()
        for _register in self.list_id_current_instruction_reg[0::2]:
            self.memory_usage_query_circuit()
            self.trace_one_instruction_qubit(_register)
            if self.instruction_index(_register) + 1 < len(
                list(self.list_id_current_instruction_reg)
            ):
                self.trace_one_instruction_qubit(_register + 1)
            if self.instruction_index(_register) + 2 < len(
                list(self.list_id_current_instruction_reg)
            ):
                self.increment_current_instruction_register()
                self.increment_current_instruction_register()
            self.instruction_reg_delegation()


class ObliviousSchmidtDecompositionTwoQubits(TwoQubitsSequentialInstruction):
    """
    Subclass of AbstractQuantumDynamicProgramming for Oblivious Schmidt Decomposition involving two qubits.

    Args:
        t (float): Total duration.
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int, optional): Number of memory units per call. Defaults to 1.
    """

    def __init__(
        self, t, num_work_qubits, num_instruction_qubits, number_muq_per_call=1
    ):
        super().__init__(
            num_work_qubits, num_instruction_qubits, number_muq_per_call, circuit=None
        )
        self.t = t
        self.id_current_work_reg = self.list_id_work_reg[0]

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits by applying the X gate."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))

    def memory_usage_query_circuit(self):
        """Defines the memory usage query circuit."""
        # Add e^-itZ
        self.c.add(gates.Unitary(sigma_Z_1q.exp(-self.t), self.id_current_work_reg))
        self.c.add(gates.Unitary(sigma_Z_1q.exp(-self.t), self.id_current_work_reg+1))

        # Add e^it rho using DME to both data qubits
        delta_swap = dSWAP(self.t)
        for decomposed_gate in two_qubit_decomposition(
            self.id_current_work_reg,
            self.id_current_instruction_reg,
            unitary=delta_swap,
        ):
            self.c.add(decomposed_gate)

        for decomposed_gate in two_qubit_decomposition(
            self.id_current_work_reg + 1,
            self.id_current_instruction_reg + 1,
            unitary=delta_swap,
        ):
            self.c.add(decomposed_gate)

        # Add e^itZ
        self.c.add(gates.Unitary(sigma_Z_1q.exp(self.t), self.id_current_work_reg))
        self.c.add(gates.Unitary(sigma_Z_1q.exp(self.t), self.id_current_work_reg+1))
    
    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits by applying the RX gate with a pi/2 rotation."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.RX(instruction_qubit, np.pi / 2))  # rho_A
            # self.c.add(gates.X(instruction_qubit + 1))  # rho_B
