import numpy as np
import scipy

from qibo import gates
from qibo.models.qdp.quantum_dynamic_programming import (
    AbstractQuantumDynamicProgramming,
    SequentialInstruction,
)
from qibo.transpiler.unitary_decompositions import two_qubit_decomposition


def unitary_expm(H, t):
    U = scipy.linalg.expm(-1j * t * H)
    return U


class TwoQubitsSequentialInstruction(AbstractQuantumDynamicProgramming):
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
        # 2 quibits
        for _register in self.list_id_current_instruction_reg[0::2]:
            self.memory_usage_query_circuit()
            self.trace_one_instruction_qubit(_register)
            self.trace_one_instruction_qubit(_register + 1)
            if self.instruction_index(_register) + 2 < len(
                list(self.list_id_current_instruction_reg)
            ):
                self.increment_current_instruction_register()
                self.increment_current_instruction_register()
            self.instruction_reg_delegation()


class ObliviousSchmidtDecompositionSingleQubit(SequentialInstruction):
    """
    Subclass of AbstractQuantumDynamicProgramming for Oblivious Schmidt Decomposition

    Args:
        t: total duration
        num_iter: number of Trotterization step
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int): Number of memory units per call.
    """

    def __init__(self, t, num_work_qubits, num_instruction_qubits, number_muq_per_call):
        super().__init__(
            num_work_qubits, num_instruction_qubits, number_muq_per_call, circuit=None
        )
        self.t = t
        self.id_current_work_reg = self.list_id_work_reg[0]

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))

    def memory_usage_query_circuit(self):
        """Defines the memory usage query circuit."""
        state_01 = np.array([0, 1, 0, 0])
        state_10 = np.array([0, 0, 1, 0])
        operator_0 = np.outer(state_10, state_01)
        operator_1 = np.outer(state_01, state_10)
        N = operator_0 + operator_1
        N_unitary = scipy.linalg.expm(-1j * N * self.t)
        for decomposed_gate in two_qubit_decomposition(
            self.id_current_work_reg,
            self.id_current_instruction_reg,
            unitary=N_unitary,
        ):
            self.c.add(decomposed_gate)

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.RX(instruction_qubit, np.pi / 2))  # rho_A
            # self.c.add(gates.X(instruction_qubit+1)) #rho_B


class ObliviousSchmidtDecompositionMultiQubits(TwoQubitsSequentialInstruction):
    """
    Subclass of AbstractQuantumDynamicProgramming for Oblivious Schmidt Decomposition

    Args:
        t: total duration
        num_iter: number of Trotterization step
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int): Number of memory units per call.
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
        """Initializes the instruction qubits."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))

    def memory_usage_query_circuit(self):
        """Defines the memory usage query circuit."""
        D = np.array([[1, 0], [0, -1]])
        unitary_D = unitary_expm(D, self.t)
        self.c.add(gates.Unitary(unitary_D, self.id_current_work_reg))
        unitary_minus_D = unitary_expm(-D, self.t)
        self.c.add(gates.Unitary(unitary_minus_D, self.id_current_work_reg))
        delta_swap = unitary_expm(
            gates.SWAP(
                self.id_current_work_reg, self.id_current_instruction_reg
            ).matrix(),
            self.t,
        )

        for decomposed_gate in two_qubit_decomposition(
            self.id_current_work_reg,
            self.id_current_instruction_reg,
            unitary=delta_swap,
        ):
            self.c.add(decomposed_gate)

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.RX(instruction_qubit, np.pi / 2))  # rho_A
            # self.c.add(gates.X(instruction_qubit+1)) #rho_B
