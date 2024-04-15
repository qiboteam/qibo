from abc import abstractmethod
from enum import Enum, auto
import numpy as np
from qibo.config import raise_error
from qibo import gates, models

class QDP_memory_type(Enum):
    """
    Enumerated type representing memory types for quantum dynamic programming.
    """
    default = auto()
    reset = auto()
    quantum_measurement_emulation = auto()

class quantum_dynamic_programming:
    """
    Class representing a quantum dynamic programming algorithm.

    Args:
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int): Number of memory units per call.
        QME_rotation_gate (callable): Optional. Rotation gate for quantum measurement emulation.
    """
    def __init__(self, num_work_qubits, num_instruction_qubits, number_muq_per_call, QME_rotation_gate=None):
        self.num_work_qubits = int(num_work_qubits)
        self.num_instruction_qubits = int(num_instruction_qubits)

        self.list_id_work_reg = np.arange(0, num_work_qubits, 1)
        self.list_id_instruction_reg = np.arange(0, num_instruction_qubits, 1) + num_work_qubits

        self.id_current_instruction_reg = self.list_id_instruction_reg[0]

        self.M = number_muq_per_call
        self.memory_type = QDP_memory_type.default
        self.c = models.Circuit(self.num_work_qubits + self.num_instruction_qubits)
        self.QME_rotation_gate = QME_rotation_gate

    def __call__(self, num_instruction_qubits_per_query):
        """
        Calls the memory call circuit.

        Args:
            num_instruction_qubits_per_query (int): Number of instruction qubits per query.

        Returns:
            qibo.models.Circuit: Entire quantum circuit.
        """
        return self.memory_call_circuit(num_instruction_qubits_per_query)

    @abstractmethod
    def memory_usage_query_circuit(self):
        """
        Defines the memory usage query circuit.
        """
        raise_error(NotImplementedError)

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits."""
        pass

    def QME(self, register, QME_rotation_gate):
        """
        Performs quantum measurement emulation.

        Args:
            register (int): The register index.
            rotation_gate (callable): The rotation gate about an axis parallel 
            to the instruction qubit(s).
        """
        import random
        coin_flip = random.choice([0, 1])
        if coin_flip == 0:
            QME_gate = QME_rotation_gate(np.pi, register) 
        elif coin_flip == 1:
            QME_gate = gates.I(register)
        self.c.add(QME_gate)

    def trace_instruction_qubit(self):
        """Traces the instruction qubit."""
        for qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.M(qubit))

    def instruction_reg_delegation(self):
        """
        Uses a work qubit as an instruction qubit.
        """
        pass

    def increment_current_instruction_register(self):
        """Increments the current instruction register index."""
        self.id_current_instruction_reg += 1

    def memory_call_circuit(self, num_instruction_qubits_per_query):
        """
        Executes the memory call circuit based on the selected memory type.

        Args:
            num_instruction_qubits_per_query (int): Number of instruction qubits per query.
        """
        if self.memory_type == QDP_memory_type.default:
            self.list_id_current_instruction_reg = self.list_id_instruction_reg[
                self.id_current_instruction_reg:self.M * num_instruction_qubits_per_query + self.id_current_instruction_reg] - 1
            self.instruction_qubits_initialization()
            for _register in self.list_id_current_instruction_reg:
                self.memory_usage_query_circuit()
                self.trace_instruction_qubit()
                self.increment_current_instruction_register()
                self.instruction_reg_delegation()

        elif self.memory_type == QDP_memory_type.reset:
            for _register_use in range(self.M):
                self.memory_usage_query_circuit()
                self.trace_instruction_qubit()
                self.single_register_reset(self.id_current_instruction_reg)
                self.instruction_reg_delegation()

        elif self.memory_type == QDP_memory_type.quantum_measurement_emulation:
            if self.QME_rotation_gate is None:
                raise TypeError("Rotation gate for QME protocol is not set.")
            for _register_use in range(self.M):
                self.memory_usage_query_circuit()
                self.trace_instruction_qubit()
                self.QME(self.id_current_instruction_reg, self.QME_rotation_gate)
                self.instruction_reg_delegation()

    @abstractmethod
    def single_register_reset(self, register):
        """
        Resets a single register.

        Args:
            register (int): The register index.

        Example reset code:
            self.c.add(gates.reset(register))
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def all_register_reset(self):
        #todo: find way to do reset
        """
        Resets all instruction registers.

        Example reset code:
            for qubit in self.num_instruction_qubits:
                self.c.add(gates.reset(qubit))
            self.id_current_instruction_reg = 0
        """
        raise_error(NotImplementedError)

    def circuit_reset(self):
        """Resets the entire quantum circuit."""
        self.c = models.Circuit(self.num_work_qubits + self.num_instruction_qubits)

class density_matrix_exponentiation(quantum_dynamic_programming):
    """
    Subclass of quantum_dynamic_programming for density matrix exponentiation.

    Args:
        theta (float): Overall rotation angle.
        N (int): Number of steps.
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int): Number of memory units per call.
    """
    def __init__(self, theta, N, num_work_qubits, num_instruction_qubits, number_muq_per_call):
        super().__init__(num_work_qubits, num_instruction_qubits, number_muq_per_call)
        self.theta = theta  # overall rotation angle
        self.N = N  # number of steps
        self.delta = theta / N  # small rotation angle
        self.memory_type = QDP_memory_type.default
        self.id_current_work_reg = self.list_id_work_reg[0]

    def memory_usage_query_circuit(self):
        """Defines the memory usage query circuit."""
        self.c.add(gates.SWAP(self.id_current_work_reg, self.id_current_instruction_reg))

    def instruction_qubits_initialization(self):
        """Initializes the instruction qubits."""
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))
