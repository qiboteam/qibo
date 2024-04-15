from abc import abstractmethod
from enum import Enum, auto
import numpy as np
from qibo.config import raise_error
from qibo import gates, models

class QDP_memory_type(Enum):
    default = auto()
    reset = auto()
    quantum_measurement_emulation = auto()

class quantum_dynamic_programming():
    """
    Class representing a quantum dynamic programming algorithm.

    Args:
        num_work_qubits (int): Number of work qubits.
        num_instruction_qubits (int): Number of instruction qubits.
        number_muq_per_call (int): Number of memory units per call.
    """
    def __init__(self, num_work_qubits, num_instruction_qubits, number_muq_per_call, QME_rotation_gate = None):
        self.num_work_qubits = int(num_work_qubits)
        self.num_instruction_qubits = int(num_instruction_qubits)
        #SAM: fix id on each qubit, if workin put in work_reg, otherwise instruction_reg
        self.list_id_work_reg = np.arange(0,num_work_qubits,1)
        self.list_id_instruction_reg = np.arange(0,num_instruction_qubits,1) + num_work_qubits
        #SAM: since instruction can be multi qubit, id_current_instruction_reg should be list/arr
        self.id_current_instruction_reg = self.list_id_instruction_reg[0]
        self.M = number_muq_per_call
        self.memory_type = QDP_memory_type.default
        self.c = models.Circuit(self.num_work_qubits+ self.num_instruction_qubits)
        self.QME_rotation_gate = QME_rotation_gate
    
    def __call__(self,num_instruction_qubits_per_query):
        # return entire circuit
        return self.memory_call_circuit(num_instruction_qubits_per_query)
        # for now we assume that the recursion step make only 1 type memory call per QDP step

    
    def memory_call_circuit(self, num_instruction_qubits_per_query):
        # return the entire circuit
        if self.memory_type == QDP_memory_type.default:
            self.list_id_current_instruction_reg = self.list_id_instruction_reg[self.id_current_instruction_reg:self.M*num_instruction_qubits_per_query+self.id_current_instruction_reg]-1
            self.instruction_qubits_initialization()
            for register in self.list_id_current_instruction_reg:
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
            if self.QME_rotation_gate == None:
                raise TypeError("Rotation gate for QME protocol is not set.")
            for _register_use in range(self.M):
                self.memory_usage_query_circuit()
                self.trace_instruction_qubit()
                self.QME(self.id_current_instruction_reg,self.QME_rotation_gate)
                self.instruction_reg_delegation()

    def single_register_reset(self,register):
        self.c.add(gates.reset(register))

    def all_register_reset(self):
        for qubit in self.num_instruction_qubits:
            self.c.add(gates.reset(qubit))
        self.id_current_instruction_reg = 0
    
    def circuit_reset(self):
        self.c = models.Circuit(self.num_work_qubits + self.num_instruction_qubits)
    
    def increment_current_instruction_register(self):
        self.id_current_instruction_reg += 1
      
    @abstractmethod
    def memory_usage_query_circuit(self):
        # N is the gate
        #todo: ask them how to change gates duration
        #self.c.add(gates.N(id_work_reg,id_instruction_reg))
        raise_error(NotImplementedError)
    
    def instruction_qubits_initialization(self):
        pass
    
    def QME(self,register,rotation_gate):
        import random
        coin_flip = random.choice([0, 1])
        if coin_flip == 0:
            QME_gate = rotation_gate(np.pi,register) # nu is the vector parallel to rho
        elif coin_flip == 1:
            QME_gate = gates.I(register)
        self.c.add(QME_gate)

    def trace_instruction_qubit(self):
        for qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.M(qubit))

    def instruction_reg_delegation(self):
        '''Function to use a work qubit (a qubit that is in list_id_work_reg) as a
        instruction qubit (a qubit that is in list_id_instruction_reg)'''
        pass

class density_matrix_exponentiation(quantum_dynamic_programming):
    def __init__(self, theta, N, num_work_qubits, num_instruction_qubits, number_muq_per_call):
        super().__init__(num_work_qubits, num_instruction_qubits, number_muq_per_call)
        self.theta = theta # overall rotation angle
        self.N = N # number of steps
        self.delta = theta/N # small rotation angle
        self.memory_type = QDP_memory_type.default
        self.id_curent_work_reg = self.list_id_work_reg[0]

    def memory_usage_query_circuit(self):
        self.c.add(gates.SWAP(self.id_curent_work_reg,self.id_current_instruction_reg))

    def instruction_qubits_initialization(self):
        for instruction_qubit in self.list_id_current_instruction_reg:
            self.c.add(gates.X(instruction_qubit))