
# coding: utf-8

import random
import numpy as np

import qibo
from qibo import gates
from qibo import quantum_info


class State:
    def __init__(self, n):
        '''
        Initializes State class
        - n: number of qubits
        - quantum register: object to hold quantum information
        - classical register: object to hold classical information
        - circuit_name: circuit name; defined in each subclass (GHZState, HadamardState, RandomState)
        '''
    
        self.n = n
        
        self.circuit_name = None
        self.circuit      = None
        self.measurement_circuit_names = []
        self.measurement_circuits      = []

        
    def create_circuit(self):
        raise NotImplemented

    
    def execute_circuit(self):
        # XXX not needed?
        pass

    
    def get_state_vector(self):
        '''
        Executes circuit by connecting to Qiskit object, and obtain state vector
        '''
        if self.circuit is None:
            self.create_circuit()
        
        # XXX add probe?
        qibo.set_backend("numpy")
        result  = self.circuit.execute()
        state_vector = result.to_dict()['state']
        return state_vector        

    
    def get_state_matrix(self):
        '''
        Obtain density matrix by taking an outer product of state vector
        '''
        state_vector = self.get_state_vector()
        state_matrix = np.outer(state_vector, state_vector.conj())
        return state_matrix
    
    
    def create_measurement_circuits(self, labels, label_format='little_endian'):
        '''
        Prepares measurement circuits
        - labels: string of Pauli matrices (e.g. XYZXX)
        '''
        
        if self.circuit is None:
            self.create_circuit()
        
        qubits = range(self.n)
        
        for label in labels:

            # for aligning to the natural little_endian way of iterating through bits below
            if label_format == 'big_endian':
                effective_label = label[::-1]
            else:
                effective_label = label[:]
            probe_circuit = qibo.Circuit(self.n)

            for qubit, letter in zip(*[qubits, effective_label]): 
                if letter == 'X':
                    probe_circuit.add(gates.H(qubit)) # H

                elif letter == 'Y':
                    probe_circuit.add(gates.S(qubit).dagger()) # S^dg
                    probe_circuit.add(gates.H(qubit))          # H

            probe_circuit.add(gates.M(*qubits))
                
            measurement_circuit_name = self.make_measurement_circuit_name(self.circuit_name, label)    
            measurement_circuit      = self.circuit + probe_circuit
            self.measurement_circuit_names.append(measurement_circuit_name)
            self.measurement_circuits.append(measurement_circuit)
        

    @staticmethod    
    def make_measurement_circuit_name(circuit_name, label):
        '''
        Measurement circuit naming convention
        '''
        name = '%s-%s' % (circuit_name, label)
        return name
    
    @staticmethod
    def sort_sample(sample) -> dict:
        '''
        Sort raw measurements into count dictionary
        '''
        sorted_sample = {}
        for shot in sample:
            s = ''.join(map(str, shot))
            if s in sorted_sample:
                sorted_sample[s] += 1
            else:
                sorted_sample[s] = 1
        return sorted_sample

    
    def execute_measurement_circuits(self, labels,
                                     backend   = 'numpy',
                                     num_shots = 100,
                                     #num_shots = 10000,
                                     label_format='little_endian'):
        '''
        Executes measurement circuits
        - labels: string of Pauli matrices (e.g. XYZXX)
        - backend: 'numpy', 'qibojit', 'pytorch', 'tensorflow' prvided by qibo 
        - num_shots: number of shots measurement is taken to get empirical frequency through counts
        '''
        if self.measurement_circuit_names == []:
            self.create_measurement_circuits(labels, label_format)
        
        circuit_names = self.measurement_circuit_names

        qibo.set_backend(backend)
 
        
        data_dict_list = []
        for i, label in enumerate(labels):
            result = self.measurement_circuits[i].execute(nshots=num_shots)
            result.samples(binary=True, registers=False)
            count_dict = self.sort_sample(result.to_dict()['samples'])

            measurement_circuit_name = self.make_measurement_circuit_name(self.circuit_name, label)
            data_dict = {'measurement_circuit_name' : measurement_circuit_name,
                         'circuit_name'             : self.circuit_name,
                         'label'                    : label,
                         'count_dict'               : count_dict,
                         'backend'                  : 'qibo: '+ backend,
                         'num_shots'                : num_shots}
            data_dict_list.append(data_dict)
        return data_dict_list


class GHZState(State):
    '''
    Constructor for GHZState class
    '''
    def __init__(self, n):
        State.__init__(self, n)
        self.circuit_name = 'GHZ'

        
    def create_circuit(self):
        circuit = qibo.Circuit(self.n)        

        circuit.add(gates.H(0))
        for i in range(1, self.n):
            circuit.add(gates.CNOT(0, i))
        
        self.circuit = circuit

        
class HadamardState(State):
    '''
    Constructor for HadamardState class
    '''
    def __init__(self, n):        
        State.__init__(self, n)
        self.circuit_name = 'Hadamard'
        
        
    def create_circuit(self):
        circuit = qibo.Circuit(self.n) 
        
        for i in range(self.n):
            circuit.add(gates.H(i))
        
        self.circuit = circuit

        
class RandomState(State):
    '''
    Constructor for RandomState class
    '''
    def __init__(self, n, seed=0, depth=40):
        State.__init__(self, n)
        self.circuit_name = 'Random-%d' % (self.n, )

        self.seed  = seed
        self.depth = depth

        
    def create_circuit(self):
        random.seed(a=self.seed)
        circuit = qibo.Circuit(self.n)

        for j in range(self.depth):
            if self.n == 1:
                op_ind = 0
            else:
                op_ind = random.randint(0, 1)
            if op_ind == 0: # U3
                qind = random.randint(0, self.n - 1)
                circuit.add(gates.U3(qind, 2*np.pi*random.random(),
                                           2*np.pi*random.random(),
                                           2*np.pi*random.random(), trainable=True))
            elif op_ind == 1: # CX
                source, target = random.sample(range(self.n), 2)
                circuit.add(gates.CNOT(source, target))
        
        self.circuit = circuit


if __name__ == '__main__':

    ############################################################
    ### Example of creating and running an experiment
    ############################################################

    n = 3
    #labels = projectors.generate_random_label_list(20, n)
    labels = ['YXY', 'IXX', 'ZYI', 'XXX', 'YZZ']
    #labels = ['YZYX', 'ZZIX', 'XXIZ', 'XZIY', 'YXYI', 'ZYYX', 'YXXX', 'IIYY', 'ZIXZ', 'IXXI', 'YZXI', 'ZZYI', 'YZXY', 'XYZI', 'XZXI', 'XZYX', 'YIXI', 'IZYY', 'ZIZX', 'YXXY']
    #labels = ['IIIX', 'IYIY', 'YYXI', 'ZZYY', 'ZYIX', 'XIII', 'XXZI', 'YXZI', 'IZXX', 'YYIZ', 'XXIY', 'XXZY', 'ZZIY', 'YIYX', 'YYZZ', 'YZXZ', 'YZYZ', 'ZXYY', 'IXIZ', 'XZII']
    #labels = Generate_All_labels(n)


    #state   = GHZState(n)
    #state   = HadamardState(n)
    state   = RandomState(n)

    state.create_circuit()
    data_dict_list = state.execute_measurement_circuits(labels)
    #print(data_dict_list)

    target_density_matrix = state.get_state_matrix()
    target_state          = state.get_state_vector()            
    #print(state.get_state_vector())

    Nr = 3
    random_DM = quantum_info.random_density_matrix(2**n, Nr)
