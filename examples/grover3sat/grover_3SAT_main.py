from qibo import gates
from qibo.models import Circuit
from grover_3SAT_functions import *
import numpy as np

file_name = 'n10.txt'

control, solution, clauses = read_file(file_name)

qubits = control[0]
clauses_num = control[1]
steps = int((np.pi/4)*np.sqrt(2**qubits))

print('# of qubits used: {}\n'.format(qubits+clauses_num+1))

q, c, ancilla, circuit = create_qc(qubits, clauses_num)

circuit = grover(circuit, q, c, ancilla, clauses, steps)

result = circuit(nshots=100)
frequencies = result.frequencies(binary=True, registers=False)

print('Sampled results:\n{}\n'.format(frequencies))

most_common_bitstring = frequencies.most_common(1)[0][0]
print('Most common bitstring: {}\n'.format(most_common_bitstring))
print('Exact cover solution: {}\n'.format(''.join(solution)))