
import random
import numpy as np
import matplotlib.pyplot as plt

from qibo import gates
from qibo.models import Circuit

def bitstring(bits):
    """An auxiliary function to manipulate bits"""
    return ''.join(str(int(b)) for b in bits)


"""Read a file containing an Exactly-1 3-SAT formula.

Example: 4 Bolean variables, 3 clauses. The literal are represented with integers 
indicating the Boolean variable. Second line shows the solution.

 4 3 1
0 1 0 0
 1 2 3
 2 3 4
 1 2 4

"""


def read_3sat_file(filepath):
    import sys
    import os

    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

    with open(filepath) as fp:
        header1 = fp.readline()
        [space, num_qubits, num_clauses, num_ones] = header1.split(' ')
        header2 = fp.readline()
        x_bits = bitstring(header2.split(' '))
        clauses = []
        for i in range(int(num_clauses)):
            clauses.append(fp.readline())

    return (int(num_qubits), int(num_clauses), x_bits, clauses)



def def_io_qubits(qubit_count, num_qubits_aux):
    """Define the specified number of input, output  and auxiliary qubits

    The number of input qubits is equal to the number of Boolean variables.
    There is one output qubit for function f
    The number of auxiliary qubits is equak to the nember of clauses
    """
    output_qubit = 0
    input_qubits = [i+1 for i in range(qubit_count)]
    aux = [qubit_count + 1 + i for i in range(num_qubits_aux)]
    return (input_qubits, output_qubit, aux)




def oracle(input_qubits, output_qubit, aux, clauses):
    """Circuit that implement function {f(x) = 1 if x==x', f(x) = 0 if x!= x'}.

    Create a circuit that verifies whether a given exactly-1 3-SAT formula
    is satisfied by the inputs. The exactly-1 version requieres exactly one literal
    out of every clause to be satiffied.
    """

    for (k, clause) in enumerate(clauses):
        [space, v1, v2, v3] = clause.split(' ')
        x1 = int(v1) - 1
        x2 = int(v2) - 1
        x3 = int(v3) - 1
        yield gates.CNOT(input_qubits[x1], aux[k])
        yield gates.CNOT(input_qubits[x2], aux[k])
        yield gates.CNOT(input_qubits[x3], aux[k])
        yield gates.X(aux[k]).controlled_by(input_qubits[x1], input_qubits[x2], input_qubits[x3])
    yield gates.X(output_qubit).controlled_by(*aux)
    for (k, clause) in enumerate(clauses):
        [space, v1, v2, v3] = clause.split(' ')
        x1 = int(v1) - 1
        x2 = int(v2) - 1
        x3 = int(v3) - 1
        yield gates.CNOT(input_qubits[x1], aux[k])
        yield gates.CNOT(input_qubits[x2], aux[k])
        yield gates.CNOT(input_qubits[x3], aux[k])
        yield gates.X(aux[k]).controlled_by(input_qubits[x1], input_qubits[x2], input_qubits[x3])

def amplify(input_qubits):

    """Generator that performs the inversion over the average step in Grover's search algorithm.
    Args:
        input_qubit (list): quantum register that encodes the problem.
    Returns:
        quantum gate geenrator that applies the amplification step.
    """

    for i in input_qubits:
        yield gates.H(i)
        yield gates.X(i)
    yield gates.H(input_qubits[0])
    yield gates.X(input_qubits[0]).controlled_by(*input_qubits[1:len(input_qubits)])
    yield gates.H(input_qubits[0])
    for i in input_qubits:
        yield gates.X(i)
        yield gates.H(i)


def make_grover_operator(c, input_qubits, output_qubit, aux, clauses,num_grover_operators):
    """Implement n iteration of the Grover algorithm.

    First append the oracles and then apply inversion about the average, and finally messure

    """

    # Make oracle (black box)
    #fun = oracle(input_qubits, output_qubit, aux, clauses)
    for i in range(num_grover_operators):    
        # Query oracle.
        c.add(oracle(input_qubits, output_qubit, aux, clauses))

        # Amplitud Amplification
        c.add(amplify(input_qubits))

    # Measurement.
    c.add(gates.M(*(input_qubits), register_name='result'))
    return c


def show_results(result, x_bits):
    frequencies = result.frequencies(binary=True, registers=False) 

    print('Sampled results:\n{}'.format(frequencies))
    plt.bar(list(frequencies.keys()), frequencies.values(), color='b')
    plt.show()

    # Check if we actually found the clause
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print('Most common bitstring: {}'.format(most_common_bitstring))
    print('Found a match: {}'.format(
        most_common_bitstring == bitstring(x_bits)))