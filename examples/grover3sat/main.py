#!/usr/bin/env python
import numpy as np
import functions
import argparse


_PARAM_NAMES = {"theta", "phi"}
parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=10, type=int)
args = vars(parser.parse_args())


def main(file_name):
    """Grover search for the instance defined by the file_name.
    Args:
        file_name (str): name of the file that contains the information of a 3SAT instance

    Returns:
        result of the Grover search and comparison with the expected solution if given.
    """
    control, solution, clauses = functions.read_file(file_name)
    qubits = control[0]
    clauses_num = control[1]
    steps = int((np.pi/4)*np.sqrt(2**qubits))
    print('Qubits encoding the solution: {}\n'.format(qubits))
    print('Total number of qubits used:  {}\n'.format(qubits + clauses_num + 1))
    q, c, ancilla, circuit = functions.create_qc(qubits, clauses_num)
    circuit = functions.grover(circuit, q, c, ancilla, clauses, steps)
    result = circuit(nshots=100)
    frequencies = result.frequencies(binary=True, registers=False)
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print('Most common bitstring: {}\n'.format(most_common_bitstring))
    if ''.join(solution) != '0'*qubits:
        print('Exact cover solution:  {}\n'.format(''.join(solution)))


file_name = 'n{}.txt'.format(args.get('nqubits'))
main(file_name)
