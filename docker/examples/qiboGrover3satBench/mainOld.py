"""Exacttly-1 3-Sat Grover algorithm.

The Grover algorithm takes a black-box oracle implementing a function
{f(x) = 1 if x==x', f(x) = 0 if x!= x'} and finds x' within a randomly
ordered sequence of N items using O(sqrt(N)) operations and O(N log(N)) gates,
with the probability p >= 2/3.




=== REFERENCE ===
An Introduction to Quantum Computing, Without the Physics
Giacomo Nannicini
https://arxiv.org/abs/1708.03684

https://www.linkedin.com/pulse/exactly-1-3-sat-problem-grovers-algorithm-breaking-rules-porfiris/

=== EXAMPLE OUTPUT ===
Solution: 0110000101
8 clause of 3 boolean variables out of n, (v1,v2,...vn), (1=T, 0=F): [' 4 5 8\n', ' 5 6 10\n', ' 4 6 8\n', ' 3 4 7\n', ' 1 2 6\n', ' 1 6 8\n', ' 1 9 10\n', ' 2 5 7\n']
Number of Grover operators R= 25
Sampled results:
Counter({'0110000101': 8, '0000010000': 1, '0100101001': 1, '0000101100': 1, '0001000101': 1,
'0011011100': 1, '1100100000': 1, '0100110101': 1, '0000000100': 1, '0011010000': 1, '1010011100': 1,
'1100001010': 1, '0010000111': 1})

Most common bitstring: 0110000101
Found a match: True


"""

import argparse
from timeit import default_timer as timer
from functionsOld import *
from qibo.models import Circuit
from qibo import gates
import qibo

def main(filename):
    """Grover search for the instance defined by the file_name.
    Args:
        filename (str): name of the file that contains the information of a 3SAT instance
    Returns:
        result of the Grover search and comparison with the expected solution if given.
    """
    #qibo.set_device("/CPU:0")
    t0 = timer()
    [num_qubits, num_clauses, xbits, clauses] = read_3sat_file(filename)
    print('Solution: {}'.format(xbits))
  

    # We start by preparing a quantum circuit for n qubits for input,  and 1 for output.
#    (input_qubits, output_qubit, aux) = set_io_qubits(num_qubits, num_clauses)
    [input_qubits, output_qubit, aux] = def_io_qubits(num_qubits, num_clauses)
    print('{} clause of 3 boolean variables out of {}, (v1,v2,...vn), (1=T, 0=F): {}'.format(num_clauses, num_qubits,
                                                                                             clauses))
    # Initialize qubits.
    circuit = Circuit(num_qubits+num_clauses+1)
    circuit.add(gates.X(output_qubit))
    circuit.add(gates.H(output_qubit))
    for i in input_qubits:
        circuit.add(gates.H(i))

    # Find the value recognized by the oracle using sqrt(N) Groover operators.
    #    num_grover_operators = int(np.floor(np.pi*np.sqrt(2**num_qubits /num_clauses )/4))
    num_grover_operators = int(np.floor(np.pi * np.sqrt(2 ** num_qubits) / 4))
    print('Number of Grover operators R= {}'.format(num_grover_operators))


    make_grover_operator(circuit, input_qubits, output_qubit, aux, clauses,num_grover_operators)




    # Sample from the circuit "rep" times.
    result = circuit.execute(nshots=5)
    print("Elapsed time: ", timer()-t0)
    #  Show results
    show_results(result, xbits)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default='data/10bit/n10i10.txt', type=str)
    args = vars(parser.parse_args())
    file_name = args.get('file')
    main(file_name)

