#!/usr/bin/env python
import argparse

import functions
import numpy as np


def main(nqubits, instance):
    """Grover search for the instance defined by the file_name.
    Args:
        nqubits (int): number of qubits for the file that contains the information of an Exact Cover instance.
        instance (int): intance used for the desired number of qubits.

    Returns:
        result of the Grover search and comparison with the expected solution if given.
    """
    control, solution, clauses = functions.read_file(nqubits, instance)
    qubits = control[0]
    clauses_num = control[1]
    steps = int((np.pi / 4) * np.sqrt(2**qubits))
    print(f"Qubits encoding the solution: {qubits}\n")
    print(f"Total number of qubits used: {qubits + clauses_num + 1}\n")
    q, c, ancilla, circuit = functions.create_qc(qubits, clauses_num)
    circuit = functions.grover(circuit, q, c, ancilla, clauses, steps)
    result = circuit(nshots=100)
    frequencies = result.frequencies(binary=True, registers=False)
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print(f"Most common bitstring: {most_common_bitstring}\n")
    if solution:
        print(f"Exact cover solution: {''.join(solution)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=10, type=int)
    parser.add_argument("--instance", default=1, type=int)
    args = vars(parser.parse_args())
    main(**args)
