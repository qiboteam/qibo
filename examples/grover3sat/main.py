#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import functions
import argparse


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
    print("Qubits encoding the solution: {}\n".format(qubits))
    print("Total number of qubits used:  {}\n".format(qubits + clauses_num + 1))
    q, c, ancilla, circuit = functions.create_qc(qubits, clauses_num)
    circuit = functions.grover(circuit, q, c, ancilla, clauses, steps)
    result = circuit(nshots=100)
    frequencies = result.frequencies(binary=True, registers=False)
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print("Most common bitstring: {}\n".format(most_common_bitstring))
    if solution:
        print("Exact cover solution:  {}\n".format("".join(solution)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=10, type=int)
    parser.add_argument("--instance", default=1, type=int)
    args = vars(parser.parse_args())
    main(**args)
