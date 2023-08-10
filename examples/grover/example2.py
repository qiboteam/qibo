import argparse

import numpy as np
from scipy.special import binom

from qibo import Circuit, gates
from qibo.models.grover import Grover


def one_sum(sum_qubits):
    c = Circuit(sum_qubits + 1)

    for q in range(sum_qubits, 0, -1):
        c.add(gates.X(q).controlled_by(*range(0, q)))

    return c


def sum_circuit(qubits):
    sum_qubits = int(np.ceil(np.log2(qubits))) + 1
    sum_circuit = Circuit(qubits + sum_qubits)
    sum_circuit.add(gates.X(qubits).controlled_by(0))
    sum_circuit.add(gates.X(qubits).controlled_by(1))
    sum_circuit.add(gates.X(qubits + 1).controlled_by(*[0, 1]))

    for qub in range(2, qubits):
        sum_circuit.add(
            one_sum(sum_qubits).on_qubits(
                *([qub] + list(range(qubits, qubits + sum_qubits)))
            )
        )

    return sum_circuit


def oracle(qubits, num_1):
    sum = sum_circuit(qubits)
    oracle = Circuit(sum.nqubits + 1)
    oracle.add(sum.on_qubits(*range(sum.nqubits)))

    booleans = np.binary_repr(num_1, int(np.ceil(np.log2(qubits)) + 1))

    for i, b in enumerate(booleans[::-1]):
        if b == "0":
            oracle.add(gates.X(qubits + i))

    oracle.add(gates.X(sum.nqubits).controlled_by(*range(qubits, sum.nqubits)))

    for i, b in enumerate(booleans[::-1]):
        if b == "0":
            oracle.add(gates.X(qubits + i))

    oracle.add(sum.invert().on_qubits(*range(sum.nqubits)))

    return oracle


def check(instance, num_1):
    res = instance.count("1") == num_1
    return res


def main(nqubits, num_1, iterative=False):
    """Create an oracle, find the states with some 1's among all the states with a fixed number of qubits
    Args:
        nqubits (int): number of qubits
        num_1 (int): number of 1's to find
        iterative (bool): use iterative model

    Returns:
        solution (str): found string
        iterations (int): number of iterations needed
    """
    oracle_circuit = oracle(nqubits, num_1)

    #################################################################
    ###################### NON ITERATIVE MODEL ######################
    #################################################################

    if not iterative:
        grover = Grover(
            oracle_circuit,
            superposition_qubits=nqubits,
            number_solutions=int(binom(nqubits, num_1)),
        )

        solution, iterations = grover()

        print("\nNON ITERATIVE MODEL: \n")

        print("The solution is", solution)
        print("Number of iterations needed:", iterations)
        print(
            "\nFound number of solutions: ",
            len(solution),
            "\nTheoretical number of solutions:",
            int(binom(nqubits, num_1)),
        )

        return solution, iterations

    #################################################################
    ######################## ITERATIVE MODEL ########################
    #################################################################

    print("\nITERATIVE MODEL: \n")

    if iterative:
        grover = Grover(
            oracle_circuit,
            superposition_qubits=nqubits,
            check=check,
            check_args=(num_1,),
        )
        solution, iterations = grover()

        print("Found solution:", solution)
        print("Number of iterations needed:", iterations)

        return solution, iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=10, type=int, help="Number of qubits.")
    parser.add_argument("--num_1", default=2, type=int, help="Number of 1's to find.")
    parser.add_argument("--iterative", action="store_true", help="Use iterative model")
    args = vars(parser.parse_args())
    main(**args)
