import argparse

import numpy as np
from scipy.special import binom as binomial

from qibo import Circuit, gates
from qibo.models import Grover


def set_ancillas_to_num(ancillas, num):
    """Set a quantum register to a specific number."""
    ind = 0
    for i in reversed(bin(num)[2:]):
        if int(i) == 1:
            yield gates.X(ancillas[ind])
        ind += 1


def add_negates_for_check(ancillas, num):
    """Adds the negates needed for control-on-zero."""
    ind = 0
    for i in reversed(bin(num)[2:]):
        if int(i) == 0:
            yield gates.X(ancillas[ind])
        ind += 1
    for i in range(len(bin(num)[2:]), len(ancillas)):
        yield gates.X(ancillas[i])


def sub_one(ancillas, controls):
    """Subtract 1 bit by bit."""
    a = ancillas
    yield gates.X(a[0]).controlled_by(*controls)
    for i in range(1, len(a)):
        controls.append(a[i - 1])
        yield gates.X(a[i]).controlled_by(*controls)


def superposition_probabilities(n, r):
    """Computes the probabilities to set the initial superposition."""

    def split_weights(n, r):
        """Auxiliary function that gets the required binomials."""
        v0 = binomial(n - 1, r)
        v1 = binomial(n - 1, r - 1)
        return v0 / (v0 + v1), v1 / (v0 + v1)

    L = []
    for i in range(n):
        for j in range(min(i, r - 1), -1, -1):
            if n - i >= r - j:
                L.append([n - i, r - j, split_weights(n - i, r - j)])
    return L


def superposition_circuit(n, r):
    """Creates an equal quantum superposition over the column choices."""
    n_anc = int(np.ceil(np.log2(r + 1)))
    ancillas = [i for i in range(n, n + n_anc)]
    c = Circuit(n + n_anc)
    c.add(set_ancillas_to_num(ancillas, r))
    tmp = n
    L = superposition_probabilities(n, r)
    for i in L:
        if tmp != i[0]:
            c.add(sub_one(ancillas, [n - tmp]))
            tmp = i[0]

        if i[2] == (0, 1):
            c.add(add_negates_for_check(ancillas, i[1]))
            c.add(gates.X(n - i[0]).controlled_by(*ancillas))
            c.add(add_negates_for_check(ancillas, i[1]))
        else:
            if i[0] != n:
                c.add(add_negates_for_check(ancillas, i[1]))
                c.add(
                    gates.RY(
                        n - i[0], float(2 * np.arccos(np.sqrt(i[2][0])))
                    ).controlled_by(*ancillas)
                )
                c.add(add_negates_for_check(ancillas, i[1]))
            else:
                c.add(gates.RY(0, float(2 * np.arccos(np.sqrt(i[2][0])))))
    c.add(sub_one(ancillas, [n - 1]))
    return c


def oracle(n, s):
    """Oracle checks whether the first s terms are 1."""
    if s > 2:
        n_anc = s - 2
        oracle = Circuit(n + n_anc + 1)
        oracle_1 = Circuit(n + n_anc + 1)
        oracle_1.add(gates.X(n + 1).controlled_by(*(0, 1)))
        for q in range(2, s - 1):
            oracle_1.add(gates.X(n + q).controlled_by(*(q, n + q - 1)))

        oracle.add(oracle_1.on_qubits(*(range(n + n_anc + 1))))
        oracle.add(gates.X(n).controlled_by(*(s - 1, n + n_anc)))
        oracle.add(oracle_1.invert().on_qubits(*(range(n + n_anc + 1))))

        return oracle

    else:
        oracle = Circuit(n + int(np.ceil(np.log2(s + 1))) + 1)
        oracle.add(gates.X(n).controlled_by(*range(s)))

        return oracle


def main(nqubits, num_1):
    """Creates a superposition circuit that finds all states with num_1 1's in a
    fixed number of qubits, then the oracle find that state where all the 1's
    are at the beginning of the bitstring. This oracle has got ancillas

    Args:
        nqubits (int): number of qubits
        num_1 (int): number of 1's to find

    Returns:
        solution (str): found string
        iterations (int): number of iterations needed
    """
    superposition = superposition_circuit(nqubits, num_1)

    oracle_circuit = oracle(nqubits, num_1)
    or_circuit = Circuit(oracle_circuit.nqubits)
    or_circuit.add(
        oracle_circuit.on_qubits(
            *(
                list(range(nqubits))
                + [oracle_circuit.nqubits - 1]
                + list(range(nqubits, oracle_circuit.nqubits - 1))
            )
        )
    )

    grover = Grover(
        or_circuit,
        superposition_circuit=superposition,
        superposition_qubits=nqubits,
        number_solutions=1,
        superposition_size=int(binomial(nqubits, num_1)),
    )

    solution, iterations = grover()

    print("The solution is", solution)
    print("Number of iterations needed:", iterations)

    return solution, iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=10, type=int, help="Number of qubits.")
    parser.add_argument("--num_1", default=2, type=int, help="Number of 1's to find.")
    args = vars(parser.parse_args())
    main(**args)
