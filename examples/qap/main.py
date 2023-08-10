"""Quadratic Assignment Problem"""

import argparse

import numpy as np
from qap import (
    hamiltonian_qap,
    qubo_qap,
    qubo_qap_energy,
    qubo_qap_feasibility,
    qubo_qap_penalty,
)
from qubo_utils import binary2spin, spin2QiboHamiltonian

parser = argparse.ArgumentParser()
parser.add_argument("--filename", default="./tiny04a.dat", type=str)


def load_qap(filename):
    """Load qap problem from a file

    The file format is compatible with the one used in QAPLIB

    """

    with open(filename) as fh:
        n = int(fh.readline())

        numbers = [float(n) for n in fh.read().split()]

        data = np.asarray(numbers).reshape(2, n, n)
        f = data[1]
        d = data[0]

    i = range(len(f))
    f[i, i] = 0
    d[i, i] = 0

    return f, d


def main(filename: str = "./tiny04a.dat"):
    print(f"Load flow and distance matrices from {filename} and make a QUBO")
    F, D = load_qap(filename)
    penalty = qubo_qap_penalty((F, D))

    linear, quadratic, offset = qubo_qap((F, D), penalty=penalty)

    print("A random solution with seed 1234 must be infeasible")
    import numpy as np

    rng = np.random.default_rng(seed=1234)
    random_solution = {i: rng.integers(2) for i in range(F.size)}
    feasibility = qubo_qap_feasibility((F, D), random_solution)
    assert not feasibility, "The random solution should be infeasible."

    print("Generate a feasible solution and check its feasibility")
    feasible_solution = np.zeros(F.shape)
    sequence = np.arange(F.shape[0])
    np.random.shuffle(sequence)
    for i in range(F.shape[0]):
        feasible_solution[i, sequence[i]] = 1
    feasible_solution = {k: v for k, v in enumerate(feasible_solution.flatten())}
    feasibility = qubo_qap_feasibility((F, D), feasible_solution)
    assert feasibility, "The fixed solution should be feasible."

    print("Calculate the energy of the solution")
    energy = qubo_qap_energy((F, D), feasible_solution)

    print("Construct a hamiltonian directly from flow and distance matrices")
    ham = hamiltonian_qap((F, D), dense=False)

    print("done.")


if __name__ == "__main__":
    # by defualt, test on the mvc.csv in the same directory
    args = parser.parse_args()
    main(args.filename)
