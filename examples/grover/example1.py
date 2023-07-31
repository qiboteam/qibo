import argparse

from qibo import Circuit, gates
from qibo.models.grover import Grover


def main(nqubits):
    """Create an oracle, find state |11...11> for a number of qubits.

    Args:
        nqubits (int): number of qubits

    Returns:
        solution (str): found string
        iterations (int): number of iterations needed
    """
    superposition = Circuit(nqubits)
    superposition.add([gates.H(i) for i in range(nqubits)])

    oracle = Circuit(nqubits + 1)
    oracle.add(gates.X(nqubits).controlled_by(*range(nqubits)))
    # Create superoposition circuit: Full superposition over the selected number qubits.

    # Generate and execute Grover class
    grover = Grover(oracle, superposition_circuit=superposition, number_solutions=1)

    solution, iterations = grover()

    print("The solution is", solution)
    print("Number of iterations needed:", iterations)

    return solution, iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", default=10, type=int, help="Number of qubits.")
    args = vars(parser.parse_args())
    main(**args)
