from qibo import gates
from qibo.models.grover import Grover
from qibo.models import Circuit
import argparse

def main(qubits):
    """Create an oracle, find state |11...11> for a number of qubits
    Args:
        qubits (int): number of qubits

    Returns:
        solution (str): found string
        iterations (int): number of iterations needed
    """
    superposition = Circuit(qubits)
    superposition.add([gates.H(i) for i in range(qubits)])


    oracle = Circuit(qubits + 1)
    oracle.add(gates.X(qubits + 1).controlled_by(*range(qubits)))
    # Create superoposition circuit: Full superposition over the selected number qubits.

    # Generate and execute Grover class
    grover = Grover(oracle, superposition_circuit=superposition, number_solutions=1)

    solution, iterations = grover()

    print('The solution is', solution)
    print('Number of iterations needed:', iterations)

    return solution, iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", default=5, type=int)
    args = vars(parser.parse_args())
    main(**args)

