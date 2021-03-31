import argparse
import numpy as np
import matplotlib.pyplot as plt
import qibo
from qibo import K
from qibo.config import log, raise_error
from qibo.hardware import gates, circuit
from qibo.hardware.scheduler import TaskScheduler
qibo.set_backend("icarusq")


parser = argparse.ArgumentParser()
parser.add_argument("--qubit", default=0, type=int)
parser.add_argument("--ngates", default=10, type=int)
parser.add_argument("--nshots", default=1000, type=int)
parser.add_argument("--address", default=None, type=str)
parser.add_argument("--username", default=None, type=str)
parser.add_argument("--password", default=None, type=str)


I = lambda q: gates.I(q)
X = lambda q: gates.RX(q, np.pi)
Y = lambda q: gates.RY(q, np.pi)
X_minus = lambda q: gates.RX(q, -1 * np.pi)
Y_minus = lambda q: gates.RY(q, -1 * np.pi)
X_half = lambda q: gates.RX(q, np.pi / 2)
Y_half = lambda q: gates.RY(q, np.pi / 2)
X_half_minus = lambda q: gates.RX(q, -1 * np.pi / 2)
Y_half_minus = lambda q: gates.RY(q, -1 * np.pi / 2)

""" clifford_group = [
    [I],
    [X],
    [X_minus],
    [Y],
    [Y_minus],
    [X, Y],
    [Y_minus, X_minus],
    [X_half],
    [X_half_minus],
    [Y_half],
    [Y_half_minus],
    [X_half, Y_half, X_half_minus],
    [X_half, Y_half_minus, X_half_minus],
    [Y_half_minus, X],
    [X_minus, Y_half],
    [Y_half, X],
    [X_minus, Y_half_minus],
    [X_half, Y],
    [Y_minus, X_half_minus],
    [X_half_minus, Y],
    [Y_minus, X_half],
    [X_half, Y_half, X_half],
    [X_half_minus, Y_half_minus, X_half_minus],
    [X_half_minus, Y_half, X_half_minus],
    [X_half, Y_half_minus, X_half],
    [X_half, Y_half],
    [X_half_minus, Y_half],
    [X_half, Y_half_minus],
    [X_half_minus, Y_half_minus],
    [Y_half_minus, X_half_minus],
    [Y_half_minus, X_half],
    [Y_half, X_half_minus],
    [Y_half, X_half]
]
 """

rotation_group = [
    [I],
    [X, X], # negative I
    [X],
    [Y],
    [X, Y],
    [Y, X],
    [X_minus],
    [Y_minus],
    [X_minus, Y],
    [Y_minus, X]
]

def randomized_benchmark(q, ngates, scheduler=None):
    """Randomized benchmarking of single qubit operations.

    Follows the approach in https://arxiv.org/abs/1009.3639 but uses a reduced
    set of gates, restricted to the XY plane and only dealing with rotations.

    Args:
        q (int): Qubit ID to test.
        gates (int): Maximum number of gate sweep to test.

    Returns:
        List of Circuit objects representing random sequence of gates.
    """
    circuits = []
    for num_gates in range(1, ngates + 1):
        initial_gates = []
        for idx in np.random.randint(0, len(rotation_group), num_gates):
            initial_gates.extend((gate(q) for gate in rotation_group[idx]))

        # Calculate the total unitary matrix that corresponds to `initial_gates`
        initial = np.eye(2)
        for gate in initial_gates:
            initial = np.matmul(gate.unitary, initial)

        # We want to find the n + 1 gate that inverts the entire sequence
        c = None
        for group in rotation_group:
            inverse_gates = [gate(q) for gate in group]
            res = np.copy(initial)
            for gate in inverse_gates:
                res = np.matmul(gate.unitary, res)
            if np.allclose(res, np.eye(2)):
                c = circuit.Circuit(2, scheduler)
                c.add(initial_gates)
                c.add(inverse_gates)
                break

        if c is None:
            raise_error(RuntimeError, "Unable to locate inverse gate")
        else:
            circuits.append(c)

    return circuits


def main(qubit, ngates, nshots, address, username, password):
    scheduler = TaskScheduler()
    if address is not None:
        K.experiment.connect(address, username, password)
    else:
        # set hard=coded calibration data
        scheduler._qubit_config = K.experiment.static.calibration_placeholder

    circuits = randomized_benchmark(qubit, ngates, scheduler)

    results = []
    for circuit in circuits:
        circuit(nshots)
        results.append(circuit.parse_result(qubit))

    sweep = range(1, ngates + 1)
    plt.plot(sweep, results)
    plt.show()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
