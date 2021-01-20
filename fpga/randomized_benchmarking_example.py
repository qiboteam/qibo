import copy
import numpy as np
import math
#import tensorflow as tf
from qibo.gates import RX, RY, I, M
from qibo.models import Circuit
from qibo.config import raise_error

X = lambda q: RX(q, np.pi)
Y = lambda q: RY(q, np.pi)
X_minus = lambda q: RX(q, -1 * np.pi)
Y_minus = lambda q: RY(q, -1 * np.pi)
X_half = lambda q: RX(q, np.pi / 2)
Y_half = lambda q: RY(q, np.pi / 2)
X_half_minus = lambda q: RX(q, -1 * np.pi / 2)
Y_half_minus = lambda q: RY(q, -1 * np.pi / 2)

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

def randomized_benchmark(q: int, gates: int) -> Circuit:
    """Randomized benchmarking of single qubit operations
    Follows the approach in https://arxiv.org/abs/1009.3639 but uses a reduced set of gates, restricted to the XY plane and only dealing with rotations

    Args:
        q (int): Qubit ID to test
        gates (int): Maximum number of gate sweep to test

    Returns: List of Circuit objects representing random sequence of gates

    """
    circuits = []
    identity = np.array([[1, 0], [0, 1]])

    for num_gates in range(1, gates + 1):
        s = sum([rotation_group[idx] for idx in np.random.randint(0, len(rotation_group), num_gates)], [])
        s = [g(q) for g in s]
        initial = identity
        for g in s:
            initial = np.matmul(g.unitary, initial)

        # We want to find the n + 1 gate that inverts the entire sequence
        for group in rotation_group:
            gates = [g(q) for g in group]
            res = initial
            for g in gates:
                res = np.matmul(g.unitary, res)
            if np.allclose(res, identity):
                gates = s + gates
                break

        else:
            raise_error(RuntimeError, "Unable to locate inverse gate")

        c = Circuit(q + 1) # Hack to select which qubit we want to check
        c.add(gates)
        circuits.append(c)

    return circuits
