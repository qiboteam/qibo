from qibo import gates, Circuit
import numpy as np
from copy import deepcopy

sigma_0 = np.array([[1, 0], [0, 1]])
sigma_1 = np.array([[0, 1], [1, 0]])
sigma_2 = np.array([[0, -1j], [1j, 0]])
sigma_3 = np.array([[1, 0], [0, -1]])

def single_qubit_tomography(c):
    c1 = deepcopy(c)
    c1.add(gates.H(0))
    c1.add(gates.M(0))
    p1 = c1().probabilities([0])
    S1 = p1[0]-p1[1]

    c2 = deepcopy(c)
    c2.add(gates.SDG(0))
    c2.add(gates.H(0))
    c2.add(gates.M(0))
    p2 = c2().probabilities([0])
    S2 = p2[0]-p2[1]

    c3 = deepcopy(c)
    c3.add(gates.M(0))
    p3 = c3().probabilities([0])
    S3 = p3[0]-p3[1]

    return (1/2)*(1/2 * sigma_0 + S1 * sigma_1 + S2 * sigma_2 + S3 * sigma_3)