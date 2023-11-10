from itertools import product

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import CliffordBackend, NumpyBackend
from qibo.quantum_info import Clifford

clifford_bkd = CliffordBackend()
numpy_bkd = NumpyBackend()

thetas = [
    th + 2 * i * np.pi for i in range(2) for th in [0, np.pi / 2, np.pi, 3 * np.pi / 2]
]
axes = ["RX", "RY", "RZ"]


@pytest.mark.parametrize("axis,theta", list(product(axes, thetas)))
def test_rotations(theta, axis):
    c = Circuit(1, density_matrix=True)
    c.add(getattr(gates, axis)(0, theta=theta))
    c.add(gates.M(0))
    clifford_state = Clifford(clifford_bkd.execute_circuit(c)).state()
    numpy_state = numpy_bkd.execute_circuit(c).state()
    print(clifford_state)
    print(numpy_state)
    numpy_bkd.assert_allclose(clifford_state, numpy_state, atol=1e-8)
