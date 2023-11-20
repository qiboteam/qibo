import numpy as np
import pytest

from qibo import gates
from qibo.backends import CliffordBackend
from qibo.quantum_info import Clifford, random_clifford


def test_clifford_run():
    c = random_clifford(3)
    c.add(gates.M(*np.random.choice(3, size=2, replace=False)))
    backend = CliffordBackend()
    result = backend.execute_circuit(c)
    obj = Clifford.run(c)
    backend.assert_allclose(obj.state(), result.state())
