import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import CliffordBackend
from qibo.quantum_info import Clifford, random_clifford


def test_clifford_run():
    c = random_clifford(3)
    c.add(gates.M(*np.random.choice(3, size=2, replace=False)))
    backend = CliffordBackend()
    result = backend.execute_circuit(c)
    obj = Clifford.run(c)
    backend.assert_allclose(obj.state(), result.state())
    backend.assert_allclose(obj.probabilities(), result.probabilities())


def test_clifford_get_stabilizers():
    c = Circuit(3)
    c.add(gates.X(2))
    c.add(gates.H(0))
    obj = Clifford.run(c)
    true_generators, true_phases = ["XII", "IZI", "IIZ"], [1, 1, -1]
    generators, phases = obj.get_stabilizers_generators()
    assert true_generators == generators
    assert true_phases == phases.tolist()
    true_stabilizers = [
        "-(X)(Z)(Z)",
        "(X)(Z)(I)",
        "-(X)(I)(Z)",
        "(X)(I)(I)",
        "-(I)(Z)(Z)",
        "(I)(Z)(I)",
        "-(I)(I)(Z)",
        "(I)(I)(I)",
    ]
    stabilizers = obj.get_stabilizers()
    assert true_stabilizers == stabilizers


def test_clifford_get_destabilizers():
    c = Circuit(3)
    c.add(gates.X(2))
    c.add(gates.H(0))
    obj = Clifford.run(c)
    true_generators, true_phases = ["ZII", "IXI", "IIX"], [1, 1, 1]
    generators, phases = obj.get_destabilizers_generators()
    assert true_generators == generators
    assert true_phases == phases.tolist()
    true_destabilizers = [
        "(Z)(X)(X)",
        "(Z)(X)(I)",
        "(Z)(I)(X)",
        "(Z)(I)(I)",
        "(I)(X)(X)",
        "(I)(X)(I)",
        "(I)(I)(X)",
        "(I)(I)(I)",
    ]
    destabilizers = obj.get_destabilizers()
    assert true_destabilizers == destabilizers
