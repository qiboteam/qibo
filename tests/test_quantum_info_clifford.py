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


@pytest.mark.parametrize("binary", [True, False])
def test_clifford_samples(binary):
    c = random_clifford(5)
    c.add(gates.M(3, register_name="3"))
    c.add(gates.M(0, 1, register_name="01"))
    obj = Clifford.run(c, nshots=50)
    samples = obj.samples(binary=binary, registers=True)
    assert "01" in samples and "3" in samples
    shapes = [(50, 2), (50, 1)] if binary else [(50,), (50,)]
    assert samples["01"].shape == shapes[0] and samples["3"].shape == shapes[1]


def test_clifford_samples_error():
    c = random_clifford(1)
    obj = Clifford.run(c, nshots=50)
    with pytest.raises(RuntimeError) as excinfo:
        Clifford.run(c).samples()
        assert str(excinfo.value) == "The circuit does not contain any measurement."
