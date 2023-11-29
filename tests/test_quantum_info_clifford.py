from collections import Counter

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import CliffordBackend
from qibo.quantum_info import Clifford, random_clifford

BACKEND = CliffordBackend()


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
def test_clifford_samples_frequencies(binary):
    c = random_clifford(5)
    c.add(gates.M(3, register_name="3"))
    c.add(gates.M(0, 1, register_name="01"))
    obj = Clifford.run(c, nshots=50)
    samples_1 = obj.samples(binary=binary, registers=True)
    samples_2 = obj.samples(binary=binary, registers=False)
    if binary:
        BACKEND.assert_allclose(samples_2, np.hstack((samples_1["3"], samples_1["01"])))
    else:
        BACKEND.assert_allclose(
            samples_2, [s1 + 4 * s2 for s1, s2 in zip(samples_1["01"], samples_1["3"])]
        )
    freq_1 = obj.frequencies(binary=binary, registers=True)
    freq_2 = obj.frequencies(binary=binary, registers=False)

    if not binary:
        freq_1 = {
            reg: Counter({f"{k:0{len(reg)}b}": v for k, v in freq.items()})
            for reg, freq in freq_1.items()
        }
        freq_2 = Counter({f"{k:03b}": v for k, v in freq_2.items()})

    for register, counter in freq_1.items():
        for bits_1, freq in counter.items():
            tot = 0
            for bits_2, counts in freq_2.items():
                flag = bits_1 == bits_2[0] if register == "3" else bits_1 == bits_2[1:]
                if flag:
                    tot += counts
            assert tot == freq


def test_clifford_samples_error():
    c = random_clifford(1)
    obj = Clifford.run(c)
    with pytest.raises(RuntimeError) as excinfo:
        obj.samples()
        assert str(excinfo.value) == "No measurement provided."
