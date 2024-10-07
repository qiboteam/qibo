import random

import numpy as np
import pytest

from qibo import Circuit, gates, get_backend
from qibo.backends import MetaBackend, NumpyBackend, set_backend
from qibo.quantum_info import random_clifford, random_density_matrix, random_statevector

numpy_bkd = NumpyBackend()


@pytest.mark.parametrize("density_matrix", [True, False])
@pytest.mark.parametrize("with_measurements", [True, False])
def test_qulacs(density_matrix, with_measurements):
    c = random_clifford(3, backend=numpy_bkd, density_matrix=density_matrix)
    if with_measurements:
        measured_qubits = random.sample([0, 1, 2], 2)
        c.add(gates.M(*measured_qubits))
    qulacs_bkd = MetaBackend.load("qulacs")
    nshots = 1000
    qulacs_res = qulacs_bkd.execute_circuit(c, nshots=nshots)
    numpy_res = numpy_bkd.execute_circuit(c, nshots=nshots)
    numpy_bkd.assert_allclose(numpy_res.probabilities(), qulacs_res.probabilities())
    if with_measurements:
        numpy_freq = numpy_res.frequencies(binary=True)
        qulacs_freq = qulacs_res.frequencies(binary=True)
        numpy_freq = [numpy_freq.get(state, 0) / nshots for state in range(8)]
        qulacs_freq = [qulacs_freq.get(state, 0) / nshots for state in range(8)]
        numpy_bkd.assert_allclose(numpy_freq, qulacs_freq, atol=1e-1)


def test_initial_state_error():
    c = Circuit(1)
    qulacs_bkd = MetaBackend.load("qulacs")
    initial_state = np.array([0.0, 1.0])
    with pytest.raises(NotImplementedError):
        qulacs_bkd.execute_circuit(c, initial_state=initial_state)


def test_set_backend():
    set_backend("qulacs")
    assert get_backend().name == "qulacs"
