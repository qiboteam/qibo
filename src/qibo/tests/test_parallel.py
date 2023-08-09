"""
Testing parallel evaluations.
"""
import sys
import numpy as np
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit, QFT
from qibo.parallel import parallel_parametrized_execution, parallel_execution


def is_parallel_supported(backend_name):  # pragma: no cover
    if "GPU" in qibo.get_device():
        return False
    if backend_name in ("tensorflow", "qibojit"):
        return False
    if sys.platform in ("darwin", "win32"):
        return False
    return True


def test_parallel_circuit_evaluation(backend, skip_parallel):  # pragma: no cover
    """Evaluate circuit for multiple input states."""
    device = qibo.get_device()
    backend_name = qibo.get_backend()
    if skip_parallel:
        pytest.skip("Skipping parallel test.")
    if not is_parallel_supported(backend_name):
        pytest.skip("Skipping parallel test due to unsupported configuration.")
    original_threads = qibo.get_threads()
    qibo.set_threads(1)

    nqubits = 10
    np.random.seed(0)
    c = QFT(nqubits)

    states = [np.random.random(2**nqubits) for i in range(5)]

    r1 = []
    for state in states:
        r1.append(c(state))

    r2 = parallel_execution(c, states=states, processes=2)
    np.testing.assert_allclose(r1, r2)
    qibo.set_threads(original_threads)


def test_parallel_parametrized_circuit(backend, skip_parallel):  # pragma: no cover
    """Evaluate circuit for multiple parameters."""
    device = qibo.get_device()
    backend_name = qibo.get_backend()
    if skip_parallel:
        pytest.skip("Skipping parallel test.")
    if not is_parallel_supported(backend_name):
        pytest.skip("Skipping parallel test due to unsupported configuration.")
    original_threads = qibo.get_threads()
    qibo.set_threads(1)

    nqubits = 5
    nlayers  = 10
    c = Circuit(nqubits)
    for l in range(nlayers):
        c.add((gates.RY(q, theta=0) for q in range(nqubits)))
        c.add((gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
        c.add((gates.RY(q, theta=0) for q in range(nqubits)))
        c.add((gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
        c.add(gates.CZ(0, nqubits-1))
    c.add((gates.RY(q, theta=0) for q in range(nqubits)))

    size = len(c.get_parameters())
    np.random.seed(0)
    parameters = [np.random.uniform(0, 2*np.pi, size) for i in range(10)]
    state = None

    r1 = []
    for params in parameters:
        c.set_parameters(params)
        r1.append(c(state))

    r2 = parallel_parametrized_execution(c, parameters=parameters, initial_state=state, processes=2)
    np.testing.assert_allclose(r1, r2)
    qibo.set_threads(original_threads)
