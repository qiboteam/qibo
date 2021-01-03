"""
Testing parallel evaluations.
"""
import numpy as np
import pytest
from qibo import get_threads, set_threads, gates, get_device
from qibo.models import Circuit, QFT
from qibo.parallel import parallel_parametrized_execution, parallel_execution


def test_parallel_circuit_evaluation():
    """Evaluate circuit for multiple input states."""
    if 'GPU' in get_device(): # pragma: no cover
        pytest.skip("unsupported configuration")
    original_threads = get_threads()
    set_threads(1)

    nqubits = 10
    np.random.seed(0)
    c = QFT(nqubits)

    states = [ np.random.random(2**nqubits) for i in range(5)]

    r1 = []
    for state in states:
        r1.append(c(state))

    r2 = parallel_execution(c, states=states, processes=2)
    np.testing.assert_allclose(r1, r2)
    set_threads(original_threads)


def test_parallel_parametrized_circuit():
    """Evaluate circuit for multiple parameters."""
    if 'GPU' in get_device(): # pragma: no cover
        pytest.skip("unsupported configuration")
    original_threads = get_threads()
    set_threads(1)

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
    parameters = [ np.random.uniform(0, 2*np.pi, size) for i in range(10) ]
    state = None

    r1 = []
    for params in parameters:
        c.set_parameters(params)
        r1.append(c(state))

    r2 = parallel_parametrized_execution(c, parameters=parameters, initial_state=state, processes=2)
    np.testing.assert_allclose(r1, r2)
    set_threads(original_threads)
