"""
Testing parallel evaluations.
"""

import sys

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.models import QFT
from qibo.parallel import (
    parallel_circuits_execution,
    parallel_execution,
    parallel_parametrized_execution,
)
from qibo.quantum_info.random_ensembles import random_statevector


@pytest.mark.skipif(sys.platform == "darwin", reason="Mac tests")
def test_parallel_states_evaluation(backend):
    """Evaluate circuit for multiple input states."""
    nqubits = 10
    backend.set_seed(0)
    circuit = QFT(nqubits)

    states = [
        random_statevector(2**nqubits, dtype=backend.complex128, backend=backend)
        for _ in range(5)
    ]

    r1 = []
    for state in states:
        r1.append(backend.execute_circuit(circuit, state))

    r2 = parallel_execution(circuit, states=states, processes=2, backend=backend)
    r1 = [x.state() for x in r1]
    r2 = [x.state() for x in r2]
    backend.assert_allclose(r1, r2)


@pytest.mark.skipif(sys.platform == "darwin", reason="Mac tests")
@pytest.mark.parametrize("use_execute_circuits", [False, True])
def test_parallel_circuit_evaluation(backend, use_execute_circuits):
    """Evaluate multiple circuits in parallel."""
    circuits = [QFT(n) for n in range(1, 11)]

    r1 = []
    for circuit in circuits:
        r1.append(backend.execute_circuit(circuit))

    if use_execute_circuits:
        r2 = backend.execute_circuits(circuits, processes=2)
    else:
        r2 = parallel_circuits_execution(circuits, processes=2, backend=backend)

    for x, y in zip(r1, r2):
        target = x.state(numpy=True)
        final = y.state(numpy=True)
        backend.assert_allclose(final, target)


@pytest.mark.skipif(sys.platform == "darwin", reason="Mac tests")
def test_parallel_circuit_states_evaluation(backend):
    """Evaluate multiple circuits in parallel with different initial states."""
    circuits = [QFT(nqubits) for nqubits in range(1, 11)]
    states = [
        random_statevector(2**nqubits, backend=backend) for nqubits in range(1, 11)
    ]

    with pytest.raises(TypeError):
        r2 = parallel_circuits_execution(
            circuits, states=1, processes=2, backend=backend
        )
    with pytest.raises(ValueError):
        r2 = parallel_circuits_execution(
            circuits, states[:3], processes=2, backend=backend
        )

    r1 = []
    for circuit, state in zip(circuits, states):
        r1.append(backend.execute_circuit(circuit, state))

    r2 = parallel_circuits_execution(circuits, states, processes=2, backend=backend)
    for x, y in zip(r1, r2):
        target = x.state(numpy=True)
        final = y.state(numpy=True)
        backend.assert_allclose(final, target)


@pytest.mark.skipif(sys.platform == "darwin", reason="Mac tests")
def test_parallel_parametrized_circuit(backend):
    """Evaluate circuit for multiple parameters."""
    nqubits = 5
    nlayers = 10
    circuit = Circuit(nqubits)
    for l in range(nlayers):
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(0, nqubits - 1, 2))
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(1, nqubits - 2, 2))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add(gates.RY(q, theta=0) for q in range(nqubits))

    size = len(circuit.get_parameters())
    np.random.seed(0)
    parameters = [np.random.uniform(0, 2 * np.pi, size) for i in range(10)]
    state = np.random.random(2**nqubits)

    r1 = []
    for params in parameters:
        circuit.set_parameters(params)
        r1.append(backend.execute_circuit(circuit, backend.cast(state)))

    r2 = parallel_parametrized_execution(
        circuit,
        parameters=parameters,
        initial_state=state,
        processes=2,
        backend=backend,
    )
    r1 = [x.state() for x in r1]
    r2 = [x.state() for x in r2]
    backend.assert_allclose(r1, r2)
