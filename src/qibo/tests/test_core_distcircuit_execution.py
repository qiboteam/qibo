"""Test functions defined in `qibo/core/distcircuit.py`."""
import pytest
import numpy as np
from qibo import K, gates
from qibo.models import Circuit
from qibo.core.distcircuit import DistributedCircuit
from qibo.tests.utils import random_state


@pytest.mark.parametrize("use_global_qubits", [False, True])
def test_distributed_circuit_execution(backend, accelerators, use_global_qubits):
    dist_c = DistributedCircuit(6, accelerators)
    c = Circuit(6)
    if use_global_qubits:
        dist_c.add((gates.H(i) for i in range(dist_c.nqubits)))
        c.add((gates.H(i) for i in range(dist_c.nqubits)))
    else:
        dist_c.add((gates.H(i) for i in range(dist_c.nlocal)))
        c.add((gates.H(i) for i in range(dist_c.nlocal)))
    dist_c.global_qubits = range(dist_c.nlocal, dist_c.nqubits)

    initial_state = random_state(c.nqubits)
    final_state = dist_c(np.copy(initial_state))
    target_state = c(np.copy(initial_state))
    K.assert_allclose(target_state, final_state)


def test_distributed_circuit_execution_pretransformed(backend, accelerators):
    dist_c = DistributedCircuit(4, accelerators)
    dist_c.add((gates.H(i) for i in range(dist_c.nglobal, 4)))
    dist_c.add(gates.SWAP(0, 2))
    dist_c.add((gates.H(i) for i in range(dist_c.nglobal, 4)))

    c = Circuit(4)
    c.add((gates.H(i) for i in range(dist_c.nglobal, 4)))
    c.add(gates.SWAP(0, 2))
    c.add((gates.H(i) for i in range(dist_c.nglobal, 4)))

    initial_state = random_state(c.nqubits)
    final_state = dist_c(np.copy(initial_state))
    target_state = c(np.copy(initial_state))
    K.assert_allclose(target_state, final_state)


def test_distributed_circuit_execution_with_swap(backend, accelerators):
    dist_c = DistributedCircuit(6, accelerators)
    dist_c.add((gates.H(i) for i in range(6)))
    dist_c.add((gates.SWAP(i, i + 1) for i in range(5)))
    dist_c.global_qubits = [0, 1]

    c = Circuit(6)
    c.add((gates.H(i) for i in range(6)))
    c.add((gates.SWAP(i, i + 1) for i in range(5)))

    initial_state = random_state(c.nqubits)
    final_state = dist_c(np.copy(initial_state))
    target_state = c(np.copy(initial_state))
    K.assert_allclose(target_state, final_state)


def test_distributed_circuit_execution_special_gate(backend, accelerators):
    dist_c = DistributedCircuit(6, accelerators)
    initial_state = random_state(dist_c.nqubits)
    dist_c.add(gates.Flatten(np.copy(initial_state)))
    dist_c.add((gates.H(i) for i in range(dist_c.nlocal)))
    dist_c.global_qubits = range(dist_c.nlocal, dist_c.nqubits)
    c = Circuit(6)
    c.add(gates.Flatten(np.copy(initial_state)))
    c.add((gates.H(i) for i in range(dist_c.nlocal)))
    K.assert_allclose(dist_c(), c())


def test_distributed_circuit_execution_controlled_gate(backend, accelerators):
    dist_c = DistributedCircuit(4, accelerators)
    dist_c.add((gates.H(i) for i in range(dist_c.nglobal, 4)))
    dist_c.add(gates.CNOT(0, 2))

    c = Circuit(4)
    c.add((gates.H(i) for i in range(dist_c.nglobal, 4)))
    c.add(gates.CNOT(0, 2))
    initial_state = random_state(c.nqubits)
    final_state = dist_c(np.copy(initial_state))
    target_state = c(np.copy(initial_state))
    K.assert_allclose(target_state, final_state)


def test_distributed_circuit_execution_controlled_by_gates(backend, accelerators):
    dist_c = DistributedCircuit(6, accelerators)
    dist_c.add([gates.H(0), gates.H(2), gates.H(3)])
    dist_c.add(gates.CNOT(4, 5))
    dist_c.add(gates.Z(1).controlled_by(0))
    dist_c.add(gates.SWAP(2, 3))
    dist_c.add([gates.X(2), gates.X(3), gates.X(4)])

    c = Circuit(6)
    c.add([gates.H(0), gates.H(2), gates.H(3)])
    c.add(gates.CNOT(4, 5))
    c.add(gates.Z(1).controlled_by(0))
    c.add(gates.SWAP(2, 3))
    c.add([gates.X(2), gates.X(3), gates.X(4)])

    initial_state = random_state(c.nqubits)
    final_state = dist_c(np.copy(initial_state))
    target_state = c(np.copy(initial_state))
    K.assert_allclose(target_state, final_state)


def test_distributed_circuit_execution_addition(backend, accelerators):
    # Attempt to add circuits with different devices
    c1 = DistributedCircuit(6, {"/GPU:0": 2, "/GPU:1": 2})
    c2 = DistributedCircuit(6, {"/GPU:0": 2})
    with pytest.raises(ValueError):
        c = c1 + c2

    c1 = DistributedCircuit(6, accelerators)
    c2 = DistributedCircuit(6, accelerators)
    c1.add([gates.H(i) for i in range(6)])
    c2.add([gates.CNOT(i, i + 1) for i in range(5)])
    c2.add([gates.Z(i) for i in range(6)])
    dist_c = c1 + c2

    c = Circuit(6)
    c.add([gates.H(i) for i in range(6)])
    c.add([gates.CNOT(i, i + 1) for i in range(5)])
    c.add([gates.Z(i) for i in range(6)])
    assert c.depth == dist_c.depth
    K.assert_allclose(dist_c(), c())


def test_distributed_circuit_empty_execution(backend, accelerators):
    # test executing a circuit with the default initial state
    c = DistributedCircuit(5, accelerators)
    final_state = c().state()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1
    K.assert_allclose(final_state, target_state)
    # test re-executing the circuit with a given initial state
    initial_state = random_state(c.nqubits)
    K.assert_allclose(c(initial_state), initial_state)
    # test executing a new circuit with a given initial state
    c = DistributedCircuit(5, accelerators)
    initial_state = random_state(c.nqubits)
    K.assert_allclose(c(initial_state), initial_state)
    # test re-executing the circuit with the default initial state
    K.assert_allclose(c(), target_state)
