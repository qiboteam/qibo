"""Test :class:`qibo.models.distcircuit.DistributedCircuit` execution."""

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.quantum_info import random_statevector


@pytest.mark.parametrize("use_global_qubits", [False, True])
def test_distributed_circuit_execution(
    backend, accelerators, use_global_qubits
):  # pragma: no cover
    dist_circuit = Circuit(6, accelerators)
    circuit = Circuit(6)
    if use_global_qubits:
        dist_circuit.add(gates.H(i) for i in range(dist_circuit.nqubits))
        circuit.add(gates.H(i) for i in range(dist_circuit.nqubits))
    else:
        dist_circuit.add(gates.H(i) for i in range(dist_circuit.nlocal))
        circuit.add(gates.H(i) for i in range(dist_circuit.nlocal))
    dist_circuit.global_qubits = range(dist_circuit.nlocal, dist_circuit.nqubits)

    initial_state = random_statevector(2**circuit.nqubits, backend=backend)
    final_state = backend.execute_circuit(dist_circuit, np.copy(initial_state))
    target_state = backend.execute_circuit(circuit, np.copy(initial_state))
    backend.assert_allclose(target_state, final_state)


def test_distributed_circuit_execution_pretransformed(
    backend, accelerators
):  # pragma: no cover
    dist_circuit = Circuit(4, accelerators)
    dist_circuit.add(gates.H(i) for i in range(dist_circuit.nglobal, 4))
    dist_circuit.add(gates.SWAP(0, 2))
    dist_circuit.add(gates.H(i) for i in range(dist_circuit.nglobal, 4))

    circuit = Circuit(4)
    circuit.add(gates.H(i) for i in range(dist_circuit.nglobal, 4))
    circuit.add(gates.SWAP(0, 2))
    circuit.add(gates.H(i) for i in range(dist_circuit.nglobal, 4))

    initial_state = random_statevector(2**circuit.nqubits, backend=backend)
    final_state = backend.execute_circuit(dist_circuit, np.copy(initial_state))
    target_state = backend.execute_circuit(circuit, np.copy(initial_state))
    backend.assert_allclose(target_state, final_state, atol=1e-7)


def test_distributed_circuit_execution_with_swap(
    backend, accelerators
):  # pragma: no cover
    dist_circuit = Circuit(6, accelerators)
    dist_circuit.add(gates.H(i) for i in range(6))
    dist_circuit.add(gates.SWAP(i, i + 1) for i in range(5))
    dist_circuit.global_qubits = [0, 1]

    circuit = Circuit(6)
    circuit.add(gates.H(i) for i in range(6))
    circuit.add(gates.SWAP(i, i + 1) for i in range(5))

    initial_state = random_statevector(2**circuit.nqubits, backend=backend)
    final_state = backend.execute_circuit(dist_circuit, np.copy(initial_state))
    target_state = backend.execute_circuit(circuit, np.copy(initial_state))
    backend.assert_allclose(target_state, final_state, atol=1e-7)


def test_distributed_circuit_execution_special_gate(
    backend, accelerators
):  # pragma: no cover
    from qibo import callbacks

    dist_circuit = Circuit(6, accelerators)
    initial_state = random_statevector(2**dist_circuit.nqubits, backend=backend)
    entropy = callbacks.EntanglementEntropy([0])
    dist_circuit.add(gates.CallbackGate(entropy))
    dist_circuit.add(gates.H(i) for i in range(dist_circuit.nlocal))
    dist_circuit.add(gates.CallbackGate(entropy))
    dist_circuit.global_qubits = range(dist_circuit.nlocal, dist_circuit.nqubits)

    circuit = Circuit(6)
    circuit.add(gates.CallbackGate(entropy))
    circuit.add(gates.H(i) for i in range(dist_circuit.nlocal))
    circuit.add(gates.CallbackGate(entropy))
    final_state = backend.execute_circuit(
        dist_circuit, initial_state=np.copy(initial_state)
    )
    target_state = backend.execute_circuit(
        circuit, initial_state=np.copy(initial_state)
    )
    backend.assert_allclose(final_state, target_state, atol=1e-7)


def test_distributed_circuit_execution_controlled_gate(
    backend, accelerators
):  # pragma: no cover
    dist_circuit = Circuit(4, accelerators)
    dist_circuit.add(gates.H(i) for i in range(dist_circuit.nglobal, 4))
    dist_circuit.add(gates.CNOT(0, 2))

    circuit = Circuit(4)
    circuit.add(gates.H(i) for i in range(dist_circuit.nglobal, 4))
    circuit.add(gates.CNOT(0, 2))

    initial_state = random_statevector(2**circuit.nqubits, backend=backend)
    final_state = backend.execute_circuit(dist_circuit, np.copy(initial_state))
    target_state = backend.execute_circuit(circuit, np.copy(initial_state))
    backend.assert_allclose(target_state, final_state)


def test_distributed_circuit_execution_controlled_by_gates(
    backend, accelerators
):  # pragma: no cover
    dist_circuit = Circuit(6, accelerators)
    dist_circuit.add([gates.H(0), gates.H(2), gates.H(3)])
    dist_circuit.add(gates.CNOT(4, 5))
    dist_circuit.add(gates.Z(1).controlled_by(0))
    dist_circuit.add(gates.SWAP(2, 3))
    dist_circuit.add([gates.X(2), gates.X(3), gates.X(4)])

    circuit = Circuit(6)
    circuit.add([gates.H(0), gates.H(2), gates.H(3)])
    circuit.add(gates.CNOT(4, 5))
    circuit.add(gates.Z(1).controlled_by(0))
    circuit.add(gates.SWAP(2, 3))
    circuit.add([gates.X(2), gates.X(3), gates.X(4)])

    initial_state = random_statevector(2**circuit.nqubits, backend=backend)
    final_state = backend.execute_circuit(dist_circuit, np.copy(initial_state))
    target_state = backend.execute_circuit(circuit, np.copy(initial_state))
    backend.assert_allclose(target_state, final_state, atol=1e-7)


def test_distributed_circuit_execution_addition(
    backend, accelerators
):  # pragma: no cover
    # Attempt to add circuits with different devices
    c1 = Circuit(6, {"/GPU:0": 2, "/GPU:1": 2})
    c2 = Circuit(6, {"/GPU:0": 2})
    with pytest.raises(ValueError):
        c = c1 + c2

    c1 = Circuit(6, accelerators)
    c2 = Circuit(6, accelerators)
    c1.add([gates.H(i) for i in range(6)])
    c2.add([gates.CNOT(i, i + 1) for i in range(5)])
    c2.add([gates.Z(i) for i in range(6)])
    dist_circuit = c1 + c2

    circuit = Circuit(6)
    circuit.add([gates.H(i) for i in range(6)])
    circuit.add([gates.CNOT(i, i + 1) for i in range(5)])
    circuit.add([gates.Z(i) for i in range(6)])
    assert c.depth == dist_circuit.depth
    final_state = backend.execute_circuit(dist_circuit)
    target_state = backend.execute_circuit(circuit)
    backend.assert_allclose(final_state, target_state, atol=1e-7)


def test_distributed_circuit_empty_execution(backend, accelerators):  # pragma: no cover
    # test executing a circuit with the default initial state
    c = Circuit(5, accelerators)
    final_state = backend.execute_circuit(c).state()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1
    backend.assert_allclose(final_state, target_state)
    # test re-executing the circuit with a given initial state
    initial_state = random_statevector(2**c.nqubits, backend=backend)
    final_state = backend.execute_circuit(c, initial_state)
    backend.assert_allclose(final_state, initial_state)
    # test executing a new circuit with a given initial state
    c = Circuit(5, accelerators)
    initial_state = random_statevector(2**c.nqubits, backend=backend)
    final_state = backend.execute_circuit(c, initial_state)
    backend.assert_allclose(final_state, initial_state)
    # test re-executing the circuit with the default initial state
    final_state = backend.execute_circuit(c)
    backend.assert_allclose(final_state, target_state)
