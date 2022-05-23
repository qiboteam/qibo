import pytest
import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibo.tests.utils import random_state


@pytest.mark.parametrize("nqubits", [5, 6])
def test_variational_layer(backend, nqubits):
    theta = 2 * np.pi * np.random.random(nqubits)

    targetc = Circuit(nqubits)
    targetc.add(gates.RY(i, t) for i, t in enumerate(theta))
    targetc.add(gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2))

    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    c = Circuit(nqubits)
    c.add(gates.VariationalLayer(range(nqubits), pairs, gates.RY, gates.CZ, theta))

    backend.assert_circuitclose(c, targetc)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_variational_layer_dagger(backend, nqubits):
    theta = 2 * np.pi * np.random.random((2, nqubits))
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    gate = gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ,
                                  theta[0], theta[1])
    c = Circuit(nqubits)
    c.add((gate, gate.dagger()))
    initial_state = random_state(nqubits)
    final_state = backend.execute_circuit(c, initial_state=np.copy(initial_state))
    backend.assert_allclose(final_state, initial_state)


@pytest.mark.skip
def test_flatten(backend):
    target_state = np.ones(4) / 2.0
    final_state = apply_gates(backend, [gates.Flatten(target_state)], nqubits=2)
    backend.assert_allclose(final_state, target_state)

    target_state = np.ones(4) / 2.0
    gate = gates.Flatten(target_state)
    with pytest.raises(ValueError):
        gate._construct_unitary()


@pytest.mark.skip
def test_callback_gate_errors():
    from qibo import callbacks
    entropy = callbacks.EntanglementEntropy([0])
    gate = gates.CallbackGate(entropy)
    with pytest.raises(ValueError):
        gate._construct_unitary()


@pytest.mark.parametrize("nqubits", [2, 3])
def test_fused_gate_construct_unitary(backend, nqubits):
    gate = gates.FusedGate(0, 1)
    gate.append(gates.H(0))
    gate.append(gates.H(1))
    gate.append(gates.CZ(0, 1))
    hmatrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    czmatrix = np.diag([1, 1, 1, -1])
    target_matrix = czmatrix @ np.kron(hmatrix, hmatrix)
    if nqubits > 2:
        gate.append(gates.TOFFOLI(0, 1, 2))
        toffoli = np.eye(8)
        toffoli[-2:, -2:] = np.array([[0, 1], [1, 0]])
        target_matrix = toffoli @ np.kron(target_matrix, np.eye(2))
    backend.assert_allclose(backend.asmatrix(gate), target_matrix)
