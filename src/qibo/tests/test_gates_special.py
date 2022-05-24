import pytest
import numpy as np
from qibo import gates
from qibo.models import Circuit
from qibo.tests.utils import random_state


@pytest.mark.skip
@pytest.mark.parametrize("qubit", [0, 1, 2])
def test_partial_trace_gate(backend, qubit):
    gate = gates.PartialTrace(qubit)
    gate.density_matrix = True
    initial_rho = random_density_matrix(3)
    final_state = gate(np.copy(initial_rho))

    zero_state = np.array([[1, 0], [0, 0]])
    target_state = np.reshape(initial_rho, 6 * (2,))
    if qubit == 0:
        target_state = np.einsum("aBCabc,Dd->DBCdbc", target_state, zero_state)
    elif qubit == 1:
        target_state = np.einsum("AbCabc,Dd->ADCadc", target_state, zero_state)
    elif qubit == 2:
        target_state = np.einsum("ABcabc,Dd->ABDabd", target_state, zero_state)
    target_state = np.reshape(target_state, (8, 8))
    K.assert_allclose(final_state, target_state)


@pytest.mark.skip
def test_partial_trace_gate_errors(backend):
    gate = gates.PartialTrace(0, 1)
    # attempt to create unitary matrix
    with pytest.raises(ValueError):
        gate._construct_unitary()
    # attempt to call on state vector
    state = np.random.random(16) + 1j * np.random.random(16)
    with pytest.raises(RuntimeError):
        gate(state)


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
def test_flatten_density_matrix(backend):
    """Check ``Flatten`` gate works with density matrices."""
    target_rho = random_density_matrix(3)
    initial_rho = np.zeros(6 * (2,))
    gate = gates.Flatten(target_rho)
    final_rho = np.reshape(gate(initial_rho), (8, 8))
    backend.assert_allclose(final_rho, target_rho)


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


@pytest.mark.parametrize("nqubits", [5, 6])
def test_variational_layer_density_matrix(backend, nqubits):
    from qibo.models import Circuit
    theta = 2 * np.pi * np.random.random(nqubits)
    c = Circuit(nqubits, density_matrix=True)
    c.add((gates.RY(i, t) for i, t in enumerate(theta)))
    c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    target_state = backend.execute_circuit(c)
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    c = Circuit(nqubits, density_matrix=True)
    c.add(gates.VariationalLayer(range(nqubits), pairs,
                                  gates.RY, gates.CZ, theta))
    final_state = backend.execute_circuit(c)
    backend.assert_allclose(target_state, final_state)
