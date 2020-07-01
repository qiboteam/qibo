"""
Test functions for gate fusion.
"""
import numpy as np
import pytest
import qibo
from qibo.models import Circuit
from qibo import gates
from qibo.tensorflow import fusion


@pytest.mark.parametrize("backend", ["custom", "matmuleinsum"])
def test_one_qubit_gate_multiplication(backend):
    """Check gate multiplication for one-qubit gates."""
    qibo.set_backend(backend)
    gate1 = gates.X(0)
    gate2 = gates.H(0)
    final_gate = gate1 @ gate2
    target_matrix = (np.array([[0, 1], [1, 0]]) @
                     np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    np.testing.assert_allclose(final_gate.unitary, target_matrix)

    final_gate = gate2 @ gate1
    target_matrix = (np.array([[1, 1], [1, -1]]) / np.sqrt(2) @
                     np.array([[0, 1], [1, 0]]))
    np.testing.assert_allclose(final_gate.unitary, target_matrix)


@pytest.mark.parametrize("backend", ["custom", "matmuleinsum"])
def test_two_qubit_gate_multiplication(backend):
    """Check gate multiplication for two-qubit gates."""
    qibo.set_backend(backend)
    theta, phi = 0.1234, 0.5432
    gate1 = gates.fSim(0, 1, theta=theta, phi=phi)
    gate2 = gates.SWAP(0, 1)
    final_gate = gate1 @ gate2
    target_matrix = (np.array([[1, 0, 0, 0],
                               [0, np.cos(theta), -1j * np.sin(theta), 0],
                               [0, -1j * np.sin(theta), np.cos(theta), 0],
                               [0, 0, 0, np.exp(-1j * phi)]]) @
                     np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                               [0, 1, 0, 0], [0, 0, 0, 1]]))
    np.testing.assert_allclose(final_gate.unitary, target_matrix)

    # Check that error is raised when target qubits do not agree
    with pytest.raises(NotImplementedError):
        final_gate = gate1 @ gates.SWAP(0, 2)
    # Reset backend for other tests
    qibo.set_backend("custom")

def test_fuse_queue_single_group():
    """Check fusion that creates a single ``FusionGroup``."""
    queue = [gates.H(0), gates.X(1), gates.CZ(0, 1)]
    fused_groups = fusion.fuse_queue(queue)
    assert len(fused_groups) == 1
    group = fused_groups[0]
    assert group.gates0 == [[queue[0]], []]
    assert group.gates1 == [[queue[1]], []]
    assert group.two_qubit_gates == [(queue[2], False)]


def test_fuse_queue_two_groups():
    """Check fusion that creates two ``FusionGroup``s."""
    queue = [gates.X(0), gates.H(1), gates.CNOT(1, 2), gates.H(2), gates.Y(1),
             gates.H(0)]
    fused_groups = fusion.fuse_queue(queue)
    assert len(fused_groups) == 2
    group1, group2 = fused_groups
    assert group1.gates0 == [[queue[0], queue[5]]]
    assert group1.gates1 == [[queue[1]]]
    assert group1.two_qubit_gates == []
    assert group2.gates0 == [[], [queue[4]]]
    assert group2.gates1 == [[], [queue[3]]]
    assert group2.two_qubit_gates == [(queue[2], False)]


# TODO: Do this test for odd ``nqubits``
def test_fuse_queue_variational_layer(nqubits=6):
    """Check fusion for common type variational circuit."""
    theta = np.pi * np.random.random((2, nqubits))
    queue0 = [gates.RY(i, theta[0, i]) for i in range(nqubits)]
    queue1 = [gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)]
    queue2 = [gates.RY(i, theta[1, i]) for i in range(nqubits)]
    queue3 = [gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)]
    queue3.append(gates.CZ(0, nqubits - 1))

    fused_groups = fusion.fuse_queue(queue0 + queue1 + queue2 + queue3)
    assert len(fused_groups) == 2 * (nqubits // 2)
    for i, group in enumerate(fused_groups[:nqubits // 2]):
        assert group.gates0 == [[queue0[2 * i]], [queue2[2 * i]]]
        assert group.gates1 == [[queue0[2 * i + 1]], [queue2[2 * i + 1]]]
        assert group.two_qubit_gates == [(queue1[i], False)]
    # FIXME: Fix the ``nqubits`` odd case
    for i, group in enumerate(fused_groups[nqubits // 2:]):
        assert group.gates0 == [[], []]
        assert group.gates1 == [[], []]
        assert group.two_qubit_gates == [(queue3[i], False)]
