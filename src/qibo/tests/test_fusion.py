"""
Test functions for gate fusion.
"""
import numpy as np
import pytest
from qibo.models import Circuit
from qibo import gates
from qibo.tensorflow import fusion


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
