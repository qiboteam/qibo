"""Test functions defined in `qibo/core/distcircuit.py`."""
import pytest
import numpy as np
from qibo import gates
from qibo.core.distcircuit import DistributedCircuit
from qibo.core.distutils import DistributedQubits
from qibo.tests.test_core_distcircuit import check_device_queues


def test_transform_queue_simple(backend):
    devices = {"/GPU:0": 1, "/GPU:1": 1}
    c = DistributedCircuit(4, devices)
    c.add((gates.H(i) for i in range(4)))
    c.queues.qubits = DistributedQubits([0], c.nqubits)
    tqueue = c.queues.transform(c.queue)
    assert len(tqueue) == 6
    for i in range(3):
        assert isinstance(tqueue[i], gates.H)
        assert tqueue[i].target_qubits == (i + 1,)
    assert isinstance(tqueue[3], gates.SWAP)
    assert tqueue[3].target_qubits == (0, 1)
    assert isinstance(tqueue[4], gates.H)
    assert tqueue[4].target_qubits == (1,)
    assert isinstance(tqueue[5], gates.SWAP)
    assert tqueue[5].target_qubits == (0, 1)


def test_transform_queue_more_gates(backend):
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(4, devices)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(2, 3))
    c.add(gates.CZ(0, 1))
    c.add(gates.CNOT(3, 0))
    c.add(gates.CNOT(1, 2))
    c.queues.qubits = DistributedQubits([2, 3], c.nqubits)
    tqueue = c.queues.transform(c.queue)

    assert len(tqueue) == 10
    assert isinstance(tqueue[0], gates.H)
    assert tqueue[0].target_qubits == (0,)
    assert isinstance(tqueue[1], gates.H)
    assert tqueue[1].target_qubits == (1,)
    assert isinstance(tqueue[2], gates.CZ)
    assert tqueue[2].target_qubits == (1,)
    assert isinstance(tqueue[3], gates.SWAP)
    assert set(tqueue[3].target_qubits) == {1, 3}
    assert isinstance(tqueue[4], gates.CNOT)
    assert tqueue[4].target_qubits == (1,)
    assert isinstance(tqueue[5], gates.CNOT)
    assert tqueue[5].target_qubits == (0,)
    assert isinstance(tqueue[6], gates.SWAP)
    assert set(tqueue[6].target_qubits) == {0, 2}
    assert isinstance(tqueue[7], gates.CNOT)
    assert tqueue[7].target_qubits == (0,)
    assert isinstance(tqueue[8], gates.SWAP)
    assert set(tqueue[8].target_qubits) == {0, 2}
    assert isinstance(tqueue[9], gates.SWAP)
    assert set(tqueue[9].target_qubits) == {1, 3}


def test_create_queue_with_global_swap(backend):
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(6, devices)
    c.add([gates.H(0), gates.H(2), gates.H(3)])
    c.add(gates.SWAP(3, 4))
    c.add([gates.X(1), gates.X(2)])
    c.queues.qubits = DistributedQubits([4, 5], c.nqubits)
    c.queues.create(c.queues.transform(c.queue))

    check_device_queues(c.queues)
    assert len(c.queues.special_queue) == 1
    assert len(c.queues.queues) == 3
    assert len(c.queues.queues[0]) == 4
    assert len(c.queues.queues[1]) == 0
    assert len(c.queues.queues[2]) == 4
    for device_group in c.queues.queues[0]:
        assert len(device_group) == 3
    for device_group in c.queues.queues[2]:
        assert len(device_group) == 2


def test_create_queue_errors():
    c = DistributedCircuit(4, {"/GPU:0": 2})
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.queues.qubits = DistributedQubits([0], c.nqubits)
    with pytest.raises(ValueError):
        c.queues.create(c.queue)

    c = DistributedCircuit(4, {"/GPU:0": 4})
    c.add(gates.SWAP(0, 1))
    c.queues.qubits = DistributedQubits([0, 1], c.nqubits)
    with pytest.raises(ValueError):
        c.queues.create(c.queue)
