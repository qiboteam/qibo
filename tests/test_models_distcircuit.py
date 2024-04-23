"""Test functions defined in `qibo/models/distcircuit.py`."""

import pytest

from qibo import Circuit, gates
from qibo.models.distcircuit import DistributedQubits


def check_device_queues(queues):
    """Asserts that global qubits do not collide with the gates to be applied."""
    for gate_group in queues.queues:
        for device_gates in gate_group:
            target_qubits = set()
            for gate in device_gates:
                target_qubits |= set(gate.original_gate.target_qubits)
            assert not queues.qubits.set & target_qubits


def test_distributed_circuit_init():
    """Check if error is raised if total devices is not a power of 2."""
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = Circuit(5, devices)
    assert c.ndevices == 4
    assert c.nglobal == 2
    devices = {"/GPU:0": 2, "/GPU:1": 1}
    with pytest.raises(ValueError):
        c = Circuit(4, devices)


def test_distributed_circuit_add_gate():
    # Attempt to add gate so that available global qubits are not enough
    c = Circuit(2, {"/GPU:0": 2})
    with pytest.raises(ValueError):
        c.add(gates.SWAP(0, 1))
    # Attempt adding noise channel
    with pytest.raises(NotImplementedError):
        c.add(gates.PauliNoiseChannel(0, [("X", 0.1), ("Z", 0.1)]))


def test_distributed_circuit_various_errors():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = Circuit(5, devices)
    # Attempt to use ``.with_pauli_noise``
    with pytest.raises(NotImplementedError):
        c.with_pauli_noise(list(zip(["X", "Y", "Z"], [0.1, 0.2, 0.1])))
    # Attempt to compile
    with pytest.raises(RuntimeError):
        c.compile()
    # Attempt to access state before being set
    with pytest.raises(RuntimeError):
        final_state = c.final_state


def test_distributed_circuit_fusion(accelerators):
    c = Circuit(4, accelerators)
    c.add(gates.H(i) for i in range(4))
    with pytest.raises(NotImplementedError):
        c.fuse()


def test_distributed_circuit_set_gates():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = Circuit(6, devices)
    c.add(gates.H(i) for i in range(4))
    c.queues.set(c.queue)

    check_device_queues(c.queues)
    assert len(c.queues.queues) == 1
    assert len(c.queues.queues[0]) == 4
    for queues in c.queues.queues[0]:
        assert len(queues) == 4


def test_distributed_circuit_set_gates_controlled():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = Circuit(6, devices)
    c.add([gates.H(0), gates.H(2), gates.H(3)])
    c.add(gates.CNOT(4, 5))
    c.add(gates.Z(1).controlled_by(0))
    c.add(gates.SWAP(2, 3))
    c.add([gates.X(2), gates.X(3), gates.X(4)])
    c.queues.set(c.queue)

    check_device_queues(c.queues)
    assert len(c.queues.queues) == 7
    for i, queue in enumerate(c.queues.queues[:-2]):
        assert len(queue) == 4 * (1 - i % 2)
    for device_group in c.queues.queues[0]:
        assert len(device_group) == 7
    for device_group in c.queues.queues[2]:
        assert len(device_group) == 1


@pytest.mark.parametrize("nqubits", [28, 29, 30, 31, 32, 33, 34])
@pytest.mark.parametrize("ndevices", [2, 4, 8, 16, 32, 64])
def test_distributed_qft_global_qubits_validity(nqubits, ndevices):
    """Check that no gates are applied to global qubits for practical QFT cases."""
    from qibo.models import QFT

    c = QFT(nqubits, accelerators={"/GPU:0": ndevices})
    c.queues.set(c.queue)  # pylint: disable=E1101
    check_device_queues(c.queues)  # pylint: disable=E1101


def test_transform_queue_simple():
    devices = {"/GPU:0": 1, "/GPU:1": 1}
    c = Circuit(4, devices)
    c.add(gates.H(i) for i in range(4))
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


def test_transform_queue_more_gates():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = Circuit(4, devices)
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


def test_create_queue_with_global_swap():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = Circuit(6, devices)
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


def test_create_queue_errors(backend):
    c = Circuit(4, {"/GPU:0": 2})
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.queues.qubits = DistributedQubits([0], c.nqubits)
    with pytest.raises(ValueError):
        c.queues.create(c.queue)

    c = Circuit(4, {"/GPU:0": 4})
    c.add(gates.SWAP(0, 1))
    c.queues.qubits = DistributedQubits([0, 1], c.nqubits)
    with pytest.raises(ValueError):
        c.queues.create(c.queue)
