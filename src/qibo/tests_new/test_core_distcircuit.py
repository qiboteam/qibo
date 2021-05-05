"""Test functions defined in `qibo/core/distcircuit.py`."""
import pytest
import numpy as np
import qibo
from qibo import gates
from qibo.models import Circuit
from qibo.core.distcircuit import DistributedCircuit
from qibo.core.distutils import DistributedQubits


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
    c = DistributedCircuit(5, devices)
    assert c.ndevices == 4
    assert c.nglobal == 2
    devices = {"/GPU:0": 2, "/GPU:1": 1}
    with pytest.raises(ValueError):
        c = DistributedCircuit(4, devices)


def test_distributed_circuit_add_gate():
    # Attempt to add gate so that available global qubits are not enough
    c = DistributedCircuit(2, {"/GPU:0": 2})
    with pytest.raises(ValueError):
        c.add(gates.SWAP(0, 1))
    # Attempt adding noise channel
    with pytest.raises(NotImplementedError):
        c.add(gates.PauliNoiseChannel(0, px=0.1, pz=0.1))


def test_distributed_circuit_various_errors():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(5, devices)
    # Attempt to use ``.with_noise``
    with pytest.raises(NotImplementedError):
        noisy_c = c.with_noise((0.1, 0.2, 0.1))
    # Attempt to compile
    with pytest.raises(RuntimeError):
        c.compile()
    # Attempt to access state before being set
    with pytest.raises(RuntimeError):
        final_state = c.final_state


def test_distributed_circuit_fusion(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = DistributedCircuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    final_state = c()
    with pytest.raises(RuntimeError):
        fused_c = c.fuse()
    qibo.set_backend(original_backend)


def test_distributed_circuit_set_gates():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(6, devices)
    c.add((gates.H(i) for i in range(4)))
    c.queues.set(c.queue)

    check_device_queues(c.queues)
    assert len(c.queues.queues) == 1
    assert len(c.queues.queues[0]) == 4
    for queues in c.queues.queues[0]:
        assert len(queues) == 4

    # Attempt to set gates before adding any gate
    c = DistributedCircuit(6, devices)
    with pytest.raises(RuntimeError):
        c.queues.set(c.queue)


def test_distributed_circuit_set_gates_controlled():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(6, devices)
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


def test_distributed_circuit_get_initial_state_default(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = DistributedCircuit(6, accelerators)
    c.queues.qubits = DistributedQubits(range(c.nglobal), c.nqubits)
    final_state = c.get_initial_state()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_distributed_circuit_get_initial_state_random(backend, accelerators):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    import itertools
    from qibo.tests_new.utils import random_state
    target_state = random_state(5)
    c = DistributedCircuit(5, accelerators)
    c.queues.qubits = DistributedQubits(range(c.nglobal), c.nqubits)
    state = c.get_initial_state(target_state)
    np.testing.assert_allclose(state, target_state)

    target_state = np.reshape(target_state, 5 * (2,))
    for i, s in enumerate(itertools.product([0, 1], repeat=c.nglobal)):
        target_piece = target_state[s].flatten()
        np.testing.assert_allclose(target_piece.ravel(), state.pieces[i])
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [28, 29, 30, 31, 32, 33, 34])
@pytest.mark.parametrize("ndevices", [2, 4, 8, 16, 32, 64])
def test_distributed_qft_global_qubits_validity(nqubits, ndevices):
    """Check that no gates are applied to global qubits for practical QFT cases."""
    from qibo.models import QFT
    c = QFT(nqubits, accelerators={"/GPU:0": ndevices})
    c.queues.set(c.queue)
    check_device_queues(c.queues)
