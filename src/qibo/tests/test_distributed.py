import pytest
import numpy as np
from qibo import gates
from qibo import models
from typing import Optional


def random_state(nqubits):
    shape = (2 ** nqubits,)
    x = np.random.random(shape) + 1j * np.random.random(shape)
    x = x / np.sqrt((np.abs(x) ** 2).sum())
    return x


def test_invalid_devices():
    """Check if error is raised if total devices is not a power of 2."""
    devices = {"/GPU:0": 2, "/GPU:1": 1}
    with pytest.raises(ValueError):
        c = models.DistributedCircuit(4, devices)


def test_ndevices():
    """Check that ``ndevices`` and ``nglobal`` is set properly."""
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = models.DistributedCircuit(5, devices)
    assert c.ndevices == 4
    assert c.nglobal == 2


def test_set_gates():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = models.DistributedCircuit(6, devices)
    c.add((gates.H(i) for i in range(6)))
    c._set_gates()
    assert c.global_qubits_list == [[4, 5], [0, 1]]
    for device in devices.keys():
        assert len(c.queues[device]) == 2
        assert len(c.queues[device][0]) == 4
        assert len(c.queues[device][1]) == 2


def test_set_gates_incomplete():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = models.DistributedCircuit(6, devices)
    c.add([gates.H(0), gates.H(2), gates.H(3)])
    c.add(gates.CNOT(4, 5))
    c.add([gates.X(1), gates.X(2)])
    c._set_gates()
    assert c.global_qubits_list == [[1, 5], [0, 3]]
    for device in devices.keys():
        assert len(c.queues[device]) == 2
        assert len(c.queues[device][0]) == 3
        assert len(c.queues[device][1]) == 3


def test_default_initialization():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = models.DistributedCircuit(6, devices)
    c._set_initial_state()
    assert c.global_qubits == {4, 5}

    final_state = c.final_state.numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1
    np.testing.assert_allclose(target_state, final_state)


@pytest.mark.parametrize("nqubits", [5, 6])
def test_user_initialization(nqubits):
    import itertools
    target_state = random_state(nqubits)

    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = models.DistributedCircuit(nqubits, devices)
    c._set_initial_state(target_state)

    final_state = c.final_state.numpy()
    np.testing.assert_allclose(target_state, final_state)

    target_state = target_state.reshape(nqubits * (2,))
    n = nqubits - c.nglobal
    for i, s in enumerate(itertools.product([0, 1], repeat=c.nglobal)):
        piece = c.pieces[i].numpy()
        target_piece = target_state[n * (slice(None),) + s]
        np.testing.assert_allclose(target_piece, piece)


def test_simple_execution():
    c = models.Circuit(6)
    c.add((gates.H(i) for i in range(6)))

    devices = {"/GPU:0": 2, "/GPU:1": 2}
    dist_c = models.DistributedCircuit(6, devices)
    dist_c.add((gates.H(i) for i in range(6)))

    initial_state = random_state(c.nqubits)
    final_state = dist_c(initial_state)
    target_state = c(initial_state)
