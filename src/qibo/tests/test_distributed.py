import pytest
import numpy as np
from qibo import gates
from qibo import models


def random_state(nqubits):
    shape = (2 ** nqubits,)
    x = np.random.random(shape) + 1j * np.random.random(shape)
    x = x / np.sqrt((np.abs(x) ** 2).sum())
    return x


def assert_global_qubits(global_qubits_list, gate_groups):
    """Asserts that global qubits do not collide with the gates to be applied."""
    assert len(global_qubits_list) == len(gate_groups)
    for global_qubits, gate_list in zip(global_qubits_list, gate_groups):
        target_qubits = set()
        for gate in gate_list:
            target_qubits |= set(gate.original_gate.qubits)
        assert not set(global_qubits) & target_qubits


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

    assert_global_qubits(c.global_qubits_list, c.queues["/GPU:0"])
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

    assert_global_qubits(c.global_qubits_list, c.queues["/GPU:0"])
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
    final_state = dist_c(initial_state).numpy()
    target_state = c(initial_state).numpy()
    np.testing.assert_allclose(target_state, final_state)


@pytest.mark.parametrize("nqubits", [7, 8, 30, 31, 32, 33])
@pytest.mark.parametrize("ndevices", [2, 4])
def test_distributed_qft_global_qubits(nqubits, ndevices):
    """Check that the generated global qubit list is the expected for QFT."""
    devices = {"/GPU:0": ndevices}
    c = models.DistributedQFT(nqubits, devices)
    c._set_gates()

    for i, queue in enumerate(c.queues["/GPU:0"]):
        print(len(queue), c.global_qubits_list[i])
        for g in queue:
            print(g.name, g.original_gate.qubits)
        print()
        print()

    assert_global_qubits(c.global_qubits_list, c.queues["/GPU:0"])
    nglobal = c.nglobal
    target_global_qubits = [list(range(nqubits - nglobal, nqubits)),
                            list(range(nqubits - 2 * nglobal, nqubits - nglobal)),
                            list(range(nglobal)),
                            list(range(nglobal, 2 * nglobal))]
    try:
        assert target_global_qubits == c.global_qubits_list
    except AssertionError:
        assert len(c.global_qubits_list) < len(target_global_qubits)


@pytest.mark.parametrize("nqubits", [7, 8])
@pytest.mark.parametrize("ndevices", [2, 4])
def test_distributed_qft_execution(nqubits, ndevices):
    devices = {"/GPU:0": ndevices}
    dist_c = models.DistributedQFT(nqubits, devices)
    c = models.QFT(nqubits)

    initial_state = random_state(nqubits)
    final_state = dist_c(initial_state).numpy()
    target_state = c(initial_state).numpy()
    np.testing.assert_allclose(target_state, final_state)
