import pytest
from qibo import gates
from qibo.models import DistributedCircuit


def test_invalid_devices():
    """Check if error is raised if total devices is not a power of 2."""
    devices = {"/GPU:0": 2, "/GPU:1": 1}
    with pytest.raises(ValueError):
        c = DistributedCircuit(4, devices)


def test_ndevices():
    """Check that ``ndevices`` and ``nglobal`` is set properly."""
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(5, devices)
    assert c.ndevices == 4
    assert c.nglobal == 2


def test_set_gates():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(6, devices)
    c.add((gates.H(i) for i in range(6)))
    c._set_gates()
    assert c.global_qubits_list == [[4, 5], [0, 1]]
    for device in devices.keys():
        assert len(c.queues[device]) == 2
        assert len(c.queues[device][0]) == 4
        assert len(c.queues[device][1]) == 2


def test_set_gates_incomplete():
    devices = {"/GPU:0": 2, "/GPU:1": 2}
    c = DistributedCircuit(6, devices)
    c.add([gates.H(0), gates.H(2), gates.H(3)])
    c.add(gates.CNOT(4, 5))
    c.add([gates.X(1), gates.X(2)])
    c._set_gates()
    assert c.global_qubits_list == [[1, 5], [0, 3]]
    for device in devices.keys():
        assert len(c.queues[device]) == 2
        assert len(c.queues[device][0]) == 3
        assert len(c.queues[device][1]) == 3
