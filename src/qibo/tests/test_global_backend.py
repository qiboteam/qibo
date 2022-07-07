import pytest
import qibo
from qibo import matrices


def test_set_backend():
    from qibo.backends import GlobalBackend
    backend = GlobalBackend()
    qibo.set_backend("numpy")
    assert qibo.get_backend() == "numpy"
    assert GlobalBackend().name == "numpy"


def test_set_precision():
    import numpy as np
    assert qibo.get_precision() == "double"
    qibo.set_precision("single")
    assert matrices.I.dtype == np.complex64
    assert qibo.get_precision() == "single"
    qibo.set_precision("double")
    assert matrices.I.dtype == np.complex128
    assert qibo.get_precision() == "double"
    with pytest.raises(ValueError):
        qibo.set_precision("test")


def test_set_device():
    qibo.set_backend("numpy")
    qibo.set_device("/CPU:0")
    assert qibo.get_device() == "/CPU:0"
    with pytest.raises(ValueError):
        qibo.set_device("test")
    with pytest.raises(ValueError):
        qibo.set_device("/GPU:0")


def test_set_threads():
    with pytest.raises(ValueError):
        qibo.set_threads(-2)
    with pytest.raises(TypeError):
        qibo.set_threads("test")

    qibo.set_backend("numpy")
    assert qibo.get_threads() == 1
    with pytest.raises(ValueError):
        qibo.set_threads(10)


def test_set_shot_batch_size():
    original_batch_size = qibo.get_batch_size()
    qibo.set_batch_size(1024)
    assert qibo.get_batch_size() == 1024
    from qibo.config import SHOT_BATCH_SIZE
    assert SHOT_BATCH_SIZE == 1024
    with pytest.raises(TypeError):
        qibo.set_batch_size("test")
    with pytest.raises(ValueError):
        qibo.set_batch_size(-10)
    with pytest.raises(ValueError):
        qibo.set_batch_size(2 ** 35)
    qibo.set_batch_size(original_batch_size)


def test_set_metropolis_threshold():
    original_threshold = qibo.get_metropolis_threshold()
    qibo.set_metropolis_threshold(100)
    assert qibo.get_metropolis_threshold() == 100
    from qibo.config import SHOT_METROPOLIS_THRESHOLD
    assert SHOT_METROPOLIS_THRESHOLD == 100
    with pytest.raises(TypeError):
        qibo.set_metropolis_threshold("test")
    with pytest.raises(ValueError):
        qibo.set_metropolis_threshold(-10)
    qibo.set_metropolis_threshold(original_threshold)


def test_circuit_execution():
    qibo.set_backend("numpy")
    c = qibo.models.Circuit(2)
    c.add(qibo.gates.H(0))
    result = c()
    unitary = c.unitary()


def test_gate_matrix():
    qibo.set_backend("numpy")
    gate = qibo.gates.H(0)
    matrix = gate.matrix
