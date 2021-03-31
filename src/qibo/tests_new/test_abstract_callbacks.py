"""Test all methods defined in `qibo/abstractions/callbacks.py`."""
import pytest
from qibo.abstractions import callbacks


def test_abstract_callback_properties():
    callback = callbacks.Callback()
    callback.nqubits = 5
    callback.append(1)
    callback.extend([2, 3])
    assert callback.nqubits == 5
    assert callback.results == [1, 2, 3]
    assert not callback.density_matrix
    assert callback._active_call == "state_vector_call"
    callback.density_matrix = True
    assert callback.density_matrix
    assert callback._active_call == "density_matrix_call"


def test_creating_abstract_callbacks():
    callback = callbacks.EntanglementEntropy()
    callback = callbacks.EntanglementEntropy([1, 2], compute_spectrum=True)
    callback = callbacks.Norm()
    callback = callbacks.Overlap()
    callback = callbacks.Energy("test")
    callback = callbacks.Gap()
    callback = callbacks.Gap(2)
    with pytest.raises(ValueError):
        callback = callbacks.Gap("test")
    with pytest.raises(TypeError):
        callback = callbacks.Gap(1.0)
