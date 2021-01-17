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


def test_partial_trace_partition():
    callback = callbacks.PartialTrace([0, 1])
    assert callback.partition == [0, 1]
    # test default partition
    callback = callbacks.PartialTrace()
    assert callback.partition is None
    callback.nqubits = 5
    assert callback.partition == [0, 1, 2]
    # test partition reversion
    callback = callbacks.PartialTrace([0, 1])
    callback.nqubits = 6
    assert callback.partition == [2, 3, 4, 5]


def test_partial_trace_einsum_string():
    func = callbacks.PartialTrace.einsum_string
    estr = func({0, 2, 4}, 5)
    assert estr == "abcdeagcie->bdgi"
    estr = func({0, 2, 4}, 5, measuring=True)
    assert estr == "abcdeabcde->bd"
    estr = func({0, 1, 3, 5, 6}, 10, measuring=False)
    assert estr == "abcdefghijabmdofgrst->cehijmorst"
    estr = func({0, 1, 3, 5, 6}, 10, measuring=True)
    assert estr == "abcdefghijabcdefghij->cehij"


def test_partial_trace_traceout():
    callback = callbacks.PartialTrace([0, 2, 3, 4])
    callback.nqubits = 6
    assert callback.traceout() == "abcdefahcdel->bfhl"


def test_creating_rest_abstract_callbacks():
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
