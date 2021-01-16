from qibo.abstractions import callbacks


def test_abstract_callback_properties():
    callback = callbacks.Callback()
    callback.nqubits = 5
    callback.append(1)
    callback.extend([2, 3])
    assert callback.nqubits == 5
    assert callback.results == [1, 2, 3]
    assert callback._active_call == "state_vector_call"
    callback.density_matrix = True
    assert callback._active_call == "density_matrix_call"
