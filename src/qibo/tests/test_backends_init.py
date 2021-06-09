import pytest
from qibo import K, backends, models, gates


def test_construct_backend(backend_name):
    bk = K.construct_backend(backend_name)
    assert bk.name == backend_name
    with pytest.raises(ValueError):
        bk = K.construct_backend("test")


def test_set_backend(backend_name):
    """Check ``set_backend`` for switching gate backends."""
    original_backend = backends.get_backend()
    backends.set_backend(backend_name)
    assert K.name == backend_name
    assert str(K) == backend_name
    assert repr(K) == backend_name
    assert K.executing_eagerly()
    h = gates.H(0)
    if backend_name == "qibotf" or backend_name == "qibojit":
        assert h.gate_op
    else:
        assert h.gate_op is None
    backends.set_backend(original_backend)


def test_set_backend_errors():
    original_backend = backends.get_backend()
    with pytest.raises(ValueError):
        backends.set_backend("test")
    if original_backend != "numpy":
        h = gates.H(0)
        with pytest.warns(RuntimeWarning):
            backends.set_backend("numpy")
    backends.set_backend(original_backend)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_set_precision(backend, precision):
    original_precision = backends.get_precision()
    backends.set_precision(precision)
    if precision == "single":
        expected_dtype = K.backend.complex64
    else:
        expected_dtype = K.backend.complex128
    assert backends.get_precision() == precision
    assert K.dtypes('DTYPECPX') == expected_dtype
    # Test that circuits use proper precision
    circuit = models.Circuit(2)
    circuit.add([gates.H(0), gates.H(1)])
    final_state = circuit()
    assert final_state.dtype == expected_dtype
    backends.set_precision(original_precision)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_set_precision_matrices(backend, precision):
    import numpy as np
    from qibo import matrices
    original_precision = backends.get_precision()
    backends.set_precision(precision)
    if precision == "single":
        assert matrices.dtype == "complex64"
        assert matrices.H.dtype == np.complex64
        assert K.matrices.dtype == "complex64"
        assert K.matrices.X.dtype == K.backend.complex64
    else:
        assert matrices.dtype == "complex128"
        assert matrices.H.dtype == np.complex128
        assert K.matrices.dtype == "complex128"
        assert K.matrices.X.dtype == K.backend.complex128
    backends.set_precision(original_precision)


def test_set_precision_errors(backend):
    original_precision = backends.get_precision()
    gate = gates.H(0)
    with pytest.warns(RuntimeWarning):
        backends.set_precision("single")
    with pytest.raises(ValueError):
        backends.set_precision("test")
    backends.set_precision(original_precision)


def test_set_device(backend):
    original_devices = {bk: bk.default_device for bk in K.constructed_backends.values()}
    if backends.get_backend() == "numpy":
        with pytest.warns(RuntimeWarning):
            backends.set_device("/CPU:0")
    else:
        backends.set_device("/CPU:0")
        assert backends.get_device() == "/CPU:0"
        with pytest.raises(ValueError):
            backends.set_device("test")
        with pytest.raises(ValueError):
            backends.set_device("/TPU:0")
        with pytest.raises(ValueError):
            backends.set_device("/gpu:10")
        with pytest.raises(ValueError):
            backends.set_device("/GPU:10")
    for bk, device in original_devices.items():
        bk.set_device(device)


def test_set_shot_batch_size():
    import qibo
    assert qibo.get_batch_size() == 2 ** 18
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
