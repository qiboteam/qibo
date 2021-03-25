import pytest
from qibo import K, backends, models, gates


def test_construct_backend(backend):
    bk = backends._construct_backend(backend)
    try:
        assert bk.name == backend
    except AssertionError:
        assert bk.name.split("_")[-1] == backend
    with pytest.raises(ValueError):
        bk = backends._construct_backend("test")


def test_set_backend(backend):
    """Check ``set_backend`` for switching gate backends."""
    original_backend = backends.get_backend()
    backends.set_backend(backend)
    if backend == "custom":
        assert K.custom_gates
        assert K.custom_einsum is None
        from qibo.core import cgates as custom_gates
        assert isinstance(gates.H(0), custom_gates.BackendGate)
    else:
        assert not K.custom_gates
        if "defaulteinsum" in backend:
            assert K.custom_einsum == "DefaultEinsum"
        else:
            assert K.custom_einsum == "MatmulEinsum"
        from qibo.core import gates as native_gates
        h = gates.H(0)
        assert isinstance(h, native_gates.BackendGate)
    backends.set_backend(original_backend)


def test_set_backend_errors():
    original_backend = backends.get_backend()
    with pytest.raises(ValueError):
        backends.set_backend("test")
    with pytest.raises(ValueError):
        backends.set_backend("numpy_custom")
    with pytest.raises(ValueError):
        backends.set_backend("numpy_badgates")
    h = gates.H(0)
    with pytest.warns(RuntimeWarning):
        backends.set_backend("numpy_defaulteinsum")
    backends.set_backend(original_backend)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_set_precision(backend, precision):
    original_backend = backends.get_backend()
    original_precision = backends.get_precision()
    backends.set_backend(backend)
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
    backends.set_backend(original_backend)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_set_precision_matrices(backend, precision):
    import numpy as np
    from qibo import matrices
    original_backend = backends.get_backend()
    original_precision = backends.get_precision()
    backends.set_backend(backend)
    backends.set_precision(precision)
    if precision == "single":
        assert matrices.dtype == np.complex64
        assert matrices.H.dtype == np.complex64
        assert K.matrices.dtype == K.backend.complex64
        assert K.matrices.X.dtype == K.backend.complex64
    else:
        assert matrices.dtype == np.complex128
        assert matrices.H.dtype == np.complex128
        assert K.matrices.dtype == K.backend.complex128
        assert K.matrices.X.dtype == K.backend.complex128
    backends.set_precision(original_precision)
    backends.set_backend(original_backend)


def test_set_precision_errors():
    original_precision = backends.get_precision()
    gate = gates.H(0)
    with pytest.warns(RuntimeWarning):
        backends.set_precision("single")
    with pytest.raises(ValueError):
        backends.set_precision("test")
    backends.set_precision(original_precision)


def test_set_device(backend):
    original_backend = backends.get_backend()
    original_device = backends.get_device()
    backends.set_backend(backend)
    if "numpy" in backend:
        with pytest.warns(RuntimeWarning):
            backends.set_device("/CPU:0")
    else:
        backends.set_device("/CPU:0")
        with pytest.raises(ValueError):
            backends.set_device("test")
        with pytest.raises(ValueError):
            backends.set_device("/TPU:0")
        with pytest.raises(ValueError):
            backends.set_device("/gpu:10")
        with pytest.raises(ValueError):
            backends.set_device("/GPU:10")
    backends.set_device(original_device)
    backends.set_backend(original_backend)


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
