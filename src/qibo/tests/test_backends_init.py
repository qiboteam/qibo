import pytest
from qibo import K, backends, models, gates


def test_construct_backend(backend_name):
    bk = K.construct_backend(backend_name)
    assert bk.name == backend_name
    with pytest.raises(ValueError):
        bk = K.construct_backend("nonexistent")


def test_pickle(backend_name):
    """Check ``K.__setstate__`` and ``K.__getstate__`` methods."""
    import dill
    from qibo.backends import Backend
    backend = Backend()
    backend.active_backend = backend.construct_backend(backend_name)
    if backend_name in ("tensorflow", "qibotf"):
        pytest.skip("Tensorflow backend cannot be pickled.")
    serial = dill.dumps(backend)
    print(type(backend))
    new_backend = dill.loads(serial)
    assert new_backend.name == backend.name
    print(type(new_backend))
    original_backend = new_backend.active_backend.name
    new_backend.active_backend = new_backend.construct_backend("numpy")


def test_set_backend(backend_name):
    """Check ``set_backend`` for switching gate backends."""
    original_backend = backends.get_backend()
    backends.set_backend(backend_name)
    assert K.name == backend_name
    if K.platform is None:
        assert str(K) == backend_name
        assert repr(K) == backend_name
    else:
        platform = K.get_platform()
        assert str(K) == f"{backend_name} ({platform})"
        assert repr(K) == f"{backend_name} ({platform})"
    assert K.executing_eagerly()
    h = gates.H(0)
    backends.set_backend(original_backend)


def test_set_backend_with_platform(backend_name):
    """Check ``set_backend`` with ``platform`` argument."""
    original_backend = backends.get_backend()
    original_platform = K.get_platform()
    backends.set_backend(backend_name, platform="test")
    current_platform = K.get_platform()
    backends.set_backend(original_backend, platform=original_platform)


def test_set_backend_errors(caplog):
    original_backend = backends.get_backend()
    with pytest.raises(ValueError):
        backends.set_backend("nonexistent")
    if original_backend != "numpy":
        h = gates.H(0)
        backends.set_backend("numpy")
        assert "WARNING" in caplog.text
    backends.set_backend(original_backend)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_set_precision(backend, precision):
    original_precision = backends.get_precision()
    backends.set_precision(precision)
    if precision == "single":
        expected_dtype = K.backend.complex64
        expected_tol = 1e-8
    else:
        expected_dtype = K.backend.complex128
        expected_tol = 1e-12
    assert backends.get_precision() == precision
    assert K.dtypes('DTYPECPX') == expected_dtype
    assert K.precision_tol == expected_tol
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


def test_set_precision_errors(backend, caplog):
    original_precision = backends.get_precision()
    gate = gates.H(0)
    backends.set_precision("single")
    assert "WARNING" in caplog.text
    with pytest.raises(ValueError):
        backends.set_precision("test")
    backends.set_precision(original_precision)


def test_set_device(backend, caplog):
    original_devices = {bk: bk.default_device for bk in K.constructed_backends.values()}
    if backends.get_backend() == "numpy":
        backends.set_device("/CPU:0")
        assert "WARNING" in caplog.text
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


def test_set_threads(backend, caplog):
    original_threads = backends.get_threads()
    bkname = backends.get_backend()
    backends.set_threads(1)
    if bkname == "numpy" or bkname == "tensorflow":
        assert "WARNING" in caplog.text
    assert backends.get_threads() == 1
    backends.set_threads(original_threads)


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


def test_set_metropolis_threshold():
    import qibo
    original_threshold = qibo.get_metropolis_threshold()
    assert original_threshold == 100000
    qibo.set_metropolis_threshold(100)
    assert qibo.get_metropolis_threshold() == 100
    from qibo.config import SHOT_METROPOLIS_THRESHOLD
    assert SHOT_METROPOLIS_THRESHOLD == 100
    with pytest.raises(TypeError):
        qibo.set_metropolis_threshold("test")
    with pytest.raises(ValueError):
        qibo.set_metropolis_threshold(-10)
    qibo.set_metropolis_threshold(original_threshold)
