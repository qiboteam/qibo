from qibotn.backends.quimb import QuimbBackend

from qibo import get_backend, set_backend


def test_backend_qibotn():
    set_backend(backend="qibotn", platform="qutensornet", runcard=None)
    assert isinstance(get_backend(), QuimbBackend)
    set_backend("numpy")
    assert get_backend().name == "numpy"
