import pytest

from qibo import get_backend, set_backend

# from qibotn.backends.quimb import QuimbBackend


@pytest.mark.skip(reason="Inverse dependency currently suppressed.")
def test_backend_qibotn():
    set_backend(backend="qibotn", platform="qutensornet", runcard=None)
    assert isinstance(get_backend(), QuimbBackend)
    set_backend("numpy")
    assert get_backend().name == "numpy"
