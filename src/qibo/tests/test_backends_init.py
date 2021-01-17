import pytest
from qibo import K, backends, gates
from qibo.backends.abstract import _AVAILABLE_BACKENDS


def test_construct_backend():
    backend = backends._construct_backend("numpy")
    assert backend.name == "numpy"
    backend = backends._construct_backend("tensorflow")
    assert backend.name == "tensorflow"
    with pytest.raises(ValueError):
        bk = backends._construct_backend("test")


@pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
def test_set_backend(backend):
    """Check ``set_backend`` for switching gate backends."""
    original_backend = backends.get_backend()
    backends.set_backend(backend)
    assert backends.get_backend() == backend

    target_name = "numpy" if "numpy" in backend else "tensorflow"
    assert K.name == target_name
    assert str(K) == target_name
    assert K.__repr__() == "{}Backend".format(target_name.capitalize())

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


# TODO: Add set_precision test (move it from base)
# TODO: Add set_device test (move it from base)
