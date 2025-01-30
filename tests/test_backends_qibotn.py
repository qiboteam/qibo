from numba import set_num_threads

from qibo import get_backend, get_threads, set_backend

# Force quimb to use qibojit default number of threads.
set_num_threads(get_threads())
from qibotn.backends.quimb import QuimbBackend


def test_backend_qibotn():
    set_backend(backend="qibotn", platform="qutensornet", runcard=None)
    assert isinstance(get_backend(), QuimbBackend)
    set_backend("numpy")
    assert get_backend().name == "numpy"
