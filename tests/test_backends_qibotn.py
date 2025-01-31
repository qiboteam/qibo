from qibo import get_backend, get_threads, set_backend


def test_backend_qibotn(numba_threads):
    # Force quimb to use qibojit default number of threads.
    with numba_threads(get_threads()):
        from qibotn.backends.quimb import QuimbBackend

        set_backend(backend="qibotn", platform="qutensornet", runcard=None)
        assert isinstance(get_backend(), QuimbBackend)
        set_backend("numpy")
        assert get_backend().name == "numpy"
