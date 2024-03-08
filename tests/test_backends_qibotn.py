import os

import qibo

# Force quimb to use qibojit default number of threads.
os.environ["NUMBA_NUM_THREADS"] = f"{qibo.get_threads()}"
from qibotn.backends.quimb import QuimbBackend

from qibo.backends import GlobalBackend


def test_backend_qibotn():
    qibo.set_backend(backend="qibotn", platform="qutensornet")
    assert isinstance(GlobalBackend(), QuimbBackend)
