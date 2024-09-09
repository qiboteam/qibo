import os

import qibo
from qibo.backends import _Global

# Force quimb to use qibojit default number of threads.
_Global.backend()
os.environ["NUMBA_NUM_THREADS"] = f"{qibo.get_threads()}"
from qibotn.backends.quimb import QuimbBackend

from qibo.backends import _Global


def test_backend_qibotn():
    qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=None)
    assert isinstance(_Global.get_backend(), QuimbBackend)

    qibo.set_backend("numpy")
