from qibotn.backends import QuimbBackend

import qibo
from qibo.backends import GlobalBackend


def test_backend_qibotn():
    qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=None)
    assert isinstance(GlobalBackend(), QuimbBackend)
