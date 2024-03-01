import pytest
from qibotn.backends import CuTensorNet, QuimbBackend

import qibo
from qibo.backends import GlobalBackend


@pytest.mark.parametrize("platform", ["cutensornet", "qutensornet"])
def test_backend_qibotn(platform):

    qibo.set_backend(backend="qibotn", platform=platform, runcard=None)

    if platform == "qutensornet":
        qibotn_backend = QuimbBackend
    else:
        qibotn_backend = CuTensorNet

    assert isinstance(GlobalBackend(), qibotn_backend)
