"""
Test functions for gate fusion.
"""
import numpy as np
import pytest
from qibo.models import Circuit
from qibo import gates
from qibo.core import fusion


def test_fusion_errors():
    # Fuse distributed circuit after gates are set
    import qibo
    if qibo.get_backend() == "qibotf":
        c = Circuit(4, accelerators={"/GPU:0": 2})
        c.add((gates.H(i) for i in range(4)))
        final_state = c()
        with pytest.raises(RuntimeError):
            fused_c = c.fuse()
