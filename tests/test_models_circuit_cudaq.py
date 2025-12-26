"""Test converting Qibo circuits into cudaq code."""

import re

import pytest

from qibo import Circuit
from qibo.gates import H, M, X

cudaq = pytest.importorskip("cudaq")


def _clean_cudaq_code(circuit_code: str) -> str:
    return re.sub(r"__nvqpp__mlirgen__[a-zA-Z0-9_]+", "", circuit_code, flags=re.M)


def test_empty():
    target = """module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__empty_circuit = "__nvqpp__mlirgen__empty_circuit_PyKernelEntryPointRewrite"}} {
  func.func @__nvqpp__mlirgen__empty_circuit() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.alloca !quake.veq<2>
    return
  }
}
"""
    c = Circuit(2)
    generated = str(c.to_cudaq())
    assert _clean_cudaq_code(target) == _clean_cudaq_code(generated)


def test_import_from_cudaq():
    @cudaq.kernel
    def cudaq_circuit():
        qvector = cudaq.qvector(2)
        h(qvector[0])
        x(qvector[0], qvector[1])
        mz(qvector[0])

    c = Circuit.from_cudaq(cudaq_circuit)
    assert isinstance(c.queue[0], H)
    assert isinstance(c.queue[1], X)
    assert isinstance(c.queue[2], X)
    assert isinstance(c.queue[3], M)
    assert c.measurement_tuples == {"var3": (0,)}
