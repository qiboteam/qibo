"""Test converting Qibo circuits into cudaq code."""

import re

import pytest

from qibo import Circuit

pytest.importorskip("cudaq")


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
