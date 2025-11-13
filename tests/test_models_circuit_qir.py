"""Test converting Qibo circuits from/into QIR code."""

import re

import numpy as np
import pytest
from openqasm3 import parser
from qbraid.transpiler.conversions.qasm2 import qasm2_to_qasm3
from qbraid.transpiler.conversions.qasm3 import qasm3_to_pyqir

from qibo import Circuit
from qibo.gates import H, M


def _clean_qir_code(circuit_code: str) -> str:
    """Replace program id at line 2 with a mock value."""
    program_id = re.sub(
        r"source_filename = \"([^\"]*)\"", r"\g<1>", circuit_code.split("\n")[1]
    )
    return re.sub(f"{program_id}", "program-id", circuit_code, flags=re.M)


def test_empty():
    target_qasm2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
"""
    target_qir = str(qasm3_to_pyqir(qasm2_to_qasm3(target_qasm2)))
    c = Circuit(2)
    generated = str(c.to_qir())
    assert _clean_qir_code(generated) == _clean_qir_code(target_qir)


def test_h_mz():
    target_qasm2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
    target_qir = str(qasm3_to_pyqir(qasm2_to_qasm3(target_qasm2)))
    c = Circuit(2)
    c.add(H(0))
    c.add(M(0, 1))
    generated = str(c.to_qir())
    assert _clean_qir_code(generated) == _clean_qir_code(target_qir)
