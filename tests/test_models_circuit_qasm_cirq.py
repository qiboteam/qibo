"""Tests executing Qibo circuits created from OpenQASM code."""

# import cirq
import numpy as np
import pytest

from qibo import Circuit, gates

# from cirq.contrib.qasm_import import circuit_from_qasm, exception


# Absolute testing tolerance for cirq-qibo comparison
_atol = 1e-7

pytest.skip(allow_module_level=True)


def test_from_qasm_simple(backend, accelerators):
    target = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];"""
    import qibo

    c = Circuit.from_qasm(target, accelerators)
    assert c.nqubits == 5
    assert c.depth == 1
    for i, gate in enumerate(c.queue):
        assert gate.__class__.__name__ == "H"
        assert gate.qubits == (i,)
    target_state = np.ones(32) / np.sqrt(32)
    final_state = backend.execute_circuit(c).state()
    backend.assert_allclose(final_state, target_state)


def test_simple_cirq(backend):
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.H(1))
    final_state_c1 = backend.execute_circuit(c1).state()

    c2 = circuit_from_qasm(c1.to_qasm())
    c2depth = len(cirq.Circuit(c2.all_operations()))
    assert c1.depth == c2depth
    final_state_c2 = (
        cirq.Simulator().simulate(c2).final_state_vector
    )  # pylint: disable=no-member
    backend.assert_allclose(final_state_c1, final_state_c2, atol=_atol)

    c3 = Circuit.from_qasm(c2.to_qasm())
    assert c3.depth == c2depth
    final_state_c3 = backend.execute_circuit(c3).state()
    backend.assert_allclose(final_state_c3, final_state_c2, atol=_atol)


def test_singlequbit_gates_cirq(backend):
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.X(1))
    c1.add(gates.Y(0))
    c1.add(gates.Z(1))
    c1.add(gates.S(0))
    c1.add(gates.SDG(1))
    c1.add(gates.T(0))
    c1.add(gates.TDG(1))
    c1.add(gates.I(0))
    final_state_c1 = backend.execute_circuit(c1).state()

    c2 = circuit_from_qasm(c1.to_qasm())
    c2depth = len(cirq.Circuit(c2.all_operations()))
    assert c1.depth == c2depth
    final_state_c2 = (
        cirq.Simulator().simulate(c2).final_state_vector
    )  # pylint: disable=no-member
    backend.assert_allclose(final_state_c1, final_state_c2, atol=_atol)

    c3 = Circuit.from_qasm(c2.to_qasm())
    assert c3.depth == c2depth
    final_state_c3 = backend.execute_circuit(c3).state()
    backend.assert_allclose(final_state_c3, final_state_c2, atol=_atol)


def test_multiqubit_gates_cirq(backend):
    c1 = Circuit(2)
    c1.add(gates.H(0))
    c1.add(gates.CNOT(0, 1))
    c1.add(gates.X(1))
    c1.add(gates.SWAP(0, 1))
    c1.add(gates.X(0).controlled_by(1))
    final_state_c1 = backend.execute_circuit(c1).state()

    c2 = circuit_from_qasm(c1.to_qasm())
    c2depth = len(cirq.Circuit(c2.all_operations()))
    assert c1.depth == c2depth
    final_state_c2 = (
        cirq.Simulator().simulate(c2).final_state_vector
    )  # pylint: disable=no-member
    backend.assert_allclose(final_state_c1, final_state_c2, atol=_atol)

    c3 = Circuit.from_qasm(c2.to_qasm())
    assert c3.depth == c2depth
    final_state_c3 = backend.execute_circuit(c3).state()
    backend.assert_allclose(final_state_c3, final_state_c2, atol=_atol)


def test_toffoli_cirq(backend):
    c1 = Circuit(3)
    c1.add(gates.Y(0))
    c1.add(gates.TOFFOLI(0, 1, 2))
    c1.add(gates.X(1))
    c1.add(gates.TOFFOLI(0, 2, 1))
    c1.add(gates.Z(2))
    c1.add(gates.TOFFOLI(1, 2, 0))
    final_state_c1 = backend.execute_circuit(c1).state()

    c2 = circuit_from_qasm(c1.to_qasm())
    c2depth = len(cirq.Circuit(c2.all_operations()))
    assert c1.depth == c2depth
    final_state_c2 = (
        cirq.Simulator().simulate(c2).final_state_vector
    )  # pylint: disable=no-member
    backend.assert_allclose(final_state_c1, final_state_c2, atol=_atol)

    c3 = Circuit.from_qasm(c2.to_qasm())
    assert c3.depth == c2depth
    final_state_c3 = backend.execute_circuit(c3).state()
    backend.assert_allclose(final_state_c3, final_state_c2, atol=_atol)


def test_parametrized_gate_cirq(backend):
    c1 = Circuit(2)
    c1.add(gates.Y(0))
    c1.add(gates.RY(1, 0.1234))
    final_state_c1 = backend.execute_circuit(c1).state()

    c2 = circuit_from_qasm(c1.to_qasm())
    c2depth = len(cirq.Circuit(c2.all_operations()))
    assert c1.depth == c2depth
    final_state_c2 = (
        cirq.Simulator().simulate(c2).final_state_vector
    )  # pylint: disable=no-member
    backend.assert_allclose(final_state_c1, final_state_c2, atol=_atol)

    c3 = Circuit.from_qasm(c2.to_qasm())
    final_state_c3 = backend.execute_circuit(c3).state()
    backend.assert_allclose(final_state_c3, final_state_c2, atol=_atol)


def test_cu1_cirq():
    c1 = Circuit(2)
    c1.add(gates.RX(0, 0.1234))
    c1.add(gates.RZ(1, 0.4321))
    c1.add(gates.CU1(0, 1, 0.567))
    # catches unknown gate "cu1"
    with pytest.raises(exception.QasmException):
        c2 = circuit_from_qasm(c1.to_qasm())


def test_ugates_cirq(backend):
    c1 = Circuit(3)
    c1.add(gates.RX(0, 0.1))
    c1.add(gates.RZ(1, 0.4))
    c1.add(gates.U2(2, 0.5, 0.6))
    final_state_c1 = backend.execute_circuit(c1).state()

    c2 = circuit_from_qasm(c1.to_qasm())
    c2depth = len(cirq.Circuit(c2.all_operations()))
    assert c1.depth == c2depth
    final_state_c2 = (
        cirq.Simulator().simulate(c2).final_state_vector
    )  # pylint: disable=no-member
    backend.assert_allclose(final_state_c1, final_state_c2, atol=_atol)

    c3 = Circuit.from_qasm(c2.to_qasm())
    assert c3.depth == c2depth
    final_state_c3 = backend.execute_circuit(c3).state()
    backend.assert_allclose(final_state_c3, final_state_c2, atol=_atol)

    c1 = Circuit(3)
    c1.add(gates.RX(0, 0.1))
    c1.add(gates.RZ(1, 0.4))
    c1.add(gates.U2(2, 0.5, 0.6))
    c1.add(gates.CU3(2, 1, 0.2, 0.3, 0.4))
    # catches unknown gate "cu3"
    with pytest.raises(exception.QasmException):
        c2 = circuit_from_qasm(c1.to_qasm())


def test_crotations_cirq():
    c1 = Circuit(3)
    c1.add(gates.RX(0, 0.1))
    c1.add(gates.RZ(1, 0.4))
    c1.add(gates.CRX(0, 2, 0.5))
    c1.add(gates.RY(1, 0.3).controlled_by(2))
    # catches unknown gate "crx"
    with pytest.raises(exception.QasmException):
        c2 = circuit_from_qasm(c1.to_qasm())


def test_from_qasm_evaluation(backend):
    import numpy as np

    target = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
h q[1];"""
    c = Circuit.from_qasm(target)
    target_state = np.ones(4) / 2.0
    backend.assert_allclose(backend.execute_circuit(c).state(), target_state)
