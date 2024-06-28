import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler._exceptions import DecompositionError
from qibo.transpiler.unroller import (
    NativeGates,
    Unroller,
    assert_decomposition,
    translate_gate,
)


def test_native_gates_from_gatelist():
    natives = NativeGates.from_gatelist([gates.RZ, gates.CZ(0, 1)])
    assert natives == NativeGates.RZ | NativeGates.CZ


def test_native_gates_from_gatelist_fail():
    with pytest.raises(ValueError):
        NativeGates.from_gatelist([gates.RZ, gates.X(0)])


def test_translate_gate_error_1q():
    natives = NativeGates(0)
    with pytest.raises(DecompositionError):
        translate_gate(gates.X(0), natives)


def test_translate_gate_error_2q():
    natives = NativeGates(0)
    with pytest.raises(DecompositionError):
        translate_gate(gates.CZ(0, 1), natives)


def test_assert_decomposition():
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.Z(0))
    circuit.add(gates.M(1))
    assert_decomposition(circuit, native_gates=NativeGates.default())


def test_assert_decomposition_fail_1q():
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    with pytest.raises(DecompositionError):
        assert_decomposition(circuit, native_gates=NativeGates.default())


@pytest.mark.parametrize("gate", [gates.CNOT(0, 1), gates.iSWAP(0, 1)])
def test_assert_decomposition_fail_2q(gate):
    circuit = Circuit(2)
    circuit.add(gate)
    with pytest.raises(DecompositionError):
        assert_decomposition(circuit, native_gates=NativeGates.default())


def test_assert_decomposition_fail_3q():
    circuit = Circuit(3)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(DecompositionError):
        assert_decomposition(circuit, native_gates=NativeGates.default())


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_unroller(natives_1q, natives_2q):
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.X(0))
    circuit.add(gates.Y(0))
    circuit.add(gates.Z(0))
    circuit.add(gates.S(0))
    circuit.add(gates.T(0))
    circuit.add(gates.SDG(0))
    circuit.add(gates.TDG(0))
    circuit.add(gates.SX(0))
    circuit.add(gates.RX(0, 0.1))
    circuit.add(gates.RY(0, 0.2))
    circuit.add(gates.RZ(0, 0.3))
    circuit.add(gates.U1(0, 0.4))
    circuit.add(gates.U2(0, 0.5, 0.6))
    circuit.add(gates.U3(0, 0.7, 0.8, 0.9))
    circuit.add(gates.GPI2(0, 0.123))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.iSWAP(0, 1))
    circuit.add(gates.FSWAP(0, 1))
    circuit.add(gates.CRX(0, 1, 0.1))
    circuit.add(gates.CRY(0, 1, 0.2))
    circuit.add(gates.CRZ(0, 1, 0.3))
    circuit.add(gates.CU1(0, 1, 0.4))
    circuit.add(gates.CU2(0, 1, 0.5, 0.6))
    circuit.add(gates.CU3(0, 1, 0.7, 0.8, 0.9))
    circuit.add(gates.RXX(0, 1, 0.1))
    circuit.add(gates.RYY(0, 1, 0.2))
    circuit.add(gates.RZZ(0, 1, 0.3))
    circuit.add(gates.fSim(0, 1, 0.4, 0.5))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    unroller = Unroller(native_gates=natives_1q | natives_2q)
    translated_circuit = unroller(circuit)
    assert_decomposition(
        translated_circuit,
        native_gates=natives_1q | natives_2q | NativeGates.RZ | NativeGates.Z,
    )


def test_measurements_non_comp_basis():
    unroller = Unroller(native_gates=NativeGates.default())
    circuit = Circuit(1)
    circuit.add(gates.M(0, basis=gates.X))
    transpiled_circuit = unroller(circuit)
    assert isinstance(transpiled_circuit.queue[2], gates.M)
    # After transpiling the measurement gate should be in the computational basis
    assert transpiled_circuit.queue[2].basis == []
