import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.abstract import NativeGates
from qibo.transpiler.unroller import (
    DecompositionError,
    DefaultUnroller,
    assert_decomposition,
)


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


def test_no_translate_single_qubit():
    unroller = DefaultUnroller(
        native_gates=NativeGates.default(), translate_single_qubit=False
    )
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.CNOT(0, 1))
    translated_circuit = unroller(circuit)
    assert isinstance(translated_circuit.queue[0], gates.X) and isinstance(
        translated_circuit.queue[2], gates.CZ
    )
