import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler._exceptions import DecompositionError
from qibo.transpiler.asserts import assert_decomposition
from qibo.transpiler.unroller import NativeGates, Unroller, translate_gate


def test_native_gates_from_gatelist():
    natives = NativeGates.from_gatelist([gates.RZ, gates.CZ(0, 1)])
    assert natives == NativeGates.RZ | NativeGates.CZ


def test_native_gates_from_gatelist_fail():
    with pytest.raises(ValueError):
        NativeGates.from_gatelist([gates.RZ, gates.X(0)])


def test_native_gate_str_list():
    testlist = ["I", "Z", "RZ", "M", "GPI2", "U3", "CZ", "iSWAP", "CNOT"]
    natives = NativeGates[testlist]
    for gate in testlist:
        assert NativeGates[gate] in natives

    natives = NativeGates[["qi", "bo"]]  # Invalid gate names
    assert natives == NativeGates(0)


def test_translate_gate_error_1q():
    natives = NativeGates(0)
    with pytest.raises(DecompositionError):
        translate_gate(gates.X(0), natives)


def test_translate_gate_error_2q():
    natives = NativeGates(0)
    with pytest.raises(DecompositionError):
        translate_gate(gates.CZ(0, 1), natives)


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


def test_temp_cnot_decomposition():
    from qibo.transpiler.pipeline import Passes

    circ = Circuit(2)
    circ.add(gates.H(0))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.M(0, 1))

    glist = [gates.GPI2, gates.RZ, gates.Z, gates.M, gates.CNOT]
    native_gates = NativeGates(0).from_gatelist(glist)

    unroller = Unroller(native_gates=native_gates)
    transpiled_circuit = unroller(circ)

    # H
    assert transpiled_circuit.queue[0].name == "z"
    assert transpiled_circuit.queue[1].name == "gpi2"
    # CNOT
    assert transpiled_circuit.queue[2].name == "cx"
    # SWAP
    assert transpiled_circuit.queue[3].name == "cx"
    assert transpiled_circuit.queue[4].name == "cx"
    assert transpiled_circuit.queue[5].name == "cx"
    # CZ
    assert transpiled_circuit.queue[6].name == "z"
    assert transpiled_circuit.queue[7].name == "gpi2"
    assert transpiled_circuit.queue[8].name == "cx"
    assert transpiled_circuit.queue[9].name == "z"
    assert transpiled_circuit.queue[10].name == "gpi2"
