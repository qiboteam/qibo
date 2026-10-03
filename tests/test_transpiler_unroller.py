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


def test_translate_gate_error_1q(backend):
    natives = NativeGates(0)
    with pytest.raises(DecompositionError):
        translate_gate(gates.X(0), natives, backend=backend)


def test_translate_gate_error_2q(backend):
    natives = NativeGates(0)
    with pytest.raises(DecompositionError):
        translate_gate(gates.CZ(0, 1), natives, backend=backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_unroller(backend, natives_1q, natives_2q):
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
    unroller = Unroller(native_gates=natives_1q | natives_2q, backend=backend)
    translated_circuit = unroller(circuit)
    assert_decomposition(
        translated_circuit,
        native_gates=natives_1q | natives_2q | NativeGates.RZ | NativeGates.Z,
    )


def test_measurements_non_comp_basis(backend):
    unroller = Unroller(native_gates=NativeGates.default(), backend=backend)
    circuit = Circuit(1)
    circuit.add(gates.M(0, basis=gates.X))
    transpiled_circuit = unroller(circuit)
    assert isinstance(transpiled_circuit.queue[2], gates.M)
    # After transpiling the measurement gate should be in the computational basis
    assert transpiled_circuit.queue[2].basis == []


def test_temp_cnot_decomposition(backend):

    circ = Circuit(2)
    circ.add(gates.H(0))
    circ.add(gates.CNOT(0, 1))
    circ.add(gates.SWAP(0, 1))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.M(0, 1))

    glist = [gates.GPI2, gates.RZ, gates.Z, gates.M, gates.CNOT]
    native_gates = NativeGates(0).from_gatelist(glist)

    unroller = Unroller(native_gates=native_gates, backend=backend)
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


@pytest.mark.parametrize(
    "gate",
    [
        gates.X(4).controlled_by(0, 1, 2, 3),
        gates.RX(3, 0.7).controlled_by(0, 1, 2),
        gates.H(3).controlled_by(0, 1, 2),
        gates.U3(2, 0.3, 0.5, 0.9).controlled_by(0, 1),
    ],
)
def test_unroller_multi_controlled_gates(backend, gate):
    circuit = Circuit(5)
    circuit.add(gate)
    natives = NativeGates.default()
    unrolled = Unroller(natives, backend=backend)(circuit)

    assert all(len(g.qubits) <= 2 and not g.is_controlled_by for g in unrolled.queue)
    assert_decomposition(unrolled, natives)

    # The unroller drops the global phase.
    original = circuit.unitary(backend)
    final = unrolled.unitary(backend)
    overlap = backend.sum(backend.conj(original) * final)
    backend.assert_allclose(final, original * overlap / backend.abs(overlap), atol=1e-8)


def _assert_equal_up_to_global_phase(backend, circuit, unrolled):
    original = circuit.unitary(backend)
    final = unrolled.unitary(backend)
    overlap = backend.sum(backend.conj(original) * final)
    backend.assert_allclose(final, original * overlap / backend.abs(overlap), atol=1e-8)


@pytest.mark.parametrize("nctrl", [4, 5, 6])
def test_unroller_dirty_ancillas(backend, nctrl):
    """Spare qubits make the unrolled multi-controlled ``X`` cheaper."""
    circuit = Circuit(nctrl + 3)
    circuit.add(gates.X(nctrl).controlled_by(*range(nctrl)))
    natives = NativeGates.default()

    without = Unroller(natives, backend=backend)(circuit)
    with_ancillas = Unroller(natives, backend=backend, use_dirty_ancillas=True)(circuit)

    assert_decomposition(with_ancillas, natives)
    _assert_equal_up_to_global_phase(backend, circuit, without)
    _assert_equal_up_to_global_phase(backend, circuit, with_ancillas)
    assert with_ancillas.gate_types[gates.CZ] < without.gate_types[gates.CZ]


def test_unroller_dirty_ancillas_without_spare_qubits(backend):
    """Without spare qubits the result is the same as without the option."""
    circuit = Circuit(5)
    circuit.add(gates.X(4).controlled_by(0, 1, 2, 3))
    natives = NativeGates.default()

    without = Unroller(natives, backend=backend)(circuit)
    with_ancillas = Unroller(natives, backend=backend, use_dirty_ancillas=True)(circuit)

    assert [g.name for g in with_ancillas.queue] == [g.name for g in without.queue]
    assert [g.qubits for g in with_ancillas.queue] == [g.qubits for g in without.queue]


def test_unroller_dirty_ancillas_do_not_change_other_gates(backend):
    """Only multi-controlled ``X`` gates benefit from auxiliary qubits."""
    circuit = Circuit(7)
    circuit.add(gates.H(0))
    circuit.add(gates.RX(3, 0.3).controlled_by(0, 1, 2))
    circuit.add(gates.CNOT(5, 6))
    natives = NativeGates.default()

    without = Unroller(natives, backend=backend)(circuit)
    with_ancillas = Unroller(natives, backend=backend, use_dirty_ancillas=True)(circuit)

    assert [g.name for g in with_ancillas.queue] == [g.name for g in without.queue]
    assert [g.qubits for g in with_ancillas.queue] == [g.qubits for g in without.queue]


def test_translate_gate_free_qubits(backend):
    gate = gates.X(4).controlled_by(0, 1, 2, 3)
    natives = NativeGates.default()

    without = translate_gate(gate, natives, backend)
    with_free = translate_gate(gate, natives, backend, free=(5, 6))

    count = lambda gate_list: sum(isinstance(g, gates.CZ) for g in gate_list)
    assert count(with_free) < count(without)
