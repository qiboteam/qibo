import itertools

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.quantum_info import random_unitary
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


def _native_gate_sets():
    """All non-empty combinations of the native gates that matter for universality."""
    names = ["Z", "RZ", "GPI2", "U3", "CZ", "iSWAP", "CNOT"]
    for size in range(1, len(names) + 1):
        for combination in itertools.combinations(names, size):
            natives = NativeGates.I | NativeGates.M
            for name in combination:
                natives |= getattr(NativeGates, name)
            yield natives


def _assert_unrolled(circuit, natives, backend, **kwargs):
    unrolled = Unroller(natives, backend=backend, **kwargs)(circuit)
    assert_decomposition(unrolled, natives | NativeGates.RZ | NativeGates.Z)
    target = backend.to_numpy(circuit.unitary(backend))
    result = backend.to_numpy(unrolled.unitary(backend))
    overlap = np.abs(np.trace(target.conj().T @ result)) / target.shape[0]
    backend.assert_allclose(overlap, 1.0, atol=1e-6)


def _lie_algebra_dimension(natives, backend):
    """Dimension of the Lie algebra generated by the native gates on two qubits.

    The gates are universal on two qubits if they generate the whole Lie algebra
    of dimension 15. The generators of the continuous gates are their derivatives
    with respect to the parameters. They are completed with all commutators and
    with the conjugations by the gates, until no new direction appears.
    """
    to_numpy = lambda gate: backend.to_numpy(gate.matrix(backend))
    identity = np.eye(2)
    rng = np.random.default_rng(0)
    generators, conjugators = [], []
    for flag, gate_class, nparams in (
        (NativeGates.RZ, gates.RZ, 1),
        (NativeGates.GPI2, gates.GPI2, 1),
        (NativeGates.U3, gates.U3, 3),
    ):
        if not natives & flag:
            continue
        angles = rng.uniform(0.3, 2.8, nparams)
        matrix = to_numpy(gate_class(0, *angles))
        conjugators.extend([np.kron(matrix, identity), np.kron(identity, matrix)])
        for j in range(nparams):
            step = np.zeros(nparams)
            step[j] = 1e-6
            derivative = (
                (
                    to_numpy(gate_class(0, *(angles + step)))
                    - to_numpy(gate_class(0, *(angles - step)))
                )
                / 2e-6
                @ matrix.conj().T
            )
            generators.extend(
                [np.kron(derivative, identity), np.kron(identity, derivative)]
            )
    if natives & NativeGates.Z:
        z_matrix = to_numpy(gates.Z(0))
        conjugators.extend([np.kron(z_matrix, identity), np.kron(identity, z_matrix)])
    swap = to_numpy(gates.SWAP(0, 1))
    for flag, gate in (
        (NativeGates.CZ, gates.CZ(0, 1)),
        (NativeGates.iSWAP, gates.iSWAP(0, 1)),
        (NativeGates.CNOT, gates.CNOT(0, 1)),
    ):
        if natives & flag:
            conjugators.extend([to_numpy(gate), swap @ to_numpy(gate) @ swap])

    pending = [
        generator - np.trace(generator) / 4 * np.eye(4) for generator in generators
    ]
    directions = np.zeros((0, 32))
    elements = []
    while pending and len(elements) < 15:
        candidate = pending.pop()
        vector = np.concatenate([candidate.real.ravel(), candidate.imag.ravel()])
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            continue
        vector = vector / norm
        vector = vector - directions.T @ (directions @ vector)
        if np.linalg.norm(vector) < 1e-6:
            continue
        directions = np.vstack([directions, vector / np.linalg.norm(vector)])
        elements.append(candidate)
        pending.extend(
            element @ candidate - candidate @ element for element in elements
        )
        pending.extend(
            conjugator @ candidate @ conjugator.conj().T for conjugator in conjugators
        )
    return len(elements)


def test_native_gates_is_universal(backend):
    universal = 0
    for natives in _native_gate_sets():
        dimension = _lie_algebra_dimension(natives, backend)
        assert natives.is_universal == (dimension == 15), natives
        universal += natives.is_universal
    assert universal == 84


@pytest.mark.parametrize(
    "natives",
    [
        NativeGates.CZ,
        NativeGates.U3,
        NativeGates.RZ | NativeGates.CZ,
        NativeGates.GPI2 | NativeGates.Z,
        NativeGates.default() & ~NativeGates.CZ,
    ],
)
def test_unroller_not_universal(backend, natives):
    with pytest.raises(DecompositionError):
        Unroller(natives, backend=backend)


def test_unroller_all_universal_native_gates(backend):
    circuit = Circuit(2)
    circuit.add(
        [
            gates.H(0),
            gates.RX(1, 0.7),
            gates.CNOT(0, 1),
            gates.T(1),
            gates.RY(0, 0.3),
            gates.CZ(1, 0),
            gates.SWAP(0, 1),
            gates.iSWAP(0, 1),
            gates.U3(1, 0.3, 0.4, 0.5),
            gates.Y(0),
        ]
    )
    for natives in _native_gate_sets():
        if natives.is_universal:
            _assert_unrolled(circuit, natives, backend)


@pytest.mark.parametrize(
    "natives", [NativeGates.default(), NativeGates.U3 | NativeGates.iSWAP]
)
@pytest.mark.parametrize(
    "gate",
    [
        lambda backend: gates.SXDG(0),
        lambda backend: gates.GPI(0, 0.3),
        lambda backend: gates.PRX(0, 0.3, 0.4),
        lambda backend: gates.CY(0, 1),
        lambda backend: gates.CSX(0, 1),
        lambda backend: gates.CSXDG(0, 1),
        lambda backend: gates.SiSWAP(0, 1),
        lambda backend: gates.ECR(0, 1),
        lambda backend: gates.RZX(0, 1, 0.3),
        lambda backend: gates.RXXYY(0, 1, 0.3),
        lambda backend: gates.H(1).controlled_by(0),
        lambda backend: gates.Y(1).controlled_by(0),
        lambda backend: gates.S(1).controlled_by(0),
        lambda backend: gates.T(1).controlled_by(0),
        lambda backend: gates.SX(1).controlled_by(0),
        lambda backend: gates.GPI2(1, 0.3).controlled_by(0),
        lambda backend: gates.Unitary(
            random_unitary(2, backend=backend), 1
        ).controlled_by(0),
    ],
)
def test_unroller_gates_without_registered_decomposition(backend, gate, natives):
    gate = gate(backend)
    circuit = Circuit(2)
    circuit.add(gate)
    _assert_unrolled(circuit, natives, backend)


@pytest.mark.parametrize(
    "natives", [NativeGates.default(), NativeGates.U3 | NativeGates.iSWAP]
)
@pytest.mark.parametrize(
    "gate",
    [
        lambda backend: gates.SWAP(1, 2).controlled_by(0),
        lambda backend: gates.iSWAP(1, 2).controlled_by(0),
        lambda backend: gates.RXX(1, 2, 0.3).controlled_by(0),
        lambda backend: gates.RZZ(1, 2, 0.3).controlled_by(0),
        lambda backend: gates.SWAP(2, 3).controlled_by(0, 1),
        lambda backend: gates.RZZ(2, 3, 0.3).controlled_by(0, 1),
    ],
)
def test_unroller_controlled_two_qubit_target(backend, gate, natives):
    gate = gate(backend)
    circuit = Circuit(max(gate.qubits) + 1)
    circuit.add(gate)
    _assert_unrolled(circuit, natives, backend)


@pytest.mark.parametrize(
    "gate",
    [
        lambda backend: gates.fSim(1, 2, 0.3, 0.4).controlled_by(0),
        lambda backend: gates.fSim(2, 3, 0.3, 0.4).controlled_by(0, 1),
        lambda backend: gates.Unitary(
            random_unitary(4, backend=backend), 1, 2
        ).controlled_by(0),
        lambda backend: gates.Unitary(
            random_unitary(4, backend=backend), 2, 3
        ).controlled_by(0, 1),
        lambda backend: gates.Unitary(random_unitary(8, backend=backend), 0, 1, 2),
    ],
)
def test_unroller_unsupported_gates(backend, gate):
    gate = gate(backend)
    circuit = Circuit(max(gate.qubits) + 1)
    circuit.add(gate)
    with pytest.raises(DecompositionError):
        Unroller(NativeGates.default(), backend=backend)(circuit)
