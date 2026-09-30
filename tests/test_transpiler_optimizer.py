import pytest

from qibo import gates
from qibo.gates.special import Barrier
from qibo.models import Circuit
from qibo.transpiler.optimizer import (
    InverseCancellation,
    Preprocessing,
    Rearrange,
    TGateRules,
)
from qibo.transpiler.pipeline import Passes


def test_preprocessing_error(star_connectivity):
    circ = Circuit(7)
    preprocesser = Preprocessing(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        preprocesser(circuit=circ)

    wire_names = [0, 1, 2, "q3", "q4"]
    circ = Circuit(5, wire_names=wire_names)
    assert circ.wire_names == wire_names


def test_preprocessing_same(star_connectivity):
    circ = Circuit(5)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1


def test_preprocessing_add(star_connectivity):
    circ = Circuit(3)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 5


def test_fusion(backend):
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.Z(0))
    circuit.add(gates.Y(0))
    circuit.add(gates.X(1))
    fusion = Rearrange(max_qubits=1)
    fused_circ = fusion(circuit, backend=backend)
    assert isinstance(fused_circ.queue[0], gates.Unitary)


def test_inverse_cancellation_pairs(backend):
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.H(0))
    circuit.add(gates.X(1))
    circuit.add(gates.Y(1))
    circuit.add(gates.Y(1))
    circuit.add(gates.X(1))
    circuit.add(gates.H(2))
    circuit.add(gates.X(0))
    circuit.add(gates.H(2))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.RX(0, -0.3))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert [gate.name for gate in reduced.queue] == ["x"]
    backend.assert_allclose(
        reduced.unitary(backend), circuit.unitary(backend), atol=1e-12
    )


def test_inverse_cancellation_no_pairs(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.H(1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 0))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert [gate.name for gate in reduced.queue] == [
        gate.name for gate in circuit.queue
    ]
    backend.assert_allclose(
        reduced.unitary(backend), circuit.unitary(backend), atol=1e-12
    )


def test_inverse_cancellation_uncancellable_gates(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.Align(0))
    circuit.add(gates.H(0))
    circuit.add(gates.M(1))
    circuit.add(gates.M(1))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert reduced.ngates == circuit.ngates

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(Barrier(0, 1))
    circuit.add(gates.H(0))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert reduced.ngates == 3

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(Barrier(1, 2))
    circuit.add(gates.H(0))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert [gate.name for gate in reduced.queue] == ["barrier"]

    circuit = Circuit(1, density_matrix=True)
    circuit.add(gates.H(0))
    circuit.add(gates.DepolarizingChannel((0,), 0.1))
    circuit.add(gates.H(0))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert reduced.ngates == 3
    assert reduced.density_matrix


def test_inverse_cancellation_controlled(backend):
    minus_identity = -backend.identity(2)

    circuit = Circuit(2)
    circuit.add(gates.Unitary(minus_identity, 1).controlled_by(0))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert reduced.ngates == 1

    circuit.add(gates.Unitary(minus_identity, 1).controlled_by(0))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert reduced.ngates == 0

    circuit = Circuit(2)
    circuit.add(gates.H(1).controlled_by(0))
    circuit.add(gates.CZ(0, 1))
    reduced = InverseCancellation()(circuit, backend=backend)
    assert reduced.ngates == 2


def test_inverse_cancellation_atol(backend):
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.RX(0, -0.3 + 1e-6))
    assert InverseCancellation()(circuit, backend=backend).ngates == 2
    assert InverseCancellation(atol=1e-5)(circuit, backend=backend).ngates == 0

    with pytest.raises(ValueError):
        InverseCancellation(atol=-1e-3)


def test_inverse_cancellation_pipeline(backend, star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 2))
    pipeline = Passes([InverseCancellation()], connectivity=star_connectivity())
    transpiled, _ = pipeline(circuit, backend=backend)
    assert [gate.name for gate in transpiled.queue] == ["cx"]


@pytest.mark.parametrize("power", range(20))
def test_t_gate_rules_powers(backend, power):
    circuit = Circuit(1)
    circuit.add(gates.T(0) for _ in range(power))
    new = TGateRules()(circuit)

    backend.assert_allclose(new.unitary(backend), circuit.unitary(backend), atol=1e-12)
    max_gates = [0, 1, 1, 2, 1, 2, 1, 1]
    assert new.ngates == max_gates[power % 8]


def test_t_gate_rules_separated_runs(backend):
    circuit = Circuit(3)
    circuit.add(gates.T(0))
    circuit.add(gates.T(1))
    circuit.add(gates.T(0))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.T(0))
    circuit.add(gates.T(0).controlled_by(1))
    circuit.add(gates.T(0))
    circuit.add(gates.M(0))
    circuit.add(gates.T(0))
    new = TGateRules()(circuit)

    assert [type(gate) for gate in new.queue] == [
        gates.S,
        gates.CNOT,
        gates.T,
        gates.T,
        gates.T,
        gates.T,
        gates.M,
        gates.T,
    ]
    assert new.queue[4].control_qubits == (1,)

    unitary_circuit = Circuit(3)
    unitary_circuit.add(gate for gate in circuit.queue if not isinstance(gate, gates.M))
    unitary_new = Circuit(3)
    unitary_new.add(gate for gate in new.queue if not isinstance(gate, gates.M))
    backend.assert_allclose(
        unitary_new.unitary(backend), unitary_circuit.unitary(backend), atol=1e-12
    )
