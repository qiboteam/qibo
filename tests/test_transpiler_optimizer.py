import math

import numpy as np
import pytest
import sympy

from qibo import Circuit, gates
from qibo.gates.special import Barrier
from qibo.transpiler.abstract import Optimizer
from qibo.transpiler.optimizer import (
    ConsolidateBlocks,
    FixedPoint,
    InverseCancellation,
    Optimize1qGatesDecomposition,
    ParametrizedGateFusion,
    Preprocessing,
    Rearrange,
    RemoveDiagonalGatesBeforeMeasurement,
    RemoveFinalReset,
    RemoveIdentityEquivalent,
    RemoveResetInZeroState,
    ResetAfterMeasureSimplification,
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


def test_preprocessing_too_many_logical_qubits(star_connectivity):
    """``Preprocessing`` raises when the logical qubits exceed the physical ones.

    Uses duplicate wire names so that all names are in the connectivity graph
    while ``nqubits`` is still larger than the number of physical qubits.
    """
    circ = Circuit(6, wire_names=[0, 1, 2, 0, 1, 2])
    preprocesser = Preprocessing(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        preprocesser(circuit=circ)


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


# (gate class, number of qubits, number of parameters that add up, number of
# parameters that must coincide, and what listing the qubits in reversed order does:
# ``1`` gives the same gate, ``-1`` flips the sign of the added parameters and
# ``None`` gives a different gate)
FUSABLE_GATES = [
    (gates.RX, 1, 1, 0, None),
    (gates.RY, 1, 1, 0, None),
    (gates.RZ, 1, 1, 0, None),
    (gates.U1, 1, 1, 0, None),
    (gates.PRX, 1, 1, 1, None),
    (gates.U1q, 1, 1, 1, None),
    (gates.CRX, 2, 1, 0, None),
    (gates.CRY, 2, 1, 0, None),
    (gates.CRZ, 2, 1, 0, None),
    (gates.CU1, 2, 1, 0, None),
    (gates.RXX, 2, 1, 0, 1),
    (gates.RYY, 2, 1, 0, 1),
    (gates.RZZ, 2, 1, 0, 1),
    (gates.RZX, 2, 1, 0, None),
    (gates.RXXYY, 2, 1, 0, 1),
    (gates.RBS, 2, 1, 0, -1),
    (gates.GIVENS, 2, 1, 0, -1),
    (gates.fSim, 2, 2, 0, 1),
]

# U gates by their number of angles
U_GATES = {1: gates.U1, 2: gates.U2, 3: gates.U3}


@pytest.mark.parametrize("gate_class,nqubits,nadded,nequal,reversal", FUSABLE_GATES)
def test_parametrized_gate_fusion_rules(
    backend, gate_class, nqubits, nadded, nequal, reversal
):
    fusion = ParametrizedGateFusion()
    qubits = tuple(range(nqubits))
    angles = backend.random_uniform(-3, 3, (2, nadded))
    equal = backend.random_uniform(-3, 3, nequal)
    first = gate_class(*qubits, *angles[0], *equal)
    second = gate_class(*qubits, *angles[1], *equal)

    # the angles add up and the merge is exact
    circuit = Circuit(nqubits)
    circuit.add([first, second])
    fused = fusion(circuit, backend=backend)
    expected = [a + b for a, b in zip(first.parameters, second.parameters)]
    expected[nadded:] = first.parameters[nadded:]
    assert fused.ngates == 1
    assert isinstance(fused.queue[0], gate_class)
    assert fused.queue[0].qubits == first.qubits
    backend.assert_allclose(fused.queue[0].parameters, expected)
    backend.assert_allclose(
        fused.unitary(backend), circuit.unitary(backend), atol=1e-12
    )
    # the input circuit and its gates are left untouched
    assert circuit.ngates == 2
    assert circuit.queue[0] is first
    assert circuit.queue[0].parameters == first.parameters

    # the second gate lists the qubits in reversed order
    if nqubits == 2 and not first.control_qubits:
        circuit = Circuit(2)
        circuit.add([first, gate_class(1, 0, *angles[1], *equal)])
        fused = fusion(circuit, backend=backend)
        assert fused.ngates == (2 if reversal is None else 1)
        if reversal is not None:
            assert fused.queue[0].qubits == (0, 1)
        backend.assert_allclose(
            fused.unitary(backend), circuit.unitary(backend), atol=1e-12
        )

    # both gates are controlled on further qubits
    if not first.control_qubits:
        controls = (nqubits, nqubits + 1)
        circuit = Circuit(nqubits + 2)
        circuit.add(
            gate_class(*qubits, *angles[i], *equal).controlled_by(*controls)
            for i in (0, 1)
        )
        fused = fusion(circuit, backend=backend)
        assert fused.ngates == 1
        assert fused.queue[0].control_qubits == controls
        backend.assert_allclose(
            fused.unitary(backend), circuit.unitary(backend), atol=1e-12
        )

    # parameters that must coincide, up to the tolerance ``atol``
    if nequal:
        circuit = Circuit(nqubits)
        circuit.add([first, gate_class(*qubits, *angles[1], *(equal + 1e-14))])
        assert fusion(circuit, backend=backend).ngates == 1
        circuit = Circuit(nqubits)
        circuit.add([first, gate_class(*qubits, *angles[1], *(equal + 1e-3))])
        assert fusion(circuit, backend=backend).ngates == 2
        assert ParametrizedGateFusion(atol=1e-2)(circuit, backend=backend).ngates == 1


# Pairs of U2 and U3 gates: a number means that many random angles, so 2 is a U2 gate
# and 3 is a U3 gate, and a tuple gives the angles of the gate
U_PAIRS = [
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    # the polar angle of the result is pi
    ((0.9, 0.3), (0.2, -0.9)),
    # angles far outside of one period
    ((40.0, -30.0, 20.0), (-25.0, 10.0, 60.0)),
]


@pytest.mark.parametrize("controls", [(), (1,), (1, 2)])
@pytest.mark.parametrize("first,second", U_PAIRS)
def test_parametrized_gate_fusion_u2_u3(backend, first, second, controls):
    circuit = Circuit(1 + len(controls))
    for angles in (first, second):
        if isinstance(angles, int):
            angles = backend.random_uniform(-7, 7, angles)
        circuit.add(U_GATES[len(angles)](0, *angles).controlled_by(*controls))
    fused = ParametrizedGateFusion()(circuit, backend=backend)

    # exact, including the global phase, with a polar angle in [0, pi]
    assert fused.ngates == 1
    assert len(fused.queue[0].parameters) == 3
    assert fused.queue[0].control_qubits == controls
    assert 0 <= fused.queue[0].parameters[0] <= np.pi + 1e-12
    backend.assert_allclose(
        fused.unitary(backend), circuit.unitary(backend), atol=1e-12
    )


@pytest.mark.parametrize("controls", [(), (1,), (1, 2)])
@pytest.mark.parametrize("u1_first", [True, False])
@pytest.mark.parametrize("other", [2, 3])
def test_parametrized_gate_fusion_u1_phase(backend, other, u1_first, controls):
    angles = backend.random_uniform(-3, 3, other)
    rotation = U_GATES[other](0, *angles).controlled_by(*controls)
    nqubits = 1 + len(controls)

    # merging a U1 gate changes the global phase by half of its angle, which is zero
    # only when the angle is a multiple of four pi
    for angle, exact in [
        (0.8, False),
        (-0.8, False),
        (2 * np.pi, False),
        (4 * np.pi, True),
        (4 * np.pi + 1e-9, False),
    ]:
        u1 = gates.U1(0, angle).controlled_by(*controls)
        circuit = Circuit(nqubits)
        circuit.add([u1, rotation] if u1_first else [rotation, u1])

        # by default only exact merges are done
        assert ParametrizedGateFusion()(circuit, backend=backend).ngates == (
            1 if exact else 2
        )

        # the global phase of gates with control qubits is a relative phase
        fused = ParametrizedGateFusion(up_to_global_phase=True)(
            circuit, backend=backend
        )
        if exact or not controls:
            assert fused.ngates == 1
            # a U2 gate stays a U2 gate, and a U3 gate stays a U3 gate
            assert len(fused.queue[0].parameters) == other
            ratio = backend.to_numpy(circuit.unitary(backend)) @ np.linalg.inv(
                backend.to_numpy(fused.unitary(backend))
            )
            backend.assert_allclose(
                ratio, np.exp(1j * angle / 2) * np.eye(2**nqubits), atol=1e-12
            )
        else:
            assert fused.ngates == 2


def test_parametrized_gate_fusion_runs_and_blocking(backend):
    circuit = Circuit(4)
    circuit.add(gates.RX(0, 0.1))
    circuit.add(gates.RX(0, 0.2))
    circuit.add(gates.RX(0, 0.3))  # a run of three becomes one gate
    circuit.add(gates.RZ(1, 0.4))
    circuit.add(gates.RY(2, 0.5))
    circuit.add(gates.RZ(1, 0.6))  # a gate on another qubit does not block
    circuit.add(gates.RZZ(1, 2, 0.7))
    circuit.add(gates.RZ(1, 0.8))  # the RZZ gate acts on qubit 1 and blocks
    circuit.add(gates.RZ(1, 0.9))
    circuit.add(gates.RX(0, 1.0))  # merges with the first run across other qubits
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(0, 1.1))  # the CNOT gate acts on qubit 0 and blocks
    circuit.add(gates.RY(2, 1.2))
    circuit.add(Barrier(2))
    circuit.add(gates.RY(2, 1.3))  # the barrier blocks
    circuit.add(gates.RY(2, 1.4))
    circuit.add(gates.M(2))
    circuit.add(gates.U3(3, 0.3, 1.2, -0.7))
    circuit.add(gates.U2(3, 0.4, 2.5))
    circuit.add(gates.U3(3, -1.3, 0.2, 0.9))  # a run of U gates becomes one U3 gate
    circuit.add(gates.U2(3, 3.1, -0.5))
    fused = ParametrizedGateFusion()(circuit, backend=backend)

    expected = [
        ("rx", (0,), (1.6,)),
        ("rz", (1,), (1.0,)),
        ("ry", (2,), (0.5,)),
        ("rzz", (1, 2), (0.7,)),
        ("rz", (1,), (1.7,)),
        ("cx", (0, 1), ()),
        ("rx", (0,), (1.1,)),
        ("ry", (2,), (1.2,)),
        ("barrier", (2,), ()),
        ("ry", (2,), (2.7,)),
        ("measure", (2,), ()),
        ("u3", (3,), None),
    ]
    assert len(fused.queue) == len(expected)
    for gate, (name, qubits, parameters) in zip(fused.queue, expected):
        assert (gate.name, gate.qubits) == (name, qubits)
        if parameters is not None:
            backend.assert_allclose(gate.parameters, parameters)

    unitary_input, unitary_output = Circuit(4), Circuit(4)
    for source, target in ((circuit, unitary_input), (fused, unitary_output)):
        target.add([g for g in source.queue if not isinstance(g, (Barrier, gates.M))])
    backend.assert_allclose(
        unitary_output.unitary(backend), unitary_input.unitary(backend), atol=1e-12
    )


NOT_FUSED = {
    "different classes with the same generator": [gates.RZ(0, 0.1), gates.U1(0, 0.2)],
    "U3 and RZ": [gates.U3(0, 0.3, 0.2, 0.1), gates.RZ(0, 0.4)],
    "GPI2 gates have no rule": [gates.GPI2(0, 0.1), gates.GPI2(0, 0.2)],
    "GPI gates have no rule": [gates.GPI(0, 0.1), gates.GPI(0, 0.2)],
    "different trainable flags": [
        gates.RX(0, 0.1, trainable=False),
        gates.RX(0, 0.2),
    ],
    "different trainable flags of U gates": [
        gates.U3(0, 0.3, 0.2, 0.1, trainable=False),
        gates.U2(0, 0.4, 0.5),
    ],
    "sympy parameter in the first gate": [
        gates.RX(0, sympy.Symbol("x")),
        gates.RX(0, 0.2),
    ],
    "sympy parameter in the second gate": [
        gates.RX(0, 0.2),
        gates.RX(0, sympy.Symbol("x")),
    ],
    "sympy parameter in a U gate": [
        gates.U3(0, sympy.Symbol("x"), 0.2, 0.1),
        gates.U2(0, 0.4, 0.5),
    ],
    "another gate in between": [gates.RX(0, 0.1), gates.RY(0, 0.2), gates.RX(0, 0.3)],
    "another gate in between U gates": [
        gates.U3(0, 0.3, 0.2, 0.1),
        gates.RX(0, 0.2),
        gates.U2(0, 0.4, 0.5),
    ],
    "different qubits": [gates.U3(0, 0.3, 0.2, 0.1), gates.U2(1, 0.4, 0.5)],
    "different control qubits": [
        gates.RX(0, 0.3).controlled_by(1),
        gates.RX(0, 0.4).controlled_by(2),
    ],
    "different control qubits of U gates": [
        gates.U3(0, 0.3, 0.2, 0.1).controlled_by(1),
        gates.U2(0, 0.4, 0.5).controlled_by(2),
    ],
    "only one gate controlled": [
        gates.RX(0, 0.3).controlled_by(1, 2),
        gates.RX(0, 0.4),
    ],
    "same qubits but controls and targets exchanged": [
        gates.RXX(0, 1, 0.1).controlled_by(2),
        gates.RXX(0, 2, 0.2).controlled_by(1),
    ],
}


# Pairs of U3 gates where the second gate undoes the first one, so the merged gate is
# the identity, and a pair with tiny polar angles that cancel
U_IDENTITY_PAIRS = [
    ((0.7, 0.4, -1.3), (-0.7, 1.3, -0.4)),
    ((1e-9, 0.4, 0.5), (-1e-9, -0.5, -0.4)),
]


@pytest.mark.parametrize("controls", [(), (1,), (1, 2)])
@pytest.mark.parametrize("first,second", U_IDENTITY_PAIRS)
def test_parametrized_gate_fusion_identity_pair(backend, first, second, controls):
    circuit = Circuit(1 + len(controls))
    for angles in (first, second):
        circuit.add(gates.U3(0, *angles).controlled_by(*controls))
    fused = ParametrizedGateFusion()(circuit, backend=backend)

    # a polar angle of zero with a vanishing phase gives an exact U1 gate
    assert [gate.name for gate in fused.queue] == [
        "cu1" if len(controls) == 1 else "u1"
    ]
    assert fused.queue[0].control_qubits == controls
    backend.assert_allclose(
        fused.unitary(backend), circuit.unitary(backend), atol=1e-12
    )


@pytest.mark.parametrize("controls", [(), (1,), (1, 2)])
def test_parametrized_gate_fusion_u2_result(backend, controls):
    # the polar angles add up to pi / 2 because lambda_2 + phi_1 = 0
    circuit = Circuit(1 + len(controls))
    circuit.add(gates.U3(0, np.pi / 4, 0.2, 0.3).controlled_by(*controls))
    circuit.add(gates.U3(0, np.pi / 4, 0.5, -0.2).controlled_by(*controls))
    fused = ParametrizedGateFusion()(circuit, backend=backend)

    assert fused.ngates == 1
    assert fused.queue[0].name == ("cu2" if len(controls) == 1 else "u2")
    assert fused.queue[0].control_qubits == controls
    backend.assert_allclose(
        fused.unitary(backend), circuit.unitary(backend), atol=1e-12
    )


@pytest.mark.parametrize("controls", [(), (1,)])
@pytest.mark.parametrize("up_to_global_phase", [False, True])
def test_parametrized_gate_fusion_u1_result(backend, up_to_global_phase, controls):
    # the polar angles cancel because lambda_2 + phi_1 = pi
    circuit = Circuit(1 + len(controls))
    circuit.add(gates.U3(0, 0.5, 0.2, 0.3).controlled_by(*controls))
    circuit.add(gates.U3(0, 0.5, 0.4, np.pi - 0.2).controlled_by(*controls))
    fused = ParametrizedGateFusion(up_to_global_phase=up_to_global_phase)(
        circuit, backend=backend
    )

    assert fused.ngates == 1
    original = backend.to_numpy(circuit.unitary(backend))
    result = backend.to_numpy(fused.unitary(backend))
    if up_to_global_phase and not controls:
        # U1 differs from the U3 gate with a polar angle of zero by a global phase
        assert fused.queue[0].name == "u1"
        assert np.isclose(np.abs(np.trace(original.conj().T @ result)), 2)
        assert not np.allclose(original, result)
    else:
        assert fused.queue[0].name == ("cu3" if len(controls) == 1 else "u3")
        np.testing.assert_allclose(result, original, atol=1e-12)


@pytest.mark.parametrize("pair", NOT_FUSED.values(), ids=NOT_FUSED.keys())
def test_parametrized_gate_fusion_not_fused(backend, pair):
    circuit = Circuit(3)
    circuit.add(pair)
    fused = ParametrizedGateFusion(up_to_global_phase=True)(circuit, backend=backend)
    assert [gate.name for gate in fused.queue] == [gate.name for gate in pair]


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize(
    "pair",
    [
        ((gates.RX, (0.1,)), (gates.RX, (0.2,))),
        ((gates.U3, (0.3, 0.2, 0.1)), (gates.U2, (0.4, 0.5))),
        ((gates.U1, (0.3,)), (gates.U2, (0.4, 0.5))),
    ],
    ids=["same class", "U3 and U2", "U1 and U2"],
)
def test_parametrized_gate_fusion_trainable(backend, pair, trainable):
    circuit = Circuit(1, density_matrix=True)
    circuit.add(
        gate_class(0, *angles, trainable=trainable) for gate_class, angles in pair
    )
    fused = ParametrizedGateFusion(up_to_global_phase=True)(circuit, backend=backend)

    # the merged gate keeps the flag, and it holds fewer parameters than the two gates
    assert fused.ngates == 1
    assert fused.density_matrix
    assert fused.queue[0].trainable == trainable
    assert len(circuit.get_parameters()) == 2 * trainable
    assert len(fused.get_parameters()) == trainable


def test_parametrized_gate_fusion_negative_atol():
    with pytest.raises(ValueError):
        ParametrizedGateFusion(atol=-1e-3)


def test_parametrized_gate_fusion_pipeline(backend, star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.RY(0, 0.1))
    circuit.add(gates.RY(0, 0.2))
    circuit.add(gates.CNOT(0, 2))
    pipeline = Passes(
        [InverseCancellation(), ParametrizedGateFusion()],
        connectivity=star_connectivity(),
    )
    transpiled, _ = pipeline(circuit, backend=backend)
    assert [gate.name for gate in transpiled.queue] == ["ry", "cx"]
    backend.assert_allclose(transpiled.queue[0].parameters, (0.3,))


def test_remove_identity_equivalent(backend):
    circuit = Circuit(2)
    circuit.add(gates.I(0))
    circuit.add(gates.RX(0, 0.0))
    circuit.add(gates.RZ(1, 2 * np.pi))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 1))
    reduced = RemoveIdentityEquivalent()(circuit, backend=backend)
    assert [gate.name for gate in reduced.queue] == ["cx", "measure"]
    assert reduced.nqubits == circuit.nqubits


def test_remove_identity_equivalent_approximation_degree(backend):
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 1e-3))
    assert RemoveIdentityEquivalent()(circuit, backend=backend).ngates == 1
    reduced = RemoveIdentityEquivalent(approximation_degree=1 - 1e-6)(
        circuit, backend=backend
    )
    assert reduced.ngates == 0

    with pytest.raises(ValueError):
        RemoveIdentityEquivalent(approximation_degree=1.5)


def test_remove_identity_equivalent_controlled(backend):
    circuit = Circuit(2)
    circuit.add(gates.CRZ(0, 1, 2 * np.pi))
    circuit.add(gates.CRX(0, 1, 0.0))
    reduced = RemoveIdentityEquivalent()(circuit, backend=backend)
    assert [gate.name for gate in reduced.queue] == ["crz"]


def test_remove_identity_equivalent_pipeline(backend, star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.RX(0, 0.0))
    circuit.add(gates.CNOT(0, 2))
    pipeline = Passes([RemoveIdentityEquivalent()], connectivity=star_connectivity())
    transpiled, _ = pipeline(circuit, backend=backend)
    assert [gate.name for gate in transpiled.queue] == ["cx"]


@pytest.mark.parametrize(
    "gate",
    [
        gates.CCZ(0, 1, 2),
        gates.CRZ(0, 1, 0.3),
        gates.CU1(0, 1, 0.3),
        gates.CZ(0, 1),
        gates.RZ(0, 0.3),
        gates.RZZ(0, 1, 0.3),
        gates.S(0),
        gates.S(1).controlled_by(0),
        gates.SDG(0),
        gates.T(0),
        gates.T(3).controlled_by(0, 1, 2),
        gates.TDG(0),
        gates.U1(0, 0.3),
        gates.Z(0),
    ],
    ids=lambda gate: f"{gate.name}-{len(gate.qubits)}",
)
def test_remove_diagonal_gates_before_measure(backend, gate):
    nqubits = 4
    circuit = Circuit(nqubits)
    circuit.add(gates.RY(qubit, 0.4 + qubit) for qubit in range(nqubits))
    circuit.add(gate)
    circuit.add(gates.M(*range(nqubits)))

    reduced = RemoveDiagonalGatesBeforeMeasurement()(circuit)
    assert [gate.name for gate in reduced.queue] == ["ry"] * nqubits + ["measure"]
    assert reduced.nqubits == circuit.nqubits
    backend.assert_allclose(
        backend.execute_circuit(reduced).probabilities(),
        backend.execute_circuit(circuit).probabilities(),
        atol=1e-12,
    )


def test_remove_diagonal_gates_before_measure_kept():
    circuit = Circuit(8, density_matrix=True)
    # a gate follows the diagonal gate
    circuit.add(gates.Z(0))
    circuit.add(gates.H(0))
    # the gate is not diagonal
    circuit.add(gates.X(1))
    # only one of the two qubits of the diagonal gate is measured next
    circuit.add(gates.CZ(2, 3))
    circuit.add(gates.H(3))
    # a barrier separates the diagonal gate from the measurement
    circuit.add(gates.RZ(4, 0.3))
    circuit.add(gates.Barrier(4))
    # the measurement is preceded by the rotation to the X basis
    circuit.add(gates.T(5))
    circuit.add(gates.M(5, basis=gates.X))
    # only the gate directly before the measurement is removed
    circuit.add(gates.RZ(6, 0.3))
    circuit.add(gates.Z(6))
    # a noise channel separates the diagonal gate from the measurement
    circuit.add(gates.S(7))
    circuit.add(gates.ResetChannel(7, [0.5, 0.5]))
    circuit.add(gates.M(0, 1, 2, 3, 4, 6, 7))

    reduced = RemoveDiagonalGatesBeforeMeasurement()(circuit)
    assert [gate.name for gate in reduced.queue] == [
        "z",
        "h",
        "x",
        "cz",
        "h",
        "rz",
        "barrier",
        "t",
        "h",
        "measure",
        "rz",
        "s",
        "ResetChannel",
        "measure",
    ]

    reduced = RemoveDiagonalGatesBeforeMeasurement()(reduced)
    assert reduced.ngates == circuit.ngates - 2


def test_remove_diagonal_gates_before_measure_pipeline(backend, star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.RY(0, 0.5))
    circuit.add(gates.RZ(0, 0.3))
    circuit.add(gates.M(0))
    pipeline = Passes(
        [RemoveDiagonalGatesBeforeMeasurement()], connectivity=star_connectivity()
    )
    transpiled, _ = pipeline(circuit, backend=backend)
    assert [gate.name for gate in transpiled.queue] == ["ry", "measure"]


def test_remove_final_reset():
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    circuit.add(gates.H(0))
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    circuit.add(gates.ResetChannel(1, [1.0, 0.0]))
    circuit.add(gates.Barrier(1))
    circuit.add(gates.ResetChannel(2, [0.5, 0.0]))
    circuit.add(gates.M(2))
    reduced = RemoveFinalReset()(circuit)
    assert [gate.name for gate in reduced.queue] == [
        "ResetChannel",
        "h",
        "ResetChannel",
        "barrier",
        "ResetChannel",
        "measure",
    ]
    assert reduced.nqubits == circuit.nqubits


def test_remove_reset_in_zero_state():
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    circuit.add(gates.H(0))
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    circuit.add(gates.ResetChannel(1, [1.0, 0.0]))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.ResetChannel(2, [1.0, 0.0]))
    circuit.add(gates.ResetChannel(1, [0.5, 0.5]))
    reduced = RemoveResetInZeroState()(circuit)
    assert [gate.name for gate in reduced.queue] == [
        "h",
        "ResetChannel",
        "cx",
        "ResetChannel",
        "ResetChannel",
    ]


def test_reset_after_measure_simplification(backend):
    circuit = Circuit(2, density_matrix=True)
    circuit.add(gates.X(1))
    circuit.add(gates.M(1, 0))
    circuit.add(gates.ResetChannel(1, [1.0, 0.0]))
    circuit.add(gates.ResetChannel(1, [1.0, 0.0]))
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    simplified = ResetAfterMeasureSimplification()(circuit)
    assert [gate.name for gate in simplified.queue] == [
        "x",
        "measure",
        "u3",
        "ResetChannel",
        "u3",
    ]

    target = backend.zero_state(2, density_matrix=True)
    for transpiled in (circuit, simplified):
        state = backend.execute_circuit(transpiled).state()
        backend.assert_allclose(state, target, atol=1e-8)

    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    unchanged = ResetAfterMeasureSimplification()(circuit)
    assert [gate.name for gate in unchanged.queue] == ["h", "ResetChannel"]


def test_reset_passes_pipeline(backend, star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.ResetChannel(0, [1.0, 0.0]))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.ResetChannel(2, [1.0, 0.0]))
    pipeline = Passes(
        [RemoveResetInZeroState(), RemoveFinalReset()],
        connectivity=star_connectivity(),
    )
    transpiled, _ = pipeline(circuit, backend=backend)
    assert [gate.name for gate in transpiled.queue] == ["cx"]


BASES = {
    "U3": ["u3", "cx"],
    "U321": ["u1", "u2", "u3", "cx"],
    "ZYZ": ["rz", "ry", "cx"],
    "ZXZ": ["rz", "rx", "cx"],
    "XZX": ["rx", "rz", "cx"],
    "XYX": ["rx", "ry", "cx"],
    "ZSX": ["rz", "sx", "cx"],
    "ZSXX": ["rz", "sx", "x", "cx"],
}


@pytest.mark.parametrize("basis", BASES.values(), ids=BASES.keys())
def test_optimize_1q_gates_decomposition_bases(backend, basis):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.T(0))
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.U3(0, 0.4, 0.5, 0.6))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.S(1))
    circuit.add(gates.SX(1))
    circuit.add(gates.Y(1))
    circuit.add(gates.RZ(0, 0.2))
    optimized = Optimize1qGatesDecomposition(basis)(circuit, backend=backend)
    assert {gate.name for gate in optimized.queue} <= set(basis)

    original = backend.to_numpy(circuit.unitary(backend))
    result = backend.to_numpy(optimized.unitary(backend))
    assert np.isclose(np.abs(np.trace(original.conj().T @ result)), 4)


@pytest.mark.parametrize("basis", [None, *BASES.values()])
def test_optimize_1q_gates_decomposition_random(backend, basis):
    circuit = Circuit(2)
    for qubit in [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]:
        circuit.add(gates.U3(qubit, *backend.random_uniform(-7, 7, 3)))
        circuit.add(gates.RX(qubit, float(backend.random_uniform(-7, 7, 1)[0])))
        circuit.add(gates.H(qubit))
    optimized = Optimize1qGatesDecomposition(basis)(circuit, backend=backend)
    assert optimized.ngates <= circuit.ngates

    original = backend.to_numpy(circuit.unitary(backend))
    result = backend.to_numpy(optimized.unitary(backend))
    assert np.isclose(np.abs(np.trace(original.conj().T @ result)), 4)


@pytest.mark.parametrize(
    "basis,theta,names",
    [
        ("ZSX", 0.0, ["rz"]),
        ("ZSX", np.pi / 2, ["rz", "sx", "rz"]),
        ("ZSX", 0.7, ["rz", "sx", "rz", "sx", "rz"]),
        ("ZSXX", np.pi, ["rz", "x", "rz"]),
        ("ZSXX", 0.7, ["rz", "sx", "rz", "sx", "rz"]),
        ("U321", 0.0, ["u1"]),
        ("U321", np.pi / 2, ["u2"]),
        ("U321", 0.7, ["u3"]),
    ],
)
def test_optimize_1q_gates_decomposition_special_angles(backend, basis, theta, names):
    circuit = Circuit(1)
    circuit.add(gates.U3(0, theta, 0.4, 0.9))
    circuit.add(gates.RZ(0, 0.1))
    optimized = Optimize1qGatesDecomposition(
        [name for name in BASES[basis] if name != "cx"]
    )(circuit, backend=backend)
    assert [gate.name for gate in optimized.queue] == names


def test_optimize_1q_gates_decomposition_identity(backend):
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.H(0))
    for basis in (None, *BASES.values()):
        for up_to_global_phase in (True, False):
            optimized = Optimize1qGatesDecomposition(
                basis, up_to_global_phase=up_to_global_phase
            )(circuit, backend=backend)
            assert optimized.ngates == 0


def test_optimize_1q_gates_decomposition_only_replaces_if_needed(backend):
    circuit = Circuit(1)
    circuit.add(gates.RZ(0, 0.1))
    circuit.add(gates.SX(0))
    circuit.add(gates.RZ(0, 0.3))

    # already in the basis and not improvable
    optimized = Optimize1qGatesDecomposition(["rz", "sx"])(circuit, backend=backend)
    assert all(new is old for new, old in zip(optimized.queue, circuit.queue))

    # out of the basis
    optimized = Optimize1qGatesDecomposition(["u3"])(circuit, backend=backend)
    assert [gate.name for gate in optimized.queue] == ["u3"]

    # without a basis, all gates count as in the basis
    optimized = Optimize1qGatesDecomposition()(circuit, backend=backend)
    assert [gate.name for gate in optimized.queue] == ["u2"]

    circuit = Circuit(1)
    circuit.add(gates.RZ(0, 0.1))
    circuit.add(gates.RZ(0, 0.3))
    optimized = Optimize1qGatesDecomposition(["rz", "sx"])(circuit, backend=backend)
    assert [gate.name for gate in optimized.queue] == ["rz"]
    assert np.isclose(optimized.queue[0].parameters[0], 0.4)


def test_optimize_1q_gates_decomposition_global_phase(backend):
    circuit = Circuit(1)
    circuit.add(gates.H(0))

    # H has determinant -1, so it is not a product of rotations
    exact = Optimize1qGatesDecomposition(["rz", "ry"], up_to_global_phase=False)(
        circuit, backend=backend
    )
    assert [gate.name for gate in exact.queue] == ["h"]
    optimized = Optimize1qGatesDecomposition(["rz", "ry"])(circuit, backend=backend)
    assert {gate.name for gate in optimized.queue} == {"rz", "ry"}

    # rotations have determinant one, so the result is exact
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.RY(0, 0.4))
    circuit.add(gates.RZ(0, 0.5))
    circuit.add(gates.RX(0, 0.6))
    exact = Optimize1qGatesDecomposition(["u3"], up_to_global_phase=False)(
        circuit, backend=backend
    )
    assert [gate.name for gate in exact.queue] == ["u3"]
    backend.assert_allclose(
        exact.unitary(backend), circuit.unitary(backend), atol=1e-12
    )


def test_optimize_1q_gates_decomposition_unusable_basis(backend):
    circuit = Circuit(1)
    circuit.add(gates.H(0))
    circuit.add(gates.T(0))
    optimized = Optimize1qGatesDecomposition(["cz", "gpi2"])(circuit, backend=backend)
    assert [gate.name for gate in optimized.queue] == ["h", "t"]


def test_optimize_1q_gates_decomposition_stops_at_other_gates(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.T(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.H(0))
    circuit.add(gates.RX(0, sympy.Symbol("x")))
    circuit.add(gates.T(0))
    circuit.add(gates.H(0).controlled_by(1))
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    circuit.add(gates.T(0))
    circuit.add(gates.H(0))
    optimized = Optimize1qGatesDecomposition(["u3", "cx"])(circuit, backend=backend)
    assert [(gate.name, gate.qubits) for gate in optimized.queue] == [
        ("u3", (0,)),
        ("cx", (0, 1)),
        ("u3", (0,)),
        ("rx", (0,)),
        ("u3", (0,)),
        ("h", (1, 0)),
        ("u3", (0,)),
        ("measure", (0,)),
        ("u3", (0,)),
    ]


def test_optimize_1q_gates_decomposition_errors(backend):
    with pytest.raises(ValueError):
        Optimize1qGatesDecomposition(atol=-1e-3)


def test_optimize_1q_gates_decomposition_pipeline(backend, star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 2))
    pipeline = Passes(
        [Optimize1qGatesDecomposition(["rz", "sx", "cx"])],
        connectivity=star_connectivity(),
    )
    transpiled, _ = pipeline(circuit, backend=backend)
    assert [gate.name for gate in transpiled.queue] == ["cx"]


@pytest.mark.parametrize(
    "basis,gate",
    [
        (["rz", "ry"], gates.RZ),
        (["rz", "rx"], gates.RZ),
        (["rx", "ry"], gates.RX),
        (["rx", "rz"], gates.RX),
    ],
)
def test_optimize_1q_gates_decomposition_merges_rotations(backend, basis, gate):
    circuit = Circuit(1)
    circuit.add(gate(0, 0.1))
    circuit.add(gate(0, 0.3))
    optimized = Optimize1qGatesDecomposition(basis)(circuit, backend=backend)
    assert [type(new) for new in optimized.queue] == [gate]
    assert np.isclose(optimized.queue[0].parameters[0], 0.4)


@pytest.mark.parametrize(
    "special,density_matrix",
    [
        (lambda: Barrier(0), False),
        (lambda: gates.Align(0), False),
        (lambda: gates.PauliNoiseChannel(0, [("X", 0.1)]), True),
    ],
    ids=["barrier", "align", "channel"],
)
def test_optimize_1q_gates_decomposition_special_gates(
    backend, special, density_matrix
):
    # the special gate only stops the runs on its own qubit, so the gates on the
    # second qubit are merged across it
    circuit = Circuit(2, density_matrix=density_matrix)
    circuit.add(gates.H(1))
    circuit.add(gates.H(0))
    circuit.add(gates.T(0))
    circuit.add(special())
    circuit.add(gates.H(0))
    circuit.add(gates.T(0))
    circuit.add(gates.T(1))
    optimized = Optimize1qGatesDecomposition(["u3", "cx"])(circuit, backend=backend)

    assert [(type(gate), gate.qubits) for gate in optimized.queue] == [
        (gates.U3, (1,)),
        (gates.U3, (0,)),
        (type(circuit.queue[3]), (0,)),
        (gates.U3, (0,)),
    ]
    assert optimized.queue[2] is circuit.queue[3]
    assert optimized.density_matrix == density_matrix


def test_optimize_1q_gates_decomposition_keeps_order(backend):
    circuit = Circuit(3)
    circuit.add(gates.X(2))
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.T(0))
    circuit.add(gates.Y(1))
    circuit.add(gates.H(1))
    circuit.add(gates.Z(2))
    optimized = Optimize1qGatesDecomposition(["u3", "cx"])(circuit, backend=backend)

    # a run stays at the place of its first gate
    assert [(gate.name, gate.qubits) for gate in optimized.queue] == [
        ("u3", (2,)),
        ("u3", (0,)),
        ("cx", (1, 2)),
        ("u3", (1,)),
        ("u3", (2,)),
    ]


def test_optimize_1q_gates_decomposition_repeated_gate(backend):
    # the same gate object added several times
    gate = gates.H(0)
    circuit = Circuit(2)
    circuit.add(gate)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gate)
    optimized = Optimize1qGatesDecomposition(["u3", "cx"])(circuit, backend=backend)

    assert [(gate.name, gate.qubits) for gate in optimized.queue] == [
        ("u3", (0,)),
        ("cx", (0, 1)),
        ("u3", (0,)),
    ]
    original = backend.to_numpy(circuit.unitary(backend))
    result = backend.to_numpy(optimized.unitary(backend))
    assert np.isclose(np.abs(np.trace(original.conj().T @ result)), 4)


def test_optimize_1q_gates_decomposition_circuit_unchanged(backend):
    circuit = Circuit(2, density_matrix=True)
    circuit.add(gates.H(0))
    circuit.add(gates.T(0))
    circuit.add(gates.CNOT(0, 1))
    queue = list(circuit.queue)
    optimized = Optimize1qGatesDecomposition(["u3", "cx"])(circuit, backend=backend)

    assert list(circuit.queue) == queue
    assert all(isinstance(gate, gates.FusedGate) is False for gate in circuit.queue)
    assert optimized.density_matrix
    assert [gate.name for gate in optimized.queue] == ["u3", "cx"]


def test_consolidate_blocks_equivalence(backend):
    rng = np.random.default_rng(42)
    for _ in range(20):
        nqubits = int(rng.integers(2, 5))
        circuit = Circuit(nqubits)
        for _ in range(int(rng.integers(5, 30))):
            q0, q1 = (int(qubit) for qubit in rng.permutation(nqubits)[:2])
            angle = float(rng.uniform(0, 3))
            circuit.add(
                [
                    gates.H(q0),
                    gates.RX(q0, angle),
                    gates.CZ(q0, q1),
                    gates.CNOT(q0, q1),
                    gates.SWAP(q0, q1),
                    gates.iSWAP(q0, q1),
                    gates.RZZ(q0, q1, angle),
                ][int(rng.integers(7))]
            )
        consolidated = ConsolidateBlocks()(circuit, backend=backend)
        backend.assert_allclose(
            consolidated.unitary(backend), circuit.unitary(backend), atol=1e-8
        )


def test_consolidate_blocks_fewer_cz_gates(backend):
    circuit = Circuit(2)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    consolidated = ConsolidateBlocks()(circuit, backend=backend)
    assert [gate.name for gate in consolidated.queue if len(gate.qubits) == 2] == [
        "cz",
        "cz",
    ]
    backend.assert_allclose(
        consolidated.unitary(backend), circuit.unitary(backend), atol=1e-8
    )

    # three CNOT gates with the same control and target qubits are only one CNOT
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 0))
    circuit.add(gates.CNOT(0, 1))
    consolidated = ConsolidateBlocks()(circuit, backend=backend)
    assert sum(len(gate.qubits) == 2 for gate in consolidated.queue) == 3
    backend.assert_allclose(
        consolidated.unitary(backend), circuit.unitary(backend), atol=1e-8
    )


def test_consolidate_blocks_single_qubit_gates_join_block(backend):
    circuit = Circuit(3)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.H(2))
    circuit.add(gates.CZ(1, 0))
    circuit.add(gates.RX(1, 0.4))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.H(1))
    consolidated = ConsolidateBlocks()(circuit, backend=backend)
    backend.assert_allclose(
        consolidated.unitary(backend), circuit.unitary(backend), atol=1e-8
    )
    assert sum(len(gate.qubits) == 2 for gate in consolidated.queue) <= 5


def test_consolidate_blocks_unchanged(backend):
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RX(1, 0.3))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.SWAP(0, 1))
    consolidated = ConsolidateBlocks()(circuit, backend=backend)
    assert consolidated.queue == circuit.queue


def test_consolidate_blocks_uncollectable_gates(backend):
    x = sympy.Symbol("x")
    circuit = Circuit(3, density_matrix=True)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.DepolarizingChannel((0,), 0.1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.SWAP(1, 2))
    circuit.add(gates.RX(2, x))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(Barrier(0, 1, 2))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.M(0, 1))
    consolidated = ConsolidateBlocks()(circuit, backend=backend)
    assert consolidated.queue == circuit.queue
    assert consolidated.density_matrix


def test_consolidate_blocks_in_passes(backend):
    circuit = Circuit(2)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.CNOT(0, 1))
    pipeline = Passes([ConsolidateBlocks()])
    transpiled, _ = pipeline(circuit, backend=backend)
    assert sum(len(gate.qubits) == 2 for gate in transpiled.queue) == 2
    backend.assert_allclose(
        transpiled.unitary(backend), circuit.unitary(backend), atol=1e-8
    )


def test_fixed_point(backend):
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.RX(0, 0.4))
    circuit.add(gates.RX(0, -0.7))
    circuit.add(gates.CZ(0, 1))
    optimizers = [InverseCancellation(), Optimize1qGatesDecomposition()]

    assert FixedPoint(optimizers, max_iterations=1)(circuit, backend).ngates == 2
    assert FixedPoint(optimizers)(circuit, backend).ngates == 0
    # the optimizers alone stop at the first iteration
    assert (
        Optimize1qGatesDecomposition()(
            InverseCancellation()(circuit, backend), backend
        ).ngates
        == 2
    )


def test_fixed_point_stops_when_unchanged(backend):
    class Counter(Optimizer):
        def __init__(self):
            self.calls = 0

        def __call__(self, circuit):
            self.calls += 1
            return circuit

    circuit = Circuit(2)
    circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
    circuit.add(gates.Unitary(backend.matrices.I(4), 0, 1))
    circuit.add(gates.M(0, 1))
    counter = Counter()
    optimized = FixedPoint([counter, counter], max_iterations=5)(circuit, backend)
    assert counter.calls == 2
    assert optimized is circuit


def test_fixed_point_connectivity(backend, star_connectivity):
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 1))
    padder = Preprocessing()
    pipeline = Passes(
        [FixedPoint([padder, InverseCancellation()])],
        connectivity=star_connectivity(),
    )
    transpiled, _ = pipeline(circuit, backend=backend)
    assert transpiled.nqubits == 5
    assert padder.connectivity is pipeline.connectivity


def test_fixed_point_errors():
    with pytest.raises(ValueError):
        FixedPoint([InverseCancellation()], max_iterations=0)


def test_inverse_cancellation_commutation(backend):
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RZ(0, 0.3))
    circuit.add(gates.X(1))
    circuit.add(gates.CNOT(0, 1))
    reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
    assert [gate.name for gate in reduced.queue] == ["rz", "x"]
    backend.assert_allclose(
        reduced.unitary(backend), circuit.unitary(backend), atol=1e-12
    )
    # the default does not look through the gates in between
    assert InverseCancellation()(circuit, backend=backend).ngates == 4


def test_inverse_cancellation_commutation_blocked(backend):
    # the Hadamard gate on the target and the rotation around X on the control do
    # not commute with the CNOT gate
    for blocker in (gates.H(1), gates.RX(0, 0.3)):
        circuit = Circuit(2)
        circuit.add(gates.CNOT(0, 1))
        circuit.add(blocker)
        circuit.add(gates.CNOT(0, 1))
        reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
        assert reduced.ngates == 3

    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.M(0))
    circuit.add(gates.CZ(0, 1))
    reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
    assert reduced.ngates == 3

    circuit = Circuit(3)
    circuit.add(gates.CZ(0, 1))
    circuit.add(Barrier(1, 2))
    circuit.add(gates.CZ(0, 1))
    reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
    assert reduced.ngates == 3

    # a barrier on other qubits does not stop the cancellation
    circuit = Circuit(3)
    circuit.add(gates.CZ(0, 1))
    circuit.add(Barrier(2))
    circuit.add(gates.CZ(0, 1))
    reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
    assert [gate.name for gate in reduced.queue] == ["barrier"]

    circuit = Circuit(2, density_matrix=True)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.DepolarizingChannel((0,), 0.1))
    circuit.add(gates.CZ(0, 1))
    reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
    assert reduced.ngates == 3


def test_inverse_cancellation_commutation_many_qubits(backend):
    # gates in between act on only one of the two qubits of the pair, and the
    # controlled-Z gate with the qubits swapped commutes with the pair
    circuit = Circuit(3)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.T(0))
    circuit.add(gates.CZ(1, 0))
    circuit.add(gates.S(1))
    circuit.add(gates.CZ(0, 1))
    reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
    assert [gate.qubits for gate in reduced.queue] == [(1, 2), (0,), (1, 0), (1,)]
    backend.assert_allclose(
        reduced.unitary(backend), circuit.unitary(backend), atol=1e-12
    )


def test_inverse_cancellation_commutation_random(backend):
    rng = np.random.default_rng(7)
    removed = 0
    for _ in range(30):
        nqubits = int(rng.integers(2, 5))
        circuit = Circuit(nqubits)
        for _ in range(int(rng.integers(5, 30))):
            q0, q1 = (int(qubit) for qubit in rng.permutation(nqubits)[:2])
            circuit.add(
                [
                    gates.H(q0),
                    gates.X(q0),
                    gates.RZ(q0, float(rng.choice([0.3, -0.3]))),
                    gates.CZ(q0, q1),
                    gates.CNOT(q0, q1),
                ][int(rng.integers(5))]
            )
        reduced = InverseCancellation(commutation=True)(circuit, backend=backend)
        backend.assert_allclose(
            reduced.unitary(backend), circuit.unitary(backend), atol=1e-12
        )
        removed += InverseCancellation()(circuit, backend=backend).ngates
        removed -= reduced.ngates
    assert removed > 0


def test_consolidate_blocks_weight(backend):
    circuit = Circuit(2)
    circuit.add([gates.CZ(0, 1), gates.CNOT(0, 1), gates.T(1), gates.SWAP(0, 1)])

    assert ConsolidateBlocks().weight == math.sqrt(2)
    consolidated = ConsolidateBlocks(weight=math.pi)(circuit, backend=backend)
    assert consolidated.ngates != circuit.ngates
    backend.assert_allclose(
        consolidated.unitary(backend), circuit.unitary(backend), atol=1e-8
    )
