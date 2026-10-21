import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler.asserts import assert_decomposition
from qibo.transpiler.decompositions import (
    gpi2_dec,
    iswap_dec,
    native_cz_dec,
    opt_dec,
    u3_dec,
)
from qibo.transpiler.decompositions.abstract import merge_global_phases
from qibo.transpiler.decompositions.direction_decompositions import direction_dec
from qibo.transpiler.decompositions.standard_decompositions import (
    _mottonen_compute_theta,
    standard_decompositions,
)
from qibo.transpiler.unroller import (
    _TWO_QUBIT_GATES,
    NativeGates,
    cz_dec,
    translate_gate,
)

default_natives = NativeGates.Z | NativeGates.RZ | NativeGates.M | NativeGates.I


def assert_matrices_allclose(gate, natives, backend):
    target_matrix = gate.matrix(backend)

    circuit = Circuit(len(gate.qubits))
    circuit.add(translate_gate(gate, natives, backend=backend))
    native_matrix = circuit.unitary(backend)

    # the matrices are equal, including the global phase
    backend.assert_allclose(native_matrix, target_matrix, atol=1e-6)
    assert_decomposition(circuit, natives)


TABLES = {
    "standard": standard_decompositions,
    "cz": cz_dec,
    "iswap": iswap_dec,
    "opt": opt_dec,
    "gpi2": gpi2_dec,
    "u3": u3_dec,
}
GATES_TO_DECOMPOSE = [
    ("H", (0,)),
    ("X", (0,)),
    ("Y", (0,)),
    ("Z", (0,)),
    ("S", (0,)),
    ("SDG", (0,)),
    ("T", (0,)),
    ("TDG", (0,)),
    ("SX", (0,)),
    ("SXDG", (0,)),
    ("RX", (0, 0.37)),
    ("RY", (0, 0.37)),
    ("RZ", (0, 0.37)),
    ("PRX", (0, 0.37, 0.91)),
    ("GPI2", (0, 0.37)),
    ("U1", (0, 0.37)),
    ("U2", (0, 0.37, 0.91)),
    ("U3", (0, 0.37, 0.91, -0.6)),
    ("CNOT", (0, 1)),
    ("CY", (0, 1)),
    ("CZ", (0, 1)),
    ("CH", (0, 1)),
    ("CSX", (0, 1)),
    ("CSXDG", (0, 1)),
    ("CRX", (0, 1, 0.37)),
    ("CRY", (0, 1, 0.37)),
    ("CRZ", (0, 1, 0.37)),
    ("CU1", (0, 1, 0.37)),
    ("CU2", (0, 1, 0.37, 0.91)),
    ("CU3", (0, 1, 0.37, 0.91, -0.6)),
    ("SWAP", (0, 1)),
    ("iSWAP", (0, 1)),
    ("SiSWAP", (0, 1)),
    ("SiSWAPDG", (0, 1)),
    ("FSWAP", (0, 1)),
    ("ECR", (0, 1)),
    ("RXX", (0, 1, 0.37)),
    ("RYY", (0, 1, 0.37)),
    ("RZZ", (0, 1, 0.37)),
    ("RZX", (0, 1, 0.37)),
    ("RXXYY", (0, 1, 0.37)),
    ("GIVENS", (0, 1, 0.37)),
    ("RBS", (0, 1, 0.37)),
    ("fSim", (0, 1, 0.37, 0.91)),
    ("SYC", (0, 1)),
    ("TOFFOLI", (0, 1, 2)),
    ("CCZ", (0, 1, 2)),
    ("DEUTSCH", (0, 1, 2, 0.37)),
]


@pytest.mark.parametrize(
    "table,gate_name,args",
    [
        (table, gate_name, args)
        for table in TABLES
        for gate_name, args in GATES_TO_DECOMPOSE
        if getattr(gates, gate_name) in TABLES[table].decompositions
    ],
)
def test_decompositions_are_exact(backend, table, gate_name, args):
    """The decompositions are equal to the gates, including their global phase."""
    gate = getattr(gates, gate_name)(*args)
    nqubits = len(gate.qubits)

    decomposition = TABLES[table](gate, backend)
    # the global phase gate is unique and the last one, if the decomposition is not exact
    global_phases = [
        i for i, g in enumerate(decomposition) if isinstance(g, gates.GlobalPhase)
    ]
    assert global_phases in ([], [len(decomposition) - 1])
    for i in global_phases:
        assert 0 < decomposition[i].parameters[0] < 2 * math.pi

    circuit = Circuit(nqubits)
    circuit.add(decomposition)

    backend.assert_allclose(circuit.unitary(backend), gate.matrix(backend), atol=1e-8)


@pytest.mark.parametrize("table", ["gpi2", "u3"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_unitary_decompositions_are_exact(backend, table, seed):
    """The U3 gate of a unitary is equal to it up to the phase of its determinant."""
    unitary = random_unitary(2, backend=backend, seed=seed)
    gate = gates.Unitary(unitary, 0)

    circuit = Circuit(1)
    circuit.add(TABLES[table](gate, backend))

    backend.assert_allclose(circuit.unitary(backend), unitary, atol=1e-8)


def test_merge_global_phases():
    decomposition = [
        gates.H(0),
        gates.GlobalPhase(1.0),
        gates.CNOT(0, 1),
        gates.GlobalPhase(2 * math.pi - 0.5),
        gates.X(1),
    ]
    merged = merge_global_phases(decomposition)
    assert [g.name for g in merged] == ["h", "cx", "x", "global_phase"]
    assert merged[-1].parameters[0] == pytest.approx(0.5)

    # phases that cancel modulo 2 pi are not kept, as the decomposition is exact
    decomposition = [gates.GlobalPhase(math.pi), gates.H(0), gates.GlobalPhase(math.pi)]
    assert [g.name for g in merge_global_phases(decomposition)] == ["h"]

    # the phase is reduced to the interval [0, 2 pi)
    merged = merge_global_phases([gates.H(0), gates.GlobalPhase(-math.pi / 4)])
    assert merged[-1].parameters[0] == pytest.approx(7 * math.pi / 4)

    # nothing is changed if there are no global phases
    decomposition = [gates.H(0), gates.X(0)]
    assert merge_global_phases(decomposition) is decomposition


@pytest.mark.parametrize("gate_class", list(native_cz_dec))
def test_native_cz_decompositions_are_exact(backend, gate_class):
    """The decompositions of the CZ gate into the native gates keep its global phase."""
    circuit = Circuit(2)
    circuit.add(native_cz_dec[gate_class])

    backend.assert_allclose(
        circuit.unitary(backend), gates.CZ(0, 1).matrix(backend), atol=1e-8
    )


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize("gate_name", ["H", "X", "Y", "I"])
def test_pauli_to_native(backend, gate_name, natives):
    gate = getattr(gates, gate_name)(0)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize("gate_name", ["RX", "RY", "RZ"])
def test_rotations_to_native(backend, gate_name, natives):
    gate = getattr(gates, gate_name)(0, theta=0.1)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize("gate_name", ["S", "SDG", "T", "TDG", "SX"])
def test_special_single_qubit_to_native(backend, gate_name, natives):
    gate = getattr(gates, gate_name)(0)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_u1_to_native(backend, natives):
    gate = gates.U1(0, theta=0.5)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_u2_to_native(backend, natives):
    gate = gates.U2(0, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_u3_to_native(backend, natives):
    gate = gates.U3(0, theta=0.2, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("natives", [NativeGates.U3, NativeGates.GPI2])
def test_gpi2_to_native(backend, natives):
    gate = gates.GPI2(0, phi=0.123)
    assert_matrices_allclose(gate, natives=natives | default_natives, backend=backend)


@pytest.mark.parametrize("gate_name", ["CNOT", "CY", "CZ", "SWAP", "iSWAP", "FSWAP"])
@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_two_qubit_to_native(backend, gate_name, natives_1q, natives_2q):
    gate = getattr(gates, gate_name)(0, 1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
@pytest.mark.parametrize("gate_name", ["CRX", "CRY", "CRZ"])
def test_controlled_rotations_to_native(backend, gate_name, natives_1q, natives_2q):
    gate = getattr(gates, gate_name)(0, 1, 0.3)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_cu1_to_native(backend, natives_1q, natives_2q):
    gate = gates.CU1(0, 1, theta=0.4)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_cu2_to_native(backend, natives_1q, natives_2q):
    gate = gates.CU2(0, 1, phi=0.2, lam=0.3)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_cu3_to_native(backend, natives_1q, natives_2q):
    gate = gates.CU3(0, 1, theta=0.2, phi=0.3, lam=0.4)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_fSim_to_native(backend, natives_1q, natives_2q):
    gate = gates.fSim(0, 1, theta=0.3, phi=0.1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize("seed", [None, 10])
@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_GeneralizedfSim_to_native(backend, natives_1q, natives_2q, seed):
    unitary = random_unitary(2, seed=seed, backend=backend)
    gate = gates.GeneralizedfSim(0, 1, unitary, phi=0.1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
@pytest.mark.parametrize("gate_name", ["RXX", "RZZ", "RYY"])
def test_rnn_to_native(backend, gate_name, natives_1q, natives_2q):
    gate = getattr(gates, gate_name)(0, 1, theta=0.1)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
def test_TOFFOLI_to_native(backend, natives_1q, natives_2q):
    gate = gates.TOFFOLI(0, 1, 2)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize("theta", [0.0, 0.1234, -1.3, math.pi / 2, 3.0])
@pytest.mark.parametrize(
    "natives_2q",
    [
        NativeGates.CZ,
        NativeGates.iSWAP,
        NativeGates.CZ | NativeGates.iSWAP,
        NativeGates.CNOT,
    ],
)
@pytest.mark.parametrize("natives_1q", [NativeGates.U3, NativeGates.GPI2])
def test_deutsch_to_native(backend, natives_1q, natives_2q, theta):
    gate = gates.DEUTSCH(0, 1, 2, theta)
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize("theta", [0.1234, math.pi / 2])
def test_deutsch_two_qubit_gate_count(backend, theta):
    gate = gates.DEUTSCH(0, 1, 2, theta)
    # six is the minimum, since the Deutsch gate is a Toffoli gate for theta = pi / 2
    assert cz_dec.count_2q(gate, backend) == 6
    assert standard_decompositions.count_2q(gate, backend) == 6


@pytest.mark.parametrize("qubits", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
@pytest.mark.parametrize("table", [cz_dec, standard_decompositions])
def test_deutsch_decomposition_on_other_qubits(backend, table, qubits):
    theta = 0.4321
    original = Circuit(3)
    original.add(gates.DEUTSCH(*qubits, theta))
    decomposed = Circuit(3)
    decomposed.add(table(gates.DEUTSCH(*qubits, theta), backend))
    unitary = backend.to_numpy(decomposed.unitary(backend))
    target = backend.to_numpy(original.unitary(backend))
    phase = np.vdot(target.ravel(), unitary.ravel())
    backend.assert_allclose(unitary * abs(phase) / phase, target, atol=1e-8)


@pytest.mark.parametrize("theta", [0.0, 0.1234, -1.3, math.pi, 3.0])
@pytest.mark.parametrize("natives_1q", [NativeGates.U3, NativeGates.GPI2])
def test_RBS_to_native_with_cnot(backend, natives_1q, theta):
    gate = gates.RBS(0, 1, theta)
    natives = natives_1q | NativeGates.CNOT | default_natives
    assert_matrices_allclose(gate, natives, backend)


def test_RBS_standard_decomposition_matches_gate_decompose():
    gate = gates.RBS(0, 1, 0.3)
    expected = gate.decompose()
    decomposed = standard_decompositions(gate)
    assert [(g.name, g.qubits, g.parameters) for g in decomposed] == [
        (g.name, g.qubits, g.parameters) for g in expected
    ]


@pytest.mark.parametrize(
    "natives_2q",
    [
        NativeGates.CZ,
        NativeGates.iSWAP,
        NativeGates.CZ | NativeGates.iSWAP,
        NativeGates.CNOT,
    ],
)
@pytest.mark.parametrize("natives_1q", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize(
    "gate",
    [
        gates.GeneralizedRBS([0], [1, 2], 0.37, 0.2),
        gates.GeneralizedRBS([0, 1], [2, 3], -1.3, 0.2),
        gates.Multiplexor([0, 1], 2, [0.1, 0.2, 0.3, 0.4], axis="y"),
        gates.Multiplexor([0, 1], 2, [0.1, 0.2, 0.3, 0.4], axis="z"),
        gates.FanOut(0, 1, 2),
        gates.FanOut(0, 1, 2, 3),
    ],
    ids=lambda gate: f"{gate.name}-{len(gate.qubits)}q",
)
def test_gates_with_own_decomposition_to_native(backend, gate, natives_1q, natives_2q):
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


@pytest.mark.parametrize("qubits", [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_gate_with_own_decomposition_on_other_qubits(backend, qubits):
    original = Circuit(3)
    original.add(gates.Multiplexor(qubits[:2], qubits[2], [0.1, 0.2, 0.3, 0.4]))
    translated = Circuit(3)
    translated.add(
        translate_gate(
            gates.Multiplexor(qubits[:2], qubits[2], [0.1, 0.2, 0.3, 0.4]),
            default_natives | NativeGates.CZ | NativeGates.GPI2,
            backend=backend,
        )
    )
    unitary = backend.to_numpy(translated.unitary(backend))
    target = backend.to_numpy(original.unitary(backend))
    phase = np.vdot(target, unitary)
    backend.assert_allclose(unitary * np.abs(phase) / phase, target, atol=1e-8)


@pytest.mark.parametrize("natives_1q", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize("seed", [None, 10])
def test_two_qubit_unitary_to_native_with_cnot(backend, natives_1q, seed):
    u = random_unitary(4, seed=seed, backend=NumpyBackend())
    gate = gates.Unitary(u, 0, 1)
    assert_matrices_allclose(
        gate, natives_1q | NativeGates.CNOT | default_natives, backend
    )


@pytest.mark.parametrize("natives_1q", [NativeGates.U3, NativeGates.GPI2])
@pytest.mark.parametrize(
    "gate",
    [
        gates.fSim(0, 1, 0.3, 0.5),
        gates.fSim(0, 1, -1.2, 2.0),
        gates.GeneralizedfSim(
            0, 1, random_unitary(2, seed=3, backend=NumpyBackend()), 0.4
        ),
    ],
    ids=lambda gate: gate.name,
)
def test_fsim_to_native_with_cnot(backend, gate, natives_1q):
    assert_matrices_allclose(
        gate, natives_1q | NativeGates.CNOT | default_natives, backend
    )


def test_gates_with_cnot_natives_use_only_cnot(backend):
    natives = NativeGates.RZ | NativeGates.SX | NativeGates.X | NativeGates.CNOT
    for gate in (
        gates.Unitary(random_unitary(4, seed=1, backend=NumpyBackend()), 0, 1),
        gates.fSim(0, 1, 0.3, 0.5),
        gates.FanOut(0, 1, 2),
    ):
        translated = translate_gate(gate, natives | default_natives, backend=backend)
        two_qubit_gates = [g for g in translated if len(g.qubits) == 2]
        assert two_qubit_gates
        assert all(isinstance(g, gates.CNOT) for g in two_qubit_gates)


@pytest.mark.parametrize("seed", [None, 10])
@pytest.mark.parametrize(
    "natives_2q",
    [NativeGates.CZ, NativeGates.iSWAP, NativeGates.CZ | NativeGates.iSWAP],
)
@pytest.mark.parametrize(
    "natives_1q",
    [NativeGates.U3, NativeGates.GPI2, NativeGates.U3 | NativeGates.GPI2],
)
@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_to_native(backend, nqubits, natives_1q, natives_2q, seed):
    u = random_unitary(2**nqubits, seed=seed, backend=NumpyBackend())
    # transform to SU(2^nqubits) form
    u = u / np.sqrt(np.linalg.det(u))
    gate = gates.Unitary(u, *range(nqubits))
    assert_matrices_allclose(gate, natives_1q | natives_2q | default_natives, backend)


def test_count_1q(backend):
    np.testing.assert_allclose(cz_dec.count_1q(gates.CNOT(0, 1), backend), 2)
    np.testing.assert_allclose(cz_dec.count_1q(gates.CRX(0, 1, 0.1), backend), 2)


def test_count_2q(backend):
    np.testing.assert_allclose(cz_dec.count_2q(gates.CNOT(0, 1), backend), 1)
    np.testing.assert_allclose(cz_dec.count_2q(gates.CRX(0, 1, 0.1), backend), 2)


@pytest.mark.parametrize(
    "natives_1q",
    [
        NativeGates.U3,
        NativeGates.GPI2,
        NativeGates.RX | NativeGates.RZ,
        NativeGates.SX | NativeGates.X,
    ],
)
@pytest.mark.parametrize("gate_name, ncz", [("CY", 1), ("SWAP", 3)])
def test_cz_count_to_native(backend, gate_name, ncz, natives_1q):
    natives = natives_1q | NativeGates.CZ | default_natives
    gate = getattr(gates, gate_name)(0, 1)
    assert_matrices_allclose(gate, natives=natives, backend=backend)
    translated = translate_gate(gate, natives, backend=backend)
    assert (
        sum(isinstance(translated_gate, gates.CZ) for translated_gate in translated)
        == ncz
    )


def test_static_decomposition_copies_gates():
    gate = gates.CNOT(0, 1)

    first = standard_decompositions(gate)
    second = standard_decompositions(gate)

    assert first[0] is not second[0]
    assert first[0] is not standard_decompositions.decompositions[gates.CNOT][0]


@pytest.mark.parametrize("controls,target", [([0, 1], 2), ([3, 1], 0)])
def test_function_decomposition_qubits(backend, controls, target):
    angles = backend.random_uniform(-np.pi, np.pi, 2 ** len(controls), seed=1)
    gate = gates.Multiplexor(controls, target, angles)

    decomposed = standard_decompositions(gate)

    assert len(decomposed) == 2 ** (len(controls) + 1)
    for rotation, cnot in zip(decomposed[::2], decomposed[1::2]):
        assert rotation.target_qubits == (target,)
        assert cnot.target_qubits == (target,)
        assert cnot.control_qubits[0] in controls

    nqubits = max(controls + [target]) + 1
    reference = Circuit(nqubits)
    reference.add(gate)
    circuit = Circuit(nqubits)
    circuit.add(decomposed)

    backend.assert_allclose(
        circuit.unitary(backend), reference.unitary(backend), atol=1e-8
    )


@pytest.mark.parametrize("shape", [(1,), (3, 1)])
def test_mottonen_compute_theta_single_angle(backend, shape):
    """A single angle (*i.e.*, no control qubits) is returned unchanged."""
    angles = backend.random_uniform(-math.pi, math.pi, shape, seed=1)

    thetas = _mottonen_compute_theta(angles, backend=backend)

    assert thetas.shape == angles.shape
    backend.assert_allclose(thetas, angles)


@pytest.mark.parametrize(
    "gate_class", list(direction_dec), ids=lambda gate: gate.__name__
)
def test_direction_dec(backend, gate_class):
    decomposition = Circuit(2)
    decomposition.add(direction_dec[gate_class])
    original = Circuit(2)
    original.add(gate_class(0, 1))
    unitary, target = decomposition.unitary(backend), original.unitary(backend)
    phase = backend.dot(backend.ravel(backend.conj(target)), backend.ravel(unitary))
    backend.assert_allclose(
        unitary * backend.conj(phase / abs(phase)), target, atol=1e-12
    )
    # the gate is applied once, in the opposite direction
    reversed_gates = [
        gate for gate in direction_dec[gate_class] if isinstance(gate, gate_class)
    ]
    assert [gate.qubits for gate in reversed_gates] == [(1, 0)]


def test_direction_dec_non_symmetric_natives(backend):
    swap = [0, 2, 1, 3]
    non_symmetric = []
    for name in _TWO_QUBIT_GATES:
        gate_class = getattr(gates, name)
        matrix = backend.to_numpy(gate_class(0, 1).matrix(backend))
        if not np.allclose(matrix, matrix[swap][:, swap]):
            non_symmetric.append(gate_class)
    assert set(non_symmetric) == set(direction_dec)
