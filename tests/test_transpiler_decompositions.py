import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler.asserts import assert_decomposition
from qibo.transpiler.decompositions import (
    _mottonen_compute_theta,
    standard_decompositions,
)
from qibo.transpiler.unroller import NativeGates, cz_dec, translate_gate

default_natives = NativeGates.Z | NativeGates.RZ | NativeGates.M | NativeGates.I


def assert_matrices_allclose(gate, natives, backend):
    target_matrix = gate.matrix(backend)

    circuit = Circuit(len(gate.qubits))
    circuit.add(translate_gate(gate, natives, backend=backend))
    native_matrix = circuit.unitary(backend)

    # The matrices are equal up to a global phase, which is the one of their overlap.
    # It is not computed from the determinants, because their roots are ambiguous for a
    # determinant on the branch cut, as the one of some gates which is exactly -1.
    overlap = backend.dot(
        backend.ravel(backend.conj(target_matrix)), backend.ravel(native_matrix)
    )
    backend.assert_allclose(
        native_matrix, overlap / backend.abs(overlap) * target_matrix, atol=1e-6
    )
    assert_decomposition(circuit, natives)


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
