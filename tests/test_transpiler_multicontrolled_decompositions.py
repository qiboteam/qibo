import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler.multicontrolled_decompositions import (
    multi_controlled_decomposition,
)
from qibo.transpiler.unroller import NativeGates, Unroller


def _ry(angle):
    return np.array(
        [
            [math.cos(angle / 2), -math.sin(angle / 2)],
            [math.sin(angle / 2), math.cos(angle / 2)],
        ],
        dtype=complex,
    )


def _su2(alpha, beta):
    return np.array([[alpha, beta], [-np.conj(beta), np.conj(alpha)]], dtype=complex)


# Matrices with determinant one.
SU2_MATRICES = {
    "identity": np.eye(2, dtype=complex),
    "minus identity": -np.eye(2, dtype=complex),
    "iX": 1j * np.array([[0, 1], [1, 0]]),
    "iZ": 1j * np.diag([1.0, -1.0]),
    "real rotation": _ry(1.1),
    "rotation close to 2pi": _ry(2 * math.pi - 1e-6),
    "rotation very close to 2pi": _ry(2 * math.pi - 1e-13),
    "diagonal": np.diag([np.exp(0.7j), np.exp(-0.7j)]),
    "real main diagonal": _su2(0.6, 0.8j),
    "real secondary diagonal": _su2(0.3 + 0.4j, math.sqrt(0.75)),
    "close to minus identity": _su2(-math.sqrt(1 - 1e-10), 1e-5j),
}
# Matrices with determinant different from one.
U2_MATRICES = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
    "Hadamard": np.array([[1, 1], [1, -1]]) / math.sqrt(2),
    "phase": np.diag([1.0, np.exp(0.9j)]),
    "global phase": np.exp(0.4j) * np.eye(2),
    "close to global phase": np.exp(0.4j) * _ry(1e-6),
    "very close to global phase": np.exp(0.4j) * _ry(1e-12),
}
MATRICES = {**SU2_MATRICES, **U2_MATRICES}


def _unitary(backend, gate_list, nqubits):
    circuit = Circuit(nqubits)
    circuit.add(gate_list)
    return circuit.unitary(backend)


def _assert_decomposition(backend, decomposition, controls, target, matrix, nqubits):
    """Compares with the exact multi-controlled gate, including the global phase."""
    gate = gates.Unitary(matrix, target).controlled_by(*controls)
    backend.assert_allclose(
        _unitary(backend, decomposition, nqubits),
        _unitary(backend, [gate], nqubits),
        atol=1e-10,
    )


def _cnot_count(gate_list, nqubits):
    circuit = Circuit(nqubits)
    circuit.add(gate_list)
    native = Unroller(NativeGates.CNOT | NativeGates.U3)(circuit)
    return native.gate_types.get(gates.CNOT, 0)


@pytest.mark.parametrize("nctrl", range(1, 7))
@pytest.mark.parametrize("name", list(MATRICES))
def test_multi_controlled_decomposition(backend, nctrl, name):
    controls, target = tuple(range(nctrl)), nctrl
    decomposition = multi_controlled_decomposition(
        backend.cast(MATRICES[name]), controls, target, backend=backend
    )
    _assert_decomposition(
        backend, decomposition, controls, target, MATRICES[name], nctrl + 1
    )


@pytest.mark.parametrize("special", [True, False])
@pytest.mark.parametrize("nctrl", range(2, 8))
def test_multi_controlled_decomposition_random(backend, nctrl, special):
    matrix = backend.to_numpy(random_unitary(2, seed=nctrl, backend=backend))
    if special:
        matrix = matrix / np.sqrt(np.linalg.det(matrix))

    controls, target = tuple(range(nctrl)), nctrl
    decomposition = multi_controlled_decomposition(
        backend.cast(matrix), controls, target, backend=backend
    )
    _assert_decomposition(backend, decomposition, controls, target, matrix, nctrl + 1)


def test_multi_controlled_decomposition_scattered_qubits(backend):
    controls, target, nqubits = (5, 1, 3, 0), 2, 7
    matrix = backend.to_numpy(random_unitary(2, seed=9, backend=backend))
    decomposition = multi_controlled_decomposition(
        backend.cast(matrix), controls, target, backend=backend
    )
    _assert_decomposition(backend, decomposition, controls, target, matrix, nqubits)


def test_multi_controlled_decomposition_no_controls(backend):
    with pytest.raises(ValueError):
        multi_controlled_decomposition(backend.matrices.X, (), 0, backend=backend)


@pytest.mark.parametrize("nctrl", range(3, 8))
def test_x_decompose_without_free_qubits(backend, nctrl):
    controls, target = tuple(range(nctrl)), nctrl
    gate = gates.X(target).controlled_by(*controls)
    decomposition = gate.decompose()

    backend.assert_allclose(
        _unitary(backend, decomposition, nctrl + 1),
        _unitary(backend, [gate], nctrl + 1),
        atol=1e-10,
    )
    # Only qubits of the gate are used
    used = {q for g in decomposition for q in g.qubits}
    assert used == set(range(nctrl + 1))


@pytest.mark.parametrize("nctrl", range(2, 5))
@pytest.mark.parametrize(
    "gate_class",
    [
        lambda q: gates.H(q),
        lambda q: gates.Y(q),
        lambda q: gates.Z(q),
        lambda q: gates.S(q),
        lambda q: gates.T(q),
        lambda q: gates.SX(q),
        lambda q: gates.RX(q, 0.3),
        lambda q: gates.RY(q, 0.4),
        lambda q: gates.RZ(q, 0.5),
        lambda q: gates.U1(q, 0.6),
        lambda q: gates.U2(q, 0.7, 0.8),
        lambda q: gates.U3(q, 0.9, 1.0, 1.1),
        lambda q: gates.Unitary(np.exp(0.3j) * _ry(0.5), q),
    ],
)
def test_decompose_multi_controlled_one_qubit_gates(backend, gate_class, nctrl):
    controls, target = tuple(range(nctrl)), nctrl
    gate = gate_class(target).controlled_by(*controls)
    decomposition = gate.decompose()

    backend.assert_allclose(
        _unitary(backend, decomposition, nctrl + 1),
        _unitary(backend, [gate], nctrl + 1),
        atol=1e-10,
    )


@pytest.mark.parametrize("nctrl", [2, 3, 5])
def test_decompose_agrees_with_multi_controlled_decomposition(backend, nctrl):
    matrix = backend.to_numpy(random_unitary(2, seed=1, backend=backend))
    controls, target = tuple(range(nctrl)), nctrl
    expected = multi_controlled_decomposition(
        backend.cast(matrix), controls, target, backend=backend
    )
    decomposition = gates.Unitary(matrix, target).controlled_by(*controls).decompose()

    assert [type(g) for g in decomposition] == [type(g) for g in expected]
    assert [g.qubits for g in decomposition] == [g.qubits for g in expected]


@pytest.mark.parametrize("name", ["real rotation", "X"])
def test_use_toffolis(backend, name):
    controls, target = tuple(range(4)), 4
    matrix = backend.cast(MATRICES[name])
    with_toffolis = multi_controlled_decomposition(
        matrix, controls, target, use_toffolis=True, backend=backend
    )
    without_toffolis = multi_controlled_decomposition(
        matrix, controls, target, use_toffolis=False, backend=backend
    )

    assert any(isinstance(g, gates.TOFFOLI) for g in with_toffolis) == (
        name == "real rotation"
    )
    assert not any(isinstance(g, gates.TOFFOLI) for g in without_toffolis)
    backend.assert_allclose(
        _unitary(backend, without_toffolis, 5),
        _unitary(backend, with_toffolis, 5),
        atol=1e-10,
    )


def _raise_global_backend_used(*args, **kwargs):
    raise RuntimeError("The global backend must not be used.")


DECOMPOSED_GATES = [
    lambda: gates.X(5).controlled_by(0, 1, 2, 3, 4),
    lambda: gates.RX(4, 0.3).controlled_by(0, 1, 2, 3),
    lambda: gates.H(3).controlled_by(0, 1, 2),
    lambda: gates.Unitary(np.exp(0.3j) * _ry(0.5), 3).controlled_by(0, 1, 2),
]


@pytest.mark.parametrize("use_toffolis", [True, False])
@pytest.mark.parametrize("gate_class", DECOMPOSED_GATES)
def test_decompose_does_not_use_global_backend(monkeypatch, gate_class, use_toffolis):
    gate = gate_class()
    monkeypatch.setattr("qibo.backends.get_backend", _raise_global_backend_used)
    decomposition = gate.decompose(use_toffolis=use_toffolis)

    assert len(decomposition) > 0
    # Parameters are plain numbers, not arrays of any backend.
    for decomposed_gate in decomposition:
        assert all(
            isinstance(parameter, (float, np.floating))
            for parameter in decomposed_gate.parameters
        )


@pytest.mark.parametrize("gate_class", DECOMPOSED_GATES)
def test_decompose_is_independent_of_global_backend(monkeypatch, backend, gate_class):
    gate = gate_class()
    expected = gate.decompose()

    monkeypatch.setattr("qibo.backends.get_backend", lambda *args, **kwargs: backend)
    decomposition = gate.decompose()

    assert [type(g) for g in decomposition] == [type(g) for g in expected]
    assert [g.qubits for g in decomposition] == [g.qubits for g in expected]
    for decomposed_gate, expected_gate in zip(decomposition, expected):
        np.testing.assert_allclose(
            decomposed_gate.parameters, expected_gate.parameters, atol=1e-12
        )


def test_decompose_does_not_modify_the_gate():
    gate = gates.X(4).controlled_by(0, 1, 2, 3)
    gate.decompose()
    assert gate.control_qubits == (0, 1, 2, 3)
    assert gate.target_qubits == (4,)


# CNOT cost of the decompositions after unrolling to CNOT and one-qubit gates,
# for 2 to 8 controls. It grows linearly for special unitary gates, and
# quadratically for the other gates.
CNOT_COUNTS = {
    "special": [4, 14, 24, 40, 56, 80, 104],
    "general": [10, 26, 50, 82, 122, 170, 226],
}


@pytest.mark.parametrize("kind", ["special", "general"])
def test_cnot_counts(backend, kind):
    matrix = backend.to_numpy(random_unitary(2, seed=1, backend=backend))
    if kind == "special":
        matrix = matrix / np.sqrt(np.linalg.det(matrix))

    counts = []
    for nctrl in range(2, 9):
        gate = gates.Unitary(matrix, nctrl).controlled_by(*range(nctrl))
        counts.append(_cnot_count(gate.decompose(), nctrl + 1))

    assert counts == CNOT_COUNTS[kind]
