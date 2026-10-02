import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.backends import NumpyBackend
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


# (number of controls, number of free qubits) with at most 11 qubits in total.
FREE_QUBITS_CASES = [
    (3, 1),
    (3, 3),
    (4, 1),
    (4, 2),
    (4, 4),
    (5, 1),
    (5, 3),
    (6, 1),
    (6, 2),
    (6, 4),
    (7, 1),
    (7, 2),
    (8, 1),
    (8, 2),
]


@pytest.mark.parametrize(("nctrl", "nfree"), FREE_QUBITS_CASES)
def test_x_decompose_with_free_qubits(backend, nctrl, nfree):
    """The free qubits can be in any state, so the whole unitary must be exact."""
    nqubits = nctrl + 1 + nfree
    controls, target, free = (
        tuple(range(nctrl)),
        nctrl,
        tuple(range(nctrl + 1, nqubits)),
    )
    gate = gates.X(target).controlled_by(*controls)
    decomposition = gate.decompose(*free)

    backend.assert_allclose(
        _unitary(backend, decomposition, nqubits),
        _unitary(backend, [gate], nqubits),
        atol=1e-10,
    )
    used = {q for g in decomposition for q in g.qubits}
    assert set(controls + (target,)) <= used <= set(range(nqubits))


def test_x_decompose_with_free_qubits_scattered(backend):
    controls, target, free, nqubits = (6, 2, 9, 0, 4), 7, (1, 8, 3), 10
    gate = gates.X(target).controlled_by(*controls)
    backend.assert_allclose(
        _unitary(backend, gate.decompose(*free), nqubits),
        _unitary(backend, [gate], nqubits),
        atol=1e-10,
    )


def test_circuit_decompose_with_free_qubits(backend):
    circuit = Circuit(8)
    circuit.add(gates.X(4).controlled_by(0, 1, 2, 3))
    circuit.add(gates.X(7).controlled_by(1, 2, 3, 4, 5))
    decomposed = circuit.decompose(6)

    assert not any(g.is_controlled_by for g in decomposed.queue)
    backend.assert_allclose(
        decomposed.unitary(backend), circuit.unitary(backend), atol=1e-10
    )


def test_free_qubits_are_ignored_by_other_gates():
    """Only multi-controlled ``X`` gates benefit from auxiliary qubits."""
    gate = gates.RX(4, 0.3).controlled_by(0, 1, 2, 3)
    with_free, without_free = gate.decompose(5, 6), gate.decompose()
    assert [type(g) for g in with_free] == [type(g) for g in without_free]
    assert [g.qubits for g in with_free] == [g.qubits for g in without_free]


@pytest.mark.parametrize("name", ["real rotation", "X"])
def test_multi_controlled_decomposition_free_qubits_errors(backend, name):
    matrix = backend.cast(MATRICES[name])
    for free in [(0,), (4,), (5, 2)]:
        with pytest.raises(ValueError):
            multi_controlled_decomposition(
                matrix, (0, 1, 2, 3), 4, free=free, backend=backend
            )


class _AcceleratorArray:
    """Array that, like a CuPy array, cannot be implicitly converted to NumPy."""

    def __init__(self, array):
        self._array = np.asarray(array)
        self.shape = self._array.shape

    def get(self):
        return self._array

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed.")


def _computation_on_accelerator(*args, **kwargs):
    raise AssertionError("Computation on the accelerator backend.")


class _AcceleratorBackend(NumpyBackend):
    """Backend of ``_AcceleratorArray`` where only moving data to the CPU is allowed."""

    eig = det = abs = angle = exp = matmul = matrix_norm = staticmethod(
        _computation_on_accelerator
    )

    def to_numpy(self, array):
        return array.get() if isinstance(array, _AcceleratorArray) else array


def _parameters(gate_list):
    return np.array([float(p) for g in gate_list for p in g.parameters])


@pytest.mark.parametrize("free", [(), (5, 6, 7)])
@pytest.mark.parametrize("name", ["real rotation", "diagonal", "Hadamard", "X"])
def test_multi_controlled_decomposition_of_accelerator_array(name, free):
    """The decomposition only moves the matrix to the CPU, and returns plain floats."""
    controls = (0, 1, 2, 3)
    expected = multi_controlled_decomposition(
        MATRICES[name], controls, 4, free, backend=NumpyBackend()
    )
    decomposition = multi_controlled_decomposition(
        _AcceleratorArray(MATRICES[name]),
        controls,
        4,
        free,
        backend=_AcceleratorBackend(),
    )

    assert [type(g) for g in decomposition] == [type(g) for g in expected]
    assert [g.qubits for g in decomposition] == [g.qubits for g in expected]
    np.testing.assert_allclose(
        _parameters(decomposition), _parameters(expected), atol=1e-12
    )
    for decomposed_gate in decomposition:
        assert all(
            isinstance(parameter, (float, np.floating))
            for parameter in decomposed_gate.parameters
        )


@pytest.mark.parametrize("name", ["real rotation", "Hadamard"])
def test_decompose_unitary_gate_with_accelerator_array(name):
    """A ``Unitary`` that holds an array of an accelerator can be decomposed."""
    gate = gates.Unitary(_AcceleratorArray(MATRICES[name]), 3, check_unitary=False)
    decomposition = gate.controlled_by(0, 1, 2).decompose()
    expected = gates.Unitary(MATRICES[name], 3).controlled_by(0, 1, 2).decompose()

    assert [type(g) for g in decomposition] == [type(g) for g in expected]
    assert [g.qubits for g in decomposition] == [g.qubits for g in expected]
    np.testing.assert_allclose(
        _parameters(decomposition), _parameters(expected), atol=1e-12
    )


def _raise_global_backend_used(*args, **kwargs):
    raise RuntimeError("The global backend must not be used.")


DECOMPOSED_GATES = [
    lambda: gates.X(5).controlled_by(0, 1, 2, 3, 4),
    lambda: gates.RX(4, 0.3).controlled_by(0, 1, 2, 3),
    lambda: gates.H(3).controlled_by(0, 1, 2),
    lambda: gates.Unitary(np.exp(0.3j) * _ry(0.5), 3).controlled_by(0, 1, 2),
]


@pytest.mark.parametrize("free", [(), (6,), (6, 7, 8)])
@pytest.mark.parametrize("gate_class", DECOMPOSED_GATES)
def test_decompose_does_not_use_global_backend(monkeypatch, gate_class, free):
    gate = gate_class()
    monkeypatch.setattr("qibo.backends.get_backend", _raise_global_backend_used)
    decomposition = gate.decompose(*free)

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


# CNOT cost of a multi-controlled ``X`` gate with 3 to 8 controls, after unrolling
# to CNOT and one-qubit gates: with the minimum number of free qubits for a V-chain
# (the number of controls minus two), and with only one free qubit.
FREE_QUBITS_CNOT_COUNTS = {
    "vchain": [14, 26, 34, 42, 50, 58],
    "one free qubit": [14, 36, 56, 72, 88, 104],
}


@pytest.mark.parametrize("kind", ["vchain", "one free qubit"])
def test_cnot_counts_with_free_qubits(kind):
    counts = []
    for nctrl in range(3, 9):
        nfree = nctrl - 2 if kind == "vchain" else 1
        nqubits = nctrl + 1 + nfree
        gate = gates.X(nctrl).controlled_by(*range(nctrl))
        free = range(nctrl + 1, nqubits)
        counts.append(_cnot_count(gate.decompose(*free), nqubits))

    assert counts == FREE_QUBITS_CNOT_COUNTS[kind]


def test_free_qubits_make_the_cost_of_x_linear():
    """Without auxiliary qubits the number of CNOTs grows quadratically."""
    nctrls = range(6, 13)
    with_free, without_free = [], []
    for nctrl in nctrls:
        gate = gates.X(nctrl).controlled_by(*range(nctrl))
        with_free.append(_cnot_count(gate.decompose(nctrl + 1), nctrl + 2))
        without_free.append(_cnot_count(gate.decompose(), nctrl + 1))

    # Constant first differences: linear. Constant second differences: quadratic.
    assert len(set(np.diff(with_free))) == 1
    assert len(set(np.diff(without_free, n=2))) == 1
    assert all(w < wo for w, wo in zip(with_free, without_free))
