from functools import reduce
from itertools import repeat

import numpy as np
import pytest
from sympy import S

import qibo
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import DepolarizingError, NoiseModel
from qibo.quantum_info import to_pauli_liouville
from qibo.tomography.gate_set_tomography import (
    GST,
    _expectation_value,
    _gate_tomography,
    _get_observable,
    _measurement_basis,
    _prepare_state,
)


def _compare_gates(g1, g2):
    assert g1.__class__.__name__ == g2.__class__.__name__
    assert g1.qubits == g2.qubits


INDEX_NQUBITS = (
    list(zip(range(4), repeat(1, 4)))
    + list(zip(range(16), repeat(2, 16)))
    + [(0, 3), (17, 1)]
)


@pytest.mark.parametrize(
    "k,nqubits",
    INDEX_NQUBITS,
)
def test__prepare_state(k, nqubits):
    correct_gates = {
        1: [[gates.I(0)], [gates.X(0)], [gates.H(0)], [gates.H(0), gates.S(0)]],
        2: [
            [gates.I(0), gates.I(1)],
            [gates.I(0), gates.X(1)],
            [gates.I(0), gates.H(1)],
            [gates.I(0), gates.H(1), gates.S(1)],
            [gates.X(0), gates.I(1)],
            [gates.X(0), gates.X(1)],
            [gates.X(0), gates.H(1)],
            [gates.X(0), gates.H(1), gates.S(1)],
            [gates.H(0), gates.I(1)],
            [gates.H(0), gates.X(1)],
            [gates.H(0), gates.H(1)],
            [gates.H(0), gates.H(1), gates.S(1)],
            [gates.H(0), gates.S(0), gates.I(1)],
            [gates.H(0), gates.S(0), gates.X(1)],
            [gates.H(0), gates.S(0), gates.H(1)],
            [gates.H(0), gates.S(0), gates.H(1), gates.S(1)],
        ],
    }
    errors = {(0, 3): ValueError, (17, 1): IndexError}
    if (k, nqubits) in [(0, 3), (17, 1)]:
        with pytest.raises(errors[(k, nqubits)]):
            prepared_states = _prepare_state(k, nqubits)
    else:
        prepared_states = _prepare_state(k, nqubits)
        for groundtruth, gate in zip(correct_gates[nqubits][k], prepared_states):
            _compare_gates(groundtruth, gate)


@pytest.mark.parametrize(
    "j,nqubits",
    INDEX_NQUBITS,
)
def test__measurement_basis(j, nqubits):
    correct_gates = {
        1: [
            [gates.M(0)],
            [gates.M(0, basis=gates.X)],
            [gates.M(0, basis=gates.Y)],
            [gates.M(0, basis=gates.Z)],
        ],
        2: [
            [gates.M(0), gates.M(1)],
            [gates.M(0), gates.M(1, basis=gates.X)],
            [gates.M(0), gates.M(1, basis=gates.Y)],
            [gates.M(0), gates.M(1)],
            [gates.M(0, basis=gates.X), gates.M(1)],
            [gates.M(0, basis=gates.X), gates.M(1, basis=gates.X)],
            [gates.M(0, basis=gates.X), gates.M(1, basis=gates.Y)],
            [gates.M(0, basis=gates.X), gates.M(1)],
            [gates.M(0, basis=gates.Y), gates.M(1)],
            [gates.M(0, basis=gates.Y), gates.M(1, basis=gates.X)],
            [gates.M(0, basis=gates.Y), gates.M(1, basis=gates.Y)],
            [gates.M(0, basis=gates.Y), gates.M(1)],
            [gates.M(0), gates.M(1)],
            [gates.M(0), gates.M(1, basis=gates.X)],
            [gates.M(0), gates.M(1, basis=gates.Y)],
            [gates.M(0), gates.M(1)],
        ],
    }
    errors = {(0, 3): ValueError, (17, 1): IndexError}
    if (j, nqubits) in [(0, 3), (17, 1)]:
        with pytest.raises(errors[(j, nqubits)]):
            prepared_gates = _measurement_basis(j, nqubits)
    else:
        prepared_gates = _measurement_basis(j, nqubits)
        for groundtruth, gate in zip(correct_gates[nqubits][j], prepared_gates):
            _compare_gates(groundtruth, gate)
            for g1, g2 in zip(groundtruth.basis, gate.basis):
                _compare_gates(g1, g2)


@pytest.mark.parametrize(
    "j,nqubits",
    INDEX_NQUBITS,
)
def test__get_observable(j, nqubits):
    correct_observables = {
        1: [
            (S(1),),
            (qibo.symbols.Z(0),),
            (qibo.symbols.Z(0),),
            (qibo.symbols.Z(0),),
        ],
        2: [
            (S(1), S(1)),
            (S(1), qibo.symbols.Z(1)),
            (S(1), qibo.symbols.Z(1)),
            (S(1), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), S(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), S(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), S(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
        ],
    }
    correct_observables[1] = [
        SymbolicHamiltonian(h[0]).form for h in correct_observables[1]
    ]
    correct_observables[2] = [
        SymbolicHamiltonian(reduce(lambda x, y: x * y, h)).form
        for h in correct_observables[2]
    ]
    errors = {(0, 3): ValueError, (17, 1): IndexError}
    if (j, nqubits) in [(0, 3), (17, 1)]:
        with pytest.raises(errors[(j, nqubits)]):
            prepared_observable = _get_observable(j, nqubits)
    else:
        prepared_observable = _get_observable(j, nqubits).form
        groundtruth = correct_observables[nqubits][j]
        assert groundtruth == prepared_observable


def test_expectation_value_nqubits_error(backend):
    nqubits = 3
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.TOFFOLI(0, 1, 2))
    test_circuit.add(gates.M(*np.arange(0, nqubits, 1)))
    with pytest.raises(ValueError):
        expectation_val = _expectation_value(
            test_circuit,
            1,
            nshots=int(1e4),
            backend=backend,
        )


@pytest.mark.parametrize(
    "nqubits, gate",
    [
        (1, gates.CNOT(0, 1)),
        (3, gates.TOFFOLI(0, 1, 2)),
    ],
)
def test_gate_tomography_value_error(backend, nqubits, gate):
    with pytest.raises(ValueError):
        matrix_jk = _gate_tomography(
            nqubits=nqubits,
            gate=gate,
            nshots=int(1e4),
            noise_model=None,
            backend=backend,
        )


def test_gate_tomography_noise_model(backend):
    nqubits = 1
    gate = gates.H(0)
    lam = 1.0
    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(lam))
    # return noise_model
    target = _gate_tomography(
        nqubits=nqubits,
        gate=gate,
        nshots=int(1e4),
        noise_model=noise_model,
        backend=backend,
    )
    exact_matrix = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    backend.assert_allclose(
        target,
        exact_matrix,
        atol=1e-1,
    )


@pytest.mark.parametrize(
    "target_gates",
    [[gates.SX(0), gates.Z(0), gates.CY(0, 1)], [gates.TOFFOLI(0, 1, 2)]],
)
@pytest.mark.parametrize("Pauli_Liouville", [False, True])
def test_GST(backend, target_gates, Pauli_Liouville):
    T = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    target_matrices = [g.matrix() for g in target_gates]
    # superoperator representation of the target gates in the Pauli basis
    target_matrices = [to_pauli_liouville(m, normalize=True) for m in target_matrices]
    gate_set = [g.__class__ for g in target_gates]

    if len(target_gates) == 3:
        empty_1q, empty_2q, *approx_gates = GST(
            gate_set=gate_set,
            nshots=int(1e4),
            include_empty=True,
            Pauli_Liouville=Pauli_Liouville,
            backend=backend,
        )

        T_2q = np.kron(T, T)
        for target, estimate in zip(target_matrices, approx_gates):
            if not Pauli_Liouville:
                G = empty_1q if estimate.shape[0] == 4 else empty_2q
                T_matrix = T if estimate.shape[0] == 4 else T_2q
                estimate = T_matrix @ np.linalg.inv(G) @ estimate @ np.linalg.inv(G)
            backend.assert_allclose(
                target,
                estimate,
                atol=1e-1,
            )
    else:
        with pytest.raises(RuntimeError):
            empty_1q, empty_2q, *approx_gates = GST(
                gate_set=[g.__class__ for g in target_gates],
                nshots=int(1e4),
                include_empty=True,
                Pauli_Liouville=Pauli_Liouville,
                backend=backend,
            )


def test_GST_invertible_matrix():
    T = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    matrices = GST(gate_set=[], Pauli_Liouville=True, T=T)
    assert True


def test_GST_non_invertible_matrix():
    T = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, -1, 0, 0]])
    with pytest.raises(ValueError):
        matrices = GST(gate_set=[], Pauli_Liouville=True, T=T)
