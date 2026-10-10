import logging
import math
from functools import reduce
from itertools import product, repeat
from unittest.mock import patch

import numpy as np
import pytest
from sympy import S

from qibo import Circuit, gates
from qibo.backends import NumpyBackend, construct_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import DepolarizingError, NoiseModel
from qibo.quantum_info.superoperator_transformations import to_pauli_liouville
from qibo.symbols import Z
from qibo.tomography import GateSetTomography, Tomography
from qibo.tomography.gate_set_tomography import (
    GST,
    _extract_gate,
    _extract_nqubits,
    _gate_tomography,
    _get_observable,
    _get_swap_pairs,
    _measurement_basis,
    _prepare_state,
)
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.placer import Random
from qibo.transpiler.router import Sabre
from qibo.transpiler.unroller import NativeGates, Unroller


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
    "j, nqubits",
    INDEX_NQUBITS,
)
def test__get_observable(j, nqubits):
    backend = NumpyBackend()
    correct_observables = {
        1: [
            (S(1),),
            (Z(0, backend=backend),),
            (Z(0, backend=backend),),
            (Z(0, backend=backend),),
        ],
        2: [
            (S(1), S(1)),
            (S(1), Z(1, backend=backend)),
            (S(1), Z(1, backend=backend)),
            (S(1), Z(1, backend=backend)),
            (Z(0, backend=backend), S(1)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), S(1)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), S(1)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), Z(1, backend=backend)),
            (Z(0, backend=backend), Z(1, backend=backend)),
        ],
    }
    correct_observables[1] = [
        SymbolicHamiltonian(h[0], backend=backend).form for h in correct_observables[1]
    ]
    correct_observables[2] = [
        SymbolicHamiltonian(reduce(lambda x, y: x * y, h), backend=backend).form
        for h in correct_observables[2]
    ]
    errors = {(0, 3): ValueError, (17, 1): IndexError}
    if (j, nqubits) in [(0, 3), (17, 1)]:
        with pytest.raises(errors[(j, nqubits)]):
            prepared_observable = _get_observable(j, nqubits, backend=str(backend))
    else:
        prepared_observable = _get_observable(j, nqubits, backend=str(backend)).form
        groundtruth = correct_observables[nqubits][j]
        assert groundtruth == prepared_observable


def test__extract_nqubits():
    correct_nqubits = [1, 1, 2, 3]
    gates_to_test = [gates.Z, (gates.RX, [np.pi / 2]), gates.CNOT, gates.TOFFOLI]
    for idx in range(len(gates_to_test)):
        gate = gates_to_test[idx]
        if isinstance(gate, tuple):
            gate, _ = gate
        if idx < 3:
            assert _extract_nqubits(gate) == correct_nqubits[idx]
        else:
            with pytest.raises(RuntimeError):
                _extract_nqubits(gate)


def test__extract_gate_default_qubits():

    gates_to_test = [
        gates.T,
        (gates.RX, [np.pi / 2]),
        (gates.Unitary, [np.eye(2)]),
        (gates.CRX, [np.pi / 3]),
        (gates.Unitary, [np.eye(4)]),
    ]

    expected_qubits_list = [
        (0,),
        (0,),
        (0,),
        (0, 1),
        (0, 1),
    ]

    for gate_spec, expected_qubits in zip(gates_to_test, expected_qubits_list):
        extracted_gate, _ = _extract_gate(gate_spec, qubits=None)
        assert extracted_gate.qubits == expected_qubits


def test__extract_gate_user_defined_qubits():

    gates_to_test = [
        gates.T,
        (gates.RX, [np.pi / 2]),
        (gates.Unitary, [np.eye(2)]),
        (gates.CRX, [np.pi / 3]),
        (gates.Unitary, [np.eye(4)]),
    ]

    qubits_list = [2, 2, 2, (2, 3), (2, 3)]
    expected_qubits_list = [
        (2,),
        (2,),
        (2,),
        (2, 3),
        (2, 3),
    ]

    for gate_spec, qubit_spec, expected_qubits in zip(
        gates_to_test, qubits_list, expected_qubits_list
    ):
        extracted_gate, _ = _extract_gate(gate_spec, qubits=qubit_spec)
        assert extracted_gate.qubits == expected_qubits


@pytest.mark.parametrize(
    "gate, error_type",
    [
        (((gates.Unitary), np.array([[1, 2], [3, 4]])), ValueError),
        ((gates.TOFFOLI), RuntimeError),
    ],
)
def test__extract_gate_error(gate, error_type):
    with pytest.raises(error_type):
        _extracted_gate, _ = _extract_gate(gate)


@pytest.mark.parametrize(
    "nqubits, gate",
    [
        (1, gates.CNOT(0, 1)),
        (3, gates.TOFFOLI(0, 1, 2)),
    ],
)
def test_gate_tomography_value_error(backend, nqubits, gate):
    with pytest.raises(ValueError):
        _gate_tomography(
            nqubits=nqubits,
            gate=[gate],
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
        gate=[gate],
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


def test_gate_tomography_single_gate(backend):
    gate = gates.X(0)
    nqubits = 1
    matrix_jk = _gate_tomography(
        nqubits=nqubits,
        gate=gate,
        nshots=int(1e4),
        noise_model=None,
        backend=backend,
    )

    ground_truth_matrix = np.array(
        [[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, -1], [-1, 1, 0, 0]]
    )
    assert matrix_jk.shape == (4**nqubits, 4**nqubits)
    backend.assert_allclose(matrix_jk, ground_truth_matrix, atol=1e-1)


GROUND_TRUTH_GATE_X = np.array(
    [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1]]
)
GROUND_TRUTH_GATE_XT_0 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [
            0,
            0,
            1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
            0,
            1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
            0,
            1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
            0,
            1 / np.sqrt(2),
            -1 / np.sqrt(2),
        ],
        [
            0,
            0,
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            0,
            0,
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            0,
            0,
            1 / np.sqrt(2),
            1 / np.sqrt(2),
            0,
            0,
            1 / np.sqrt(2),
            1 / np.sqrt(2),
        ],
        [1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0],
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [
            0,
            0,
            -1 / np.sqrt(2),
            1 / np.sqrt(2),
            0,
            0,
            -1 / np.sqrt(2),
            1 / np.sqrt(2),
            0,
            0,
            -1 / np.sqrt(2),
            1 / np.sqrt(2),
            0,
            0,
            -1 / np.sqrt(2),
            1 / np.sqrt(2),
        ],
        [
            0,
            0,
            -1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
            0,
            -1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
            0,
            -1 / np.sqrt(2),
            -1 / np.sqrt(2),
            0,
            0,
            -1 / np.sqrt(2),
            -1 / np.sqrt(2),
        ],
        [-1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0],
    ],
    dtype=np.complex128,
)
GROUND_TRUTH_GATE_XT_1 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
        [0] * 16,
        [0] * 16,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0] * 16,
        [0] * 16,
        [-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.complex128,
)
GROUND_TRUTH_NULL_2 = np.array(
    [
        [1] * 16,
        [0] * 16,
        [0] * 16,
        [1] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [1] * 16,
        [0] * 16,
        [0] * 16,
        [1] * 16,
    ],
    dtype=np.complex128,
)
GROUND_TRUTH_I_2 = np.array(
    [
        [1] * 16,
        [0] * 16,
        [0] * 16,
        [1] * 16,
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0] * 16,
        [0] * 16,
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0] * 16,
        [0] * 16,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0] * 16,
        [0] * 16,
        [1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.complex128,
)


@pytest.mark.parametrize(
    "nqubits, gate, ancilla, raise_error, ground_truth_matrix",
    [
        (1, [gates.X(0)], 0, False, GROUND_TRUTH_GATE_X),
        (2, [gates.X(0), gates.T(1)], 0, False, GROUND_TRUTH_GATE_XT_0),
        (2, [gates.X(0), gates.T(1)], 1, False, GROUND_TRUTH_GATE_XT_1),
        (2, [], 2, False, GROUND_TRUTH_NULL_2),
        (2, [gates.Unitary(np.eye(4), 0, 1)], 1, False, GROUND_TRUTH_I_2),
        (2, [gates.X(0), gates.T(1), gates.S(0)], 1, True, np.nan),
    ],
)
def test_gate_tomography_apply_ancillas(
    backend, nqubits, gate, ancilla, raise_error, ground_truth_matrix
):
    if raise_error:
        with pytest.raises(ValueError):
            matrix_jk = _gate_tomography(
                nqubits=nqubits,
                gate=gate,
                nshots=int(1e4),
                noise_model=None,
                backend=backend,
                ancilla=ancilla,
            )
    else:
        matrix_jk = _gate_tomography(
            nqubits=nqubits,
            gate=gate,
            nshots=int(1e4),
            noise_model=None,
            backend=backend,
            ancilla=ancilla,
        )
        backend.assert_allclose(matrix_jk, ground_truth_matrix, rtol=1e-1, atol=1e-1)
        assert matrix_jk.shape == (4**nqubits, 4**nqubits)


def test_gate_tomography_ancilla_error(backend):
    ancilla = 3
    nqubits = 2
    gate_list = [gates.T(0), gates.TDG(0)]
    with pytest.raises(ValueError):
        _gate_tomography(
            nqubits=nqubits,
            gate=gate_list,
            nshots=int(1e4),
            noise_model=None,
            backend=backend,
            ancilla=ancilla,
        )


def test__get_swap_pairs():
    true_swap_pairs = [[(0, 2)], [(1, 2)], [(0, 2), (1, 3)]]
    for ancilla in range(3):
        [gates.T(0), gates.TDG(0)]

        nqubits = 2
        additional_qubits = 0 if ancilla is None else (1 if ancilla in (0, 1) else 2)
        circ = Circuit(nqubits + additional_qubits, density_matrix=True)
        swap_pairs = _get_swap_pairs(circ.nqubits, ancilla)

        assert true_swap_pairs[ancilla] == swap_pairs


@pytest.mark.parametrize(
    "target_gates",
    [
        [
            gates.SX(0),
            gates.RX(0, np.pi / 4),
            gates.PRX(0, np.pi, np.pi / 2),
            gates.Unitary(np.array([[1, 0], [0, 1]]), 0),
            gates.CY(0, 1),
        ],
        [gates.TOFFOLI(0, 1, 2)],
    ],
)
@pytest.mark.parametrize("pauli_liouville", [False, True])
def test_GST(backend, target_gates, pauli_liouville):
    T = np.array(
        [[1.0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]], dtype=np.complex128
    )
    T = backend.cast(T)
    target_matrices = [g.matrix(backend=backend) for g in target_gates]
    # superoperator representation of the target gates in the Pauli basis
    target_matrices = [
        to_pauli_liouville(m, normalize=True, backend=backend) for m in target_matrices
    ]

    gate_set = [
        ((g.__class__, list(g.parameters)) if g.parameters else g.__class__)
        for g in target_gates
    ]

    if len(target_gates) == 5:
        empty_1q, empty_2q, *approx_gates = GST(
            gate_set=gate_set,
            nshots=int(1e4),
            include_empty=True,
            pauli_liouville=pauli_liouville,
            backend=backend,
        )
        T_2q = backend.kron(T, T)
        for target, estimate in zip(target_matrices, approx_gates):
            if not pauli_liouville:
                G = empty_1q if estimate.shape[0] == 4 else empty_2q
                G_inv = backend.inv(G)
                T_matrix = T if estimate.shape[0] == 4 else T_2q
                estimate = T_matrix @ G_inv @ estimate @ G_inv
            backend.assert_allclose(
                target,
                estimate,
                atol=1e-1,
            )
    else:
        with pytest.raises(RuntimeError):
            empty_1q, empty_2q, *approx_gates = GST(
                gate_set=gate_set,
                nshots=int(1e4),
                include_empty=True,
                pauli_liouville=pauli_liouville,
                backend=backend,
            )


def test_GST_2qb_basis_op_diff_registers(backend):
    gate_set = [gates.T, gates.TDG, gates.S]
    with pytest.raises(RuntimeError):
        GST(
            gate_set=gate_set,
            two_qubit_basis_op_diff_registers=True,
            include_empty=False,
        )


@pytest.mark.parametrize(
    "gate_set",
    [
        [gates.T, gates.T, gates.T],
        [gates.CNOT],
        [gates.CNOT, gates.CNOT],
    ],
)
def test_GST_2qb_basis_op_diff_registers_incorrect_gates(backend, gate_set):
    with pytest.raises(RuntimeError):
        GST(
            gate_set=gate_set,
            two_qubit_basis_op_diff_registers=True,
            include_empty=False,
        )


def test_GST_2qb_basis_op_diff_registers_param_gates(backend):
    gate_set = [
        [gates.T, gates.TDG],
        [(gates.RX, [np.pi / 4]), (gates.RY, [np.pi / 3])],
        [(gates.Unitary, [np.eye(2)]), (gates.Unitary, [np.eye(2)])],
    ]

    ground_truth_matrices = [
        np.kron(gates.T(0).matrix(backend), gates.TDG(0).matrix(backend)),
        np.kron(
            gates.RX(0, np.pi / 4).matrix(backend),
            gates.RY(0, np.pi / 3).matrix(backend),
        ),
        np.eye(4),
    ]

    for _i in range(3):
        test_matrix = GST(
            gate_set=gate_set[_i],
            nshots=int(1e4),
            two_qubit_basis_op_diff_registers=True,
            include_empty=False,
            backend=backend,
        )
        ground_truth_matrix = GST(
            gate_set=[((gates.Unitary), ground_truth_matrices[_i])],
            nshots=int(1e4),
            include_empty=False,
            backend=backend,
        )
        backend.assert_allclose(test_matrix[0], ground_truth_matrix[0], atol=1e-1)


def test_GST_invertible_matrix(backend):

    ground_truths = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ]

    T = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    matrices = GST(
        gate_set=[gates.X, gates.Y, gates.Z],
        pauli_liouville=True,
        gauge_matrix=T,
        backend=backend,
    )

    for ground_truth, test_matrix in zip(ground_truths, matrices):
        backend.assert_allclose(test_matrix, ground_truth, atol=1e-1)


def test_GST_non_invertible_matrix(backend):
    T = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, -1, 0, 0]])
    with pytest.raises(ValueError):
        GST(gate_set=[], pauli_liouville=True, gauge_matrix=T, backend=backend)


def test_GST_with_transpiler(backend, star_connectivity):

    target_gates = [gates.SX(0), gates.Z(0), gates.CNOT(0, 1)]
    gate_set = [
        ((g.__class__, list(g.parameters)) if g.parameters else g.__class__)
        for g in target_gates
    ]
    # standard not transpiled GST
    empty_1q, empty_2q, *approx_gates = GST(
        gate_set=gate_set,
        nshots=int(1e4),
        include_empty=True,
        pauli_liouville=False,
        backend=backend,
        transpiler=None,
    )
    # define transpiler
    connectivity = star_connectivity()
    transpiler = Passes(
        connectivity=connectivity,
        passes=[
            Preprocessing(),
            Random(),
            Sabre(),
            Unroller(NativeGates.default(), backend=backend),
        ],
    )
    # transpiled GST
    T_empty_1q, T_empty_2q, *T_approx_gates = GST(
        gate_set=gate_set,
        nshots=int(1e4),
        include_empty=True,
        pauli_liouville=False,
        backend=backend,
        transpiler=transpiler,
    )

    backend.assert_allclose(empty_1q, T_empty_1q, atol=1e-1)
    backend.assert_allclose(empty_2q, T_empty_2q, atol=1e-1)
    for standard, transpiled in zip(approx_gates, T_approx_gates):
        backend.assert_allclose(standard, transpiled, atol=1e-1)


GAUGE = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
"""Gram matrix of the ideal fiducial states and measurements of a single qubit."""

MEASUREMENT_BASES = (gates.Z, gates.X, gates.Y, gates.Z)
"""Gates of the bases that measure :math:`I`, :math:`X`, :math:`Y` and :math:`Z`."""

PREPARATIONS = ((gates.I,), (gates.X,), (gates.H,), (gates.H, gates.S))
"""Gates that prepare :math:`|0\\rangle`, :math:`|1\\rangle`, :math:`|+\\rangle`
and :math:`|y+\\rangle`."""

TARGETS = [
    (gates.SX, (0,)),
    (gates.RX, (0, math.pi / 4)),
    (gates.PRX, (0, math.pi, math.pi / 2)),
    (gates.Unitary, (np.eye(2), 0)),
    (gates.CY, (0, 1)),
]
"""Gates, and their arguments, characterized in the tests."""


@pytest.mark.parametrize(
    "nqubits, target_gates, auxiliary, ground_truth",
    [
        (1, [gates.X(0)], [0], GROUND_TRUTH_GATE_X),
        (2, [gates.X(0), gates.T(1)], [0], GROUND_TRUTH_GATE_XT_0),
        (2, [gates.X(0), gates.T(1)], [1], GROUND_TRUTH_GATE_XT_1),
        (2, [], [0, 1], GROUND_TRUTH_NULL_2),
        (2, [gates.Unitary(np.eye(4), 0, 1)], [1], GROUND_TRUTH_I_2),
    ],
)
def test_call_auxiliary(backend, nqubits, target_gates, auxiliary, ground_truth):
    """Auxiliary qubits reset the qubits they replace before the circuit acts."""
    circuit = Circuit(nqubits)
    circuit.add(target_gates)

    matrix = GateSetTomography()(
        circuit, nshots=int(1e4), auxiliary=auxiliary, backend=backend
    )

    assert matrix.shape == (4**nqubits, 4**nqubits)
    backend.assert_allclose(matrix, ground_truth, rtol=1e-1, atol=1e-1)


def test_call_auxiliary_pauli_liouville(backend):
    """The Gram matrix is estimated without auxiliary qubits, thus it is invertible."""
    circuit = Circuit(1)
    circuit.add(gates.X(0))

    estimate = GateSetTomography()(
        circuit,
        nshots=int(1e4),
        auxiliary=[0],
        pauli_liouville=True,
        backend=backend,
    )

    # reset to the zero state, followed by the X gate
    target = np.zeros((4, 4))
    target[0, 0] = 1
    target[3, 0] = -1

    backend.assert_allclose(estimate, target, atol=1e-1)


def test_call_gauge_matrix(backend):
    """The estimate in the Pauli-Liouville representation is defined up to the gauge."""
    circuit = Circuit(1)
    circuit.add(gates.RX(0, math.pi / 3))

    target = to_pauli_liouville(
        circuit.unitary(backend), normalize=True, backend=backend
    )
    gauge = backend.cast(GAUGE, dtype=backend.complex128)

    # the exact Gram matrix is given, to isolate the effect of the gauge matrix
    kwargs = {
        "nshots": int(1e4),
        "pauli_liouville": True,
        "gram_matrix": GAUGE,
        "backend": backend,
    }
    gst = GateSetTomography()

    backend.assert_allclose(gst(circuit, **kwargs), target, atol=1e-1)
    backend.assert_allclose(
        gst(circuit, gauge_matrix=GAUGE, **kwargs), target, atol=1e-1
    )
    # the identity gauge matrix is a similarity transformation of the default one
    backend.assert_allclose(
        gst(circuit, gauge_matrix=np.eye(4), **kwargs),
        backend.inv(gauge) @ target @ gauge,
        atol=1e-1,
    )


@pytest.mark.parametrize(
    "gauge",
    [
        np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, -1, 0, 0]]),
        np.zeros((4, 4)),
        np.eye(3),
    ],
)
def test_call_gauge_matrix_errors(backend, gauge):
    """The gauge matrix must be an invertible 4 x 4 matrix."""
    with pytest.raises(ValueError):
        GateSetTomography()(
            Circuit(1), pauli_liouville=True, gauge_matrix=gauge, backend=backend
        )


@pytest.mark.parametrize("nqubits", [1, 2])
def test_call_gram_matrix(backend, nqubits):
    """The matrix of an empty circuit is the Gram matrix of the ideal fiducials."""
    gram = GateSetTomography()(Circuit(nqubits), nshots=int(1e4), backend=backend)

    ideal = GAUGE if nqubits == 1 else np.kron(GAUGE, GAUGE)

    backend.assert_allclose(gram, ideal, atol=1e-1)


def test_call_gram_matrix_argument(backend):
    """A given Gram matrix is used, instead of estimating it again."""
    circuit = Circuit(1)
    circuit.add(gates.RX(0, math.pi / 3))

    target = to_pauli_liouville(
        circuit.unitary(backend), normalize=True, backend=backend
    )

    gst = GateSetTomography()
    with patch.object(
        GateSetTomography,
        "execute",
        autospec=True,
        side_effect=GateSetTomography.execute,
    ) as execute:
        estimated = gst(circuit, nshots=int(1e4), pauli_liouville=True, backend=backend)
        given = gst(
            circuit,
            nshots=int(1e4),
            pauli_liouville=True,
            gram_matrix=GAUGE,
            backend=backend,
        )

    # the Gram matrix and the circuit, then only the circuit
    assert [len(call.args[1]) for call in execute.call_args_list] == [12, 12, 12]
    backend.assert_allclose(estimated, target, atol=1e-1)
    backend.assert_allclose(given, target, atol=1e-1)


@pytest.mark.parametrize("gate_class, arguments", TARGETS)
def test_call_matrix(backend, gate_class, arguments):
    """The estimate is the Pauli-Liouville matrix in the basis of the fiducials."""
    gate = gate_class(*arguments)
    circuit = Circuit(len(gate.qubits))
    circuit.add(gate)

    matrix = GateSetTomography()(circuit, nshots=int(1e4), backend=backend)

    ideal = GAUGE if circuit.nqubits == 1 else np.kron(GAUGE, GAUGE)
    target = to_pauli_liouville(gate.matrix(backend), normalize=True, backend=backend)

    assert matrix.shape == (4**circuit.nqubits, 4**circuit.nqubits)
    backend.assert_allclose(
        matrix, target @ backend.cast(ideal, dtype=backend.complex128), atol=1e-1
    )


def test_call_noise_model(backend):
    """A fully depolarizing noise destroys all the information, but the identity."""
    circuit = Circuit(1)
    circuit.add(gates.H(0))

    noise_model = NoiseModel()
    noise_model.add(DepolarizingError(1.0))

    matrix = GateSetTomography()(
        circuit, nshots=int(1e4), noise_model=noise_model, backend=backend
    )

    exact = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    backend.assert_allclose(matrix, exact, atol=1e-1)


@pytest.mark.parametrize(
    "first, second",
    [
        (gates.T(0), gates.TDG(1)),
        (gates.RX(0, math.pi / 4), gates.RY(1, math.pi / 3)),
        (gates.Unitary(np.eye(2), 0), gates.Unitary(np.eye(2), 1)),
    ],
)
def test_call_parallel_gates(backend, first, second):
    """Single-qubit gates on different qubits are characterized simultaneously."""
    parallel = Circuit(2)
    parallel.add([first, second])

    joint = Circuit(2)
    joint.add(
        gates.Unitary(backend.kron(first.matrix(backend), second.matrix(backend)), 0, 1)
    )

    gst = GateSetTomography()
    backend.assert_allclose(
        gst(parallel, nshots=int(1e4), backend=backend),
        gst(joint, nshots=int(1e4), backend=backend),
        atol=1e-1,
    )


@pytest.mark.parametrize("gate_class, arguments", TARGETS)
def test_call_pauli_liouville(backend, gate_class, arguments):
    """The estimate agrees with the exact Pauli-Liouville matrix of the gate."""
    gate = gate_class(*arguments)
    circuit = Circuit(len(gate.qubits))
    circuit.add(gate)

    estimate = GateSetTomography()(
        circuit, nshots=int(1e4), pauli_liouville=True, backend=backend
    )

    target = to_pauli_liouville(gate.matrix(backend), normalize=True, backend=backend)
    backend.assert_allclose(estimate, target, atol=1e-1)


@pytest.mark.parametrize("pauli_liouville", [False, True])
@pytest.mark.parametrize("nqubits, ncircuits", [(1, 12), (2, 240)])
def test_call_qibolab(nqubits, ncircuits, pauli_liouville):
    """The default transpiler of ``qibolab`` places all the circuits in the same qubits.

    The outcomes of the dummy platform are not physical, so only the execution is tested,
    and the ideal Gram matrix is given to avoid inverting a random one.
    """
    pytest.importorskip("qibolab")
    backend = construct_backend("qibolab", platform="dummy")

    circuit = Circuit(nqubits)
    circuit.add(gates.X(qubit) for qubit in range(nqubits))

    with patch.object(
        GateSetTomography,
        "execute",
        autospec=True,
        side_effect=GateSetTomography.execute,
    ) as execute:
        matrix = GateSetTomography()(
            circuit,
            nshots=10,
            pauli_liouville=pauli_liouville,
            gram_matrix=GAUGE if nqubits == 1 else np.kron(GAUGE, GAUGE),
            backend=backend,
        )

    executed = execute.call_args_list[-1].args[1]
    placements = {tuple(map(str, gst_circuit.wire_names)) for gst_circuit in executed}

    assert matrix.shape == (4**nqubits, 4**nqubits)
    assert len(executed) == ncircuits
    assert len(placements) == 1


def test_call_transpiler(backend, star_connectivity):
    """Transpiled circuits give the same estimates."""
    transpiler = Passes(
        connectivity=star_connectivity(),
        passes=[
            Preprocessing(),
            Random(),
            Sabre(),
            Unroller(NativeGates.default(), backend=backend),
        ],
    )

    gst = GateSetTomography()
    for target in (gates.SX(0), gates.CNOT(0, 1)):
        circuit = Circuit(len(target.qubits))
        circuit.add(target)

        backend.assert_allclose(
            gst(circuit, nshots=int(1e4), transpiler=transpiler, backend=backend),
            gst(circuit, nshots=int(1e4), backend=backend),
            atol=1e-1,
        )


@pytest.mark.parametrize("nqubits", [1, 2])
def test_circuits(nqubits):
    """Circuits prepare a fiducial state, apply the circuit, and measure a Pauli string."""
    circuit = Circuit(nqubits)
    circuit.add(gates.T(qubit) for qubit in range(nqubits))
    original = list(circuit.queue)

    gst = GateSetTomography()
    circuits = gst.circuits(circuit)

    digits = tuple(product(range(4), repeat=nqubits))
    expected = [(state, pauli) for state in digits for pauli in digits[1:]]

    assert isinstance(gst, Tomography)
    assert len(circuits) == len(expected) == 4**nqubits * (4**nqubits - 1)

    for gst_circuit, (state, pauli) in zip(circuits, expected):
        preparation = [
            (gate, (qubit,))
            for qubit, digit in enumerate(state)
            for gate in PREPARATIONS[digit]
        ]
        operation = [(gates.T, (qubit,)) for qubit in range(nqubits)]
        queue = [(type(gate), gate.qubits) for gate in gst_circuit.queue]

        assert gst_circuit.nqubits == nqubits
        assert gst_circuit.density_matrix
        # the measurements are preceded by the gates of the change of basis
        assert queue[: len(preparation + operation)] == preparation + operation
        assert len(gst_circuit.measurements) == nqubits
        for qubit, measured in enumerate(gst_circuit.measurements):
            expected_basis = gates.M(qubit, basis=MEASUREMENT_BASES[pauli[qubit]]).basis
            assert measured.qubits == (qubit,)
            assert [(type(gate), gate.qubits) for gate in measured.basis] == [
                (type(gate), gate.qubits) for gate in expected_basis
            ]

    # the given circuit is not modified
    assert list(circuit.queue) == original
    assert not circuit.measurements


@pytest.mark.parametrize(
    "nqubits, auxiliary, swaps",
    [
        (1, [0], [(0, 1)]),
        (2, [0], [(0, 2)]),
        (2, [1], [(1, 2)]),
        (2, [0, 1], [(0, 2), (1, 3)]),
        (2, [1, 0], [(1, 2), (0, 3)]),
    ],
)
def test_circuits_auxiliary(nqubits, auxiliary, swaps):
    """Auxiliary qubits are appended, and swapped with the qubits they replace."""
    circuits = GateSetTomography().circuits(Circuit(nqubits), auxiliary)

    assert len(circuits) == 4**nqubits * (4**nqubits - 1)
    for gst_circuit in circuits:
        assert gst_circuit.nqubits == nqubits + len(auxiliary)
        assert [
            gate.qubits for gate in gst_circuit.queue if isinstance(gate, gates.SWAP)
        ] == swaps
        # only the qubits of the circuit are measured
        assert [gate.qubits for gate in gst_circuit.measurements] == [
            (qubit,) for qubit in range(nqubits)
        ]


def test_circuits_errors():
    """The circuit must be on one or two qubits, without measurements."""
    gst = GateSetTomography()

    measured = Circuit(1)
    measured.add(gates.M(0))
    with pytest.raises(ValueError):
        gst.circuits(measured)

    toffoli = Circuit(3)
    toffoli.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ValueError):
        gst.circuits(toffoli)

    for nqubits, auxiliary in [(1, [1]), (2, [2]), (2, [-1]), (2, [0, 0]), (2, [0.5])]:
        with pytest.raises(ValueError):
            gst.circuits(Circuit(nqubits), auxiliary)


def test_gst_deprecated(backend, caplog):
    """The old ``GST`` function logs a warning that points to the new class."""
    with caplog.at_level(logging.WARNING):
        matrices = GST([gates.X], nshots=int(1e2), backend=backend)

    assert "deprecated" in caplog.text
    assert "GateSetTomography" in caplog.text
    assert len(matrices) == 1
