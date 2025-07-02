from functools import reduce
from itertools import repeat

import numpy as np
import pytest
from sympy import S

from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import DepolarizingError, NoiseModel
from qibo.quantum_info.superoperator_transformations import to_pauli_liouville
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
    correct_observables = {
        1: [
            (S(1),),
            (symbols.Z(0),),
            (symbols.Z(0),),
            (symbols.Z(0),),
        ],
        2: [
            (S(1), S(1)),
            (S(1), symbols.Z(1)),
            (S(1), symbols.Z(1)),
            (S(1), symbols.Z(1)),
            (symbols.Z(0), S(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), S(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), S(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), symbols.Z(1)),
            (symbols.Z(0), symbols.Z(1)),
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


def test__extract_nqubits():
    correct_nqubits = [1, 1, 2, 3]
    gates_to_test = [gates.Z, (gates.RX, [np.pi / 2]), gates.CNOT, gates.TOFFOLI]
    for idx in range(0, len(gates_to_test)):
        if idx < 3:
            assert _extract_nqubits(gates_to_test[idx]) == correct_nqubits[idx]
        else:
            with pytest.raises(RuntimeError):
                nqubits = _extract_nqubits(gates_to_test[idx])


@pytest.mark.parametrize(
    "idx",
    (
        5
        * [
            None,
        ],
        [2, 2, 2, (2, 3), (2, 3)],
    ),
)
def test__extract_gate(idx):

    chosen_idx = idx

    gates_to_test = [
        (gates.T),
        ((gates.RX, [np.pi / 2])),
        ((gates.Unitary, [np.eye(2)])),
        ((gates.CRX, [np.pi / 3])),
        ((gates.Unitary, [np.eye(4)])),
    ]

    if all(i is None for i in chosen_idx):
        correct_gates = [
            gates.T(0),
            gates.RX(0, np.pi / 2),
            gates.Unitary(np.eye(2), 0),
            gates.CRX(0, 1, np.pi / 3),
            gates.Unitary(np.eye(4), 0, 1),
        ]
    else:
        correct_gates = [
            gates.T(2),
            gates.RX(2, np.pi / 2),
            gates.Unitary(np.eye(2), 2),
            gates.CRX(2, 3, np.pi / 3),
            gates.Unitary(np.eye(4), 2, 3),
        ]

    for _i in range(len(gates_to_test)):
        extracted_gate, _ = _extract_gate(gates_to_test[_i], idx=chosen_idx[_i])

        assert extracted_gate.qubits == correct_gates[_i].qubits
        if _i in (0, 1):
            assert extracted_gate.init_kwargs == correct_gates[_i].init_kwargs
            assert extracted_gate.parameters == correct_gates[_i].parameters
        elif _i == 2:
            assert (
                extracted_gate.init_args[0].all()
                == correct_gates[_i].init_args[0].all()
            )
        elif _i == 3:
            assert extracted_gate.control_qubits == correct_gates[_i].control_qubits
            assert extracted_gate.target_qubits == correct_gates[_i].target_qubits
            assert extracted_gate.parameters == correct_gates[_i].parameters
        elif _i == 4:
            assert (
                extracted_gate.init_args[0].all()
                == correct_gates[_i].init_args[0].all()
            )
            assert extracted_gate.target_qubits == correct_gates[_i].target_qubits


@pytest.mark.parametrize(
    "gate, error_type",
    [
        (((gates.RX), [np.eye(2)]), ValueError),
        (((gates.Unitary), np.array([[1, 2], [3, 4]])), ValueError),
        ((gates.TOFFOLI), RuntimeError),
    ],
)
def test__extract_gate_error(gate, error_type):
    with pytest.raises(error_type):
        extracted_gate, _ = _extract_gate(gate)


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


def test_gate_tomography_apply_ancillas(backend):
    nqubits = 2
    gate_list = [gates.T(0), gates.TDG(0), gates.S(0)]
    with pytest.raises(ValueError):
        matrix_jk = _gate_tomography(
            nqubits=nqubits,
            gate=gate_list,
            nshots=int(1e4),
            noise_model=None,
            backend=backend,
        )


@pytest.mark.parametrize(
    "ancilla",
    [
        (3),
    ],
)
def test_gate_tomography_ancilla_error(backend, ancilla):
    nqubits = 2
    gate_list = [gates.T(0), gates.TDG(0)]
    with pytest.raises(ValueError):
        matrix_jk = _gate_tomography(
            nqubits=nqubits,
            gate=gate_list,
            nshots=int(1e4),
            noise_model=None,
            backend=backend,
            ancilla=ancilla,
        )


def test__get_swap_pairs():
    true_swap_pairs = [[(0, 2)], [(1, 2)], [(0, 2), (1, 3)]]
    for ancilla in range(0, 3):
        gate_list = [gates.T(0), gates.TDG(0)]

        nqubits = 2
        additional_qubits = 0 if ancilla is None else (1 if ancilla in (0, 1) else 2)
        circ = Circuit(nqubits + additional_qubits, density_matrix=True)
        swap_pairs = _get_swap_pairs(circ, ancilla)

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
        T_2q = backend.np.kron(T, T)
        for target, estimate in zip(target_matrices, approx_gates):
            if not pauli_liouville:
                G = empty_1q if estimate.shape[0] == 4 else empty_2q
                G_inv = backend.np.linalg.inv(G)
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
        if len(gate_set) > 2:
            matrices = GST(
                gate_set=gate_set,
                two_qubit_basis_op_diff_registers=True,
                include_empty=False,
            )


@pytest.mark.parametrize(
    "gate_set",
    [
        [gates.T, gates.CNOT],
        [gates.CNOT, gates.T],
        [gates.CNOT, gates.CNOT],
    ],
)
def test_GST_2qb_basis_op_diff_registers_wrong_gates(backend, gate_set):
    with pytest.raises(RuntimeError):
        matrices = GST(
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
        np.kron(gates.T(0).matrix(), gates.TDG(0).matrix()),
        np.kron(gates.RX(0, np.pi / 4).matrix(), gates.RY(0, np.pi / 3).matrix()),
        np.eye(4),
    ]

    for _i in range(0, 3):
        test_matrix = GST(
            gate_set=gate_set[_i],
            nshots=int(1e4),
            two_qubit_basis_op_diff_registers=True,
            include_empty=False,
        )
        ground_truth_matrix = GST(
            gate_set=[((gates.Unitary), ground_truth_matrices[_i])],
            nshots=int(1e4),
            include_empty=False,
        )
        backend.assert_allclose(test_matrix[0], ground_truth_matrix[0], atol=1e-1)


def test_GST_invertible_matrix():
    T = np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]])
    matrices = GST(gate_set=[], pauli_liouville=True, gauge_matrix=T)
    assert True


def test_GST_non_invertible_matrix():
    T = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, -1, 0, 0]])
    with pytest.raises(ValueError):
        matrices = GST(gate_set=[], pauli_liouville=True, gauge_matrix=T)


def test_GST_with_transpiler(backend, star_connectivity):
    import networkx as nx

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
