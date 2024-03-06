from functools import reduce
from itertools import repeat

import numpy as np
import pytest

import qibo
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import DepolarizingError, NoiseModel
from qibo.quantum_info import to_pauli_liouville
from qibo.tomography.gate_set_tomography import (
    GST,
    GST_execute_circuit,
    _get_observable,
    execute_GST,
    measurement_basis,
    prepare_states,
    reset_register,
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
def test_prepare_states(k, nqubits):
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
            prepared_states = prepare_states(k, nqubits)
    else:
        prepared_states = prepare_states(k, nqubits)
        for groundtruth, gate in zip(correct_gates[nqubits][k], prepared_states):
            _compare_gates(groundtruth, gate)


@pytest.mark.parametrize(
    "j,nqubits",
    INDEX_NQUBITS,
)
def test_measurement_basis(j, nqubits):
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
            prepared_gates = measurement_basis(j, nqubits)
    else:
        prepared_gates = measurement_basis(j, nqubits)
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
            (qibo.symbols.I(0),),
            (qibo.symbols.Z(0),),
            (qibo.symbols.Z(0),),
            (qibo.symbols.Z(0),),
        ],
        2: [
            (qibo.symbols.I(0), qibo.symbols.I(1)),
            (qibo.symbols.I(0), qibo.symbols.Z(1)),
            (qibo.symbols.I(0), qibo.symbols.Z(1)),
            (qibo.symbols.I(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.I(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.I(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.Z(1)),
            (qibo.symbols.Z(0), qibo.symbols.I(1)),
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


def test_reset_register_valid_tuple_1qb():
    # Test for valid tuple
    nqubits = 1
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))

    invert_register = (0,)
    inverse_circuit = reset_register(test_circuit, invert_register)

    correct_gates = [
        [gates.H(0)],
    ]
    for groundtruth, gate in zip(correct_gates, inverse_circuit.queue):
        assert isinstance(gate, type(groundtruth[0]))


def test_reset_register0_twoqubitcircuit():
    # Test resetting qubit 0
    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.S(1))

    inverse_circuit = reset_register(test_circuit, (0,))

    correct_gates = [
        [gates.H(0)],
    ]

    for groundtruth, gate in zip(correct_gates, inverse_circuit.queue):
        assert isinstance(gate, type(groundtruth[0]))


def test_reset_register0_singlequbitcircuit():
    # Test resetting qubit 0

    nqubits = 1
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.RX(0, np.pi / 3))

    inverse_circuit = reset_register(test_circuit, (0,))

    correct_gates = [
        [gates.RX(0, np.pi / 3).dagger()],
        [gates.H(0)],
    ]

    for groundtruth, gate in zip(correct_gates, inverse_circuit.queue):
        assert isinstance(gate, type(groundtruth[0]))


def test_reset_register1_twoqubitcircuit():
    # Test resetting qubit 1
    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.S(1))

    inverse_circuit = reset_register(test_circuit, (1,))

    correct_gates = [[gates.S(0).dagger()]]

    for groundtruth, gate in zip(correct_gates, inverse_circuit.queue):
        assert isinstance(gate, type(groundtruth[0]))


def test_reset_register1_singlequbitcircuit():
    # Test resetting qubit 1
    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.S(0))
    test_circuit.add(gates.RX(1, np.pi / 3))

    inverse_circuit = reset_register(test_circuit, (1,))

    correct_gates = [[gates.RX(1, np.pi / 3).dagger()]]

    for groundtruth, gate in zip(correct_gates, inverse_circuit.queue):
        assert isinstance(gate, type(groundtruth[0]))


def test_reset_register_2():
    # Test resetting both qubits
    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.CNOT(0, 1))

    inverse_circuit = reset_register(test_circuit, (0, 1))

    correct_gates = [
        [gates.CNOT(0, 1)],
        [gates.H(0)],
    ]
    for groundtruth, gate in zip(correct_gates, inverse_circuit.queue):
        assert isinstance(gate, type(groundtruth[0]))


@pytest.mark.parametrize("a, b", [(0, 2), (1, 2), (2, 3)])
def test_reset_register_invalid_tuple(a, b):
    # Test resetting both qubits

    nqubits = 2
    test_circuit = qibo.models.Circuit(nqubits)
    test_circuit.add(gates.H(0))
    test_circuit.add(gates.CNOT(0, 1))

    # Check if NameError is raised
    with pytest.raises(NameError):
        inverse_circuit = reset_register(test_circuit, (a, b))


def test_GST(backend):
    target_gates = [gates.SX(0), gates.Z(0), gates.CY(0, 1)]
    target_matrices = [g.matrix() for g in target_gates]
    # superoperator representation of the target gates in the pauli basis
    target_matrices = [to_pauli_liouville(m, normalize=True) for m in target_matrices]
    gate_set = {g.__class__ for g in target_gates}
    lam = 1e-10
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    empty_1q, empty_2q, *approx_gates = GST(
        gate_set, noise_model=depol, include_empty=True, backend=backend
    )
    for target, estimate in zip(target_matrices, approx_gates):
        backend.assert_allclose(target, estimate, atol=1e-6)
