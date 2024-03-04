import random
from functools import reduce
from itertools import repeat

import numpy as np
import pytest

import qibo
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.noise import DepolarizingError, NoiseModel
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


@pytest.mark.parametrize(
    "j,nqubits",
    random.choices(INDEX_NQUBITS[:-2], k=5),
)
def test_GST_execute_circuit(backend, j, nqubits):
    # pick k
    k = random.choice(range(4**nqubits))
    # pick a gate
    gate = random.choice(
        [
            None,
            gates.SX(0),
            gates.CZ(0, 1),
            gates.RY(0, theta=2 * np.pi * np.random.rand()),
            gates.CNOT(1, 0),
        ]
    )
    # define a noise model
    lam = 0.2
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    # create a test circuit
    circuit = Circuit(nqubits, density_matrix=True)
    # prepare j-k settings
    initial_state = prepare_states(k, nqubits)
    meas_basis = measurement_basis(j, nqubits)
    circuit.add(initial_state)
    if gate is not None:
        circuit.add(gate)
    circuit.add(meas_basis)
    circuit = depol.apply(circuit)
    result = backend.execute_circuit(circuit, nshots=1000)
    # build the observable
    obs = _get_observable(j, nqubits)
    expv = result.expectation_from_samples(obs)
    backend.assert_allclose(
        expv,
        GST_execute_circuit(circuit, k, j, nshots=1000, backend=backend),
        atol=1e-1,
    )


@pytest.mark.parametrize(
    "gate",
    [
        None,
        gates.SX(0),
        gates.CZ(0, 1),
        gates.RY(0, theta=2 * np.pi * np.random.rand()),
        gates.CNOT(1, 0),
    ],
)
def test_execute_GST(backend, gate):
    if gate is None:
        nqubits = 1
        target = backend.zero_state(nqubits)
    else:
        target = gate.matrix()
        nqubits = int(np.log2(target.shape[0]))
    # define a noise model
    lam = 1e-2
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    estimate = execute_GST(
        nqubits=nqubits, gate=gate, noise_model=depol, backend=backend
    )
    backend.assert_allclose(target, estimate, atol=1e-2)


def test_GST_one_qubit_empty_circuit(backend):
    nqubits = 1
    control_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=None, backend=backend
    )
    test_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=None, backend=backend
    )
    backend.assert_allclose(control_result, test_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_empty_circuit(backend):
    nqubits = 2
    np.random.seed(42)
    control_result = execute_GST(nqubits)
    np.random.seed(42)
    test_result = execute_GST(nqubits)

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_Hgate(backend):
    nqubits = 1
    test_gate = gates.H(0)
    control_result = execute_GST(nqubits, gate=test_gate)
    test_result = execute_GST(nqubits, gate=test_gate)
    backend.assert_allclose(control_result, test_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_RXgate(backend):
    nqubits = 1
    test_gate = gates.RX(0, np.pi / 7)
    control_result = execute_GST(nqubits, gate=test_gate)
    test_result = execute_GST(nqubits, gate=test_gate)
    backend.assert_allclose(control_result, test_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_CNOTgate(backend):
    nqubits = 2
    test_gate = gates.CNOT(0, 1)
    np.random.seed(42)
    control_result = execute_GST(nqubits, gate=test_gate, backend=backend)
    np.random.seed(42)
    test_result = execute_GST(nqubits, gate=test_gate, backend=backend)

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_CRXgate(backend):
    nqubits = 2
    test_gate = gates.CRX(0, 1, np.pi / 7)
    np.random.seed(42)
    control_result = execute_GST(nqubits, gate=test_gate)
    np.random.seed(42)
    test_result = execute_GST(nqubits, gate=test_gate)
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_gate_with_valid_reset_register_tuple(backend):
    nqubits = 1
    invert_register = (0,)
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_gate_with_valid_reset_register_tuple(backend):
    nqubits = 2
    invert_register = (1,)
    np.random.seed(42)
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_with_param_gate_with_valid_reset_register_tuple(backend):
    nqubits = 1
    test_gate = gates.RX(0, np.pi / 7)
    invert_register = (0,)
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register
    )
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_param_gate_with_valid_reset_register_tuple(backend):
    nqubits = 2
    test_gate = gates.CNOT(0, 1)
    invert_register = (1,)
    np.random.seed(42)
    control_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits=nqubits, gate=None, invert_register=invert_register, backend=backend
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_two_qubit_with_gate_with_valid_reset_register_tuple(backend):
    nqubits = 2
    test_gate = gates.CZ(0, 1)
    invert_register = (0, 1)
    np.random.seed(42)
    control_result = execute_GST(
        nqubits=nqubits,
        gate=test_gate,
        invert_register=invert_register,
        backend=backend,
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits=nqubits,
        gate=test_gate,
        invert_register=invert_register,
        backend=backend,
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("a, b", [(0, 2), (1, 2), (2, 3)])
def test_GST_two_qubit_with_gate_with_invalid_reset_register_tuple(a, b):
    nqubits = 2
    test_gate = gates.CZ(0, 1)
    invert_register = (a, b)
    with pytest.raises(NameError):
        result = execute_GST(
            nqubits=nqubits, gate=test_gate, invert_register=invert_register
        )


def test_GST_empty_circuit_with_invalid_qb(backend):
    nqubits = 3
    # Check if ValueError is raised
    with pytest.raises(ValueError, match="nqubits needs to be either 1 or 2"):
        result = execute_GST(
            nqubits, gate=None, invert_register=None, noise_model=None, backend=backend
        )


def test_GST_with_gate_with_invalid_qb(backend):
    nqubits = 3
    test_gate = gates.CNOT(0, 1)

    # Check if ValueError is raised
    with pytest.raises(ValueError):
        result = execute_GST(
            nqubits,
            gate=test_gate,
            invert_register=None,
            noise_model=None,
            backend=backend,
        )


def test_GST_with_gate_with_invalid_qb(backend):
    nqubits = 2
    test_gate = gates.H(0)

    # Check if ValueError is raised
    with pytest.raises(ValueError):
        result = execute_GST(
            nqubits,
            gate=test_gate,
            invert_register=None,
            noise_model=None,
            backend=backend,
        )


def test_GST_one_qubit_empty_circuit_with_noise(backend):
    lam = 0.5
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    noise_model = depol
    nqubits = 1
    control_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )
    test_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )
    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST_one_qubit_empty_circuit_with_noise(backend):
    nshots = int(1e4)
    lam = 0.5
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    noise_model = depol
    nqubits = 2
    np.random.seed(42)
    control_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )
    np.random.seed(42)
    test_result = execute_GST(
        nqubits, gate=None, invert_register=None, noise_model=depol, backend=backend
    )

    backend.assert_allclose(test_result, control_result, rtol=5e-2, atol=5e-2)


def test_GST(backend):
    target_gates = [gates.SX(0), gates.Z(0), gates.CY(0, 1)]
    target_matrices = [g.matrix() for g in target_gates]
    gate_set = {g.__class__ for g in target_gates}
    lam = 0.1
    depol = NoiseModel()
    depol.add(DepolarizingError(lam))
    empty_1q, empty_2q, *approx_gates = GST(
        gate_set, noise_model=depol, include_empty=True, backend=backend
    )
