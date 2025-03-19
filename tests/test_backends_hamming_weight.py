"""Tests for HammingWeight backend."""

from itertools import product

import numpy as np
import pytest
from scipy.special import binom

from qibo import Circuit, gates, get_backend, set_backend
from qibo.backends import HammingWeightBackend, NumpyBackend, _get_engine_name
from qibo.quantum_info import random_statevector

numpy_bkd = NumpyBackend()


def construct_hamming_weight_backend(backend):
    return HammingWeightBackend(_get_engine_name(backend))


def test_set_backend(backend):
    hamming_bkd = construct_hamming_weight_backend(backend)
    platform = _get_engine_name(backend)
    set_backend("hamming_weight", platform=platform)
    assert isinstance(get_backend(), HammingWeightBackend)
    global_platform = get_backend().platform
    assert global_platform == platform


def test_global_backend(backend):
    construct_hamming_weight_backend(backend)
    set_backend(backend.name, platform=backend.platform)
    hamming_bkd = HammingWeightBackend()
    target = get_backend().name if backend.name == "numpy" else get_backend().platform
    assert hamming_bkd.platform == target


def get_full_initial_state(state, weight, nqubits, backend):
    if (
        backend._dict_indexes is None
        or list(backend._dict_indexes.keys())[0].count("1") != weight
    ):
        backend._dict_indexes = backend._get_lexicographical_order(nqubits, weight)

    full_state = backend.engine.np.zeros(2**nqubits, dtype=backend.engine.np.complex128)
    for i, j in backend._dict_indexes.values():
        full_state[j] = state[i]

    return full_state


SINGLE_QUBIT_GATES = ["Z", "S", "SDG", "T", "TDG", "I", "RZ"]


@pytest.mark.parametrize("gate", SINGLE_QUBIT_GATES)
@pytest.mark.parametrize("weight", [1, 2, 3])
def test_single_qubit_gates(backend, gate, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(3, weight))
    initial_state = random_statevector(dim, backend=backend, seed=1237)
    initial_state_full = get_full_initial_state(initial_state, weight, 3, hamming_bkd)

    qubits = [0, 2]
    if gate == "RZ":
        theta = 0.123
        gate1 = gates.RZ(qubits[0], theta)
        gate2 = gates.RZ(qubits[1], theta)
    else:
        gate1 = getattr(gates, gate)(qubits[0])
        gate2 = getattr(gates, gate)(qubits[1])

    circuit = Circuit(3, density_matrix=False)
    circuit.add(gate1)
    circuit.add(gate2)

    result = backend.execute_circuit(circuit, initial_state=initial_state_full)
    state = result.state()

    hamming_result = hamming_bkd.execute_circuit(
        circuit, weight=weight, initial_state=initial_state
    )

    hamming_full_state = hamming_result.full_state()

    backend.assert_allclose(hamming_full_state, state, atol=1e-8)

    hamming_result._backend._dict_indexes = None
    assert result.symbolic() == hamming_result.symbolic()
    assert result.symbolic(max_terms=1) == hamming_result.symbolic(max_terms=1)

    # with controls
    circuit = Circuit(3, density_matrix=False)
    if gate == "RZ":
        theta = 0.123
        gate1 = gates.RZ(qubits[0], theta)
        gate2 = gates.RZ(qubits[1], theta)
    else:
        gate1 = getattr(gates, gate)(qubits[0])
        gate2 = getattr(gates, gate)(qubits[1])

    circuit.add(gate1.controlled_by(1))
    circuit.add(gate2.controlled_by(0))

    initial_state_full = get_full_initial_state(initial_state, weight, 3, hamming_bkd)
    result = backend.execute_circuit(circuit, initial_state=initial_state_full)
    state = result.state()

    hamming_result = hamming_bkd.execute_circuit(
        circuit, weight=weight, initial_state=initial_state
    )
    hamming_result._backend._dict_indexes = None
    hamming_full_state = hamming_result.full_state()

    backend.assert_allclose(hamming_full_state, state, atol=1e-8)
    assert result.symbolic() == hamming_result.symbolic()


TWO_QUBIT_GATES = ["CZ", "SWAP", "iSWAP", "SiSWAP", "SiSWAPDG", "FSWAP", "SYC", "RZZ"]


@pytest.mark.parametrize("weight", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("gate", TWO_QUBIT_GATES)
def test_two_qubit_gates(backend, gate, weight):
    nqubits = 5
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(nqubits, weight))
    initial_state = random_statevector(dim, backend=backend, seed=1237)
    initial_state_full = get_full_initial_state(
        initial_state, weight, nqubits, hamming_bkd
    )

    qubits1 = [0, 1]
    qubits2 = [2, 1]
    if gate == "RZZ":
        theta = 0.123
        gate1 = gates.RZZ(*qubits1, theta)
        gate2 = gates.RZZ(*qubits2, theta)
    else:
        gate1 = getattr(gates, gate)(*qubits1)
        gate2 = getattr(gates, gate)(*qubits2)

    circuit = Circuit(nqubits, density_matrix=False)
    circuit.add(gate1)
    circuit.add(gate2)

    result = backend.execute_circuit(circuit, initial_state=initial_state_full)
    state = result.state()

    hamming_result = hamming_bkd.execute_circuit(
        circuit, weight=weight, initial_state=initial_state
    )
    hamming_full_state = hamming_result.full_state()

    backend.assert_allclose(hamming_full_state, state, atol=1e-8)

    hamming_result._backend._dict_indexes = None
    assert result.symbolic() == hamming_result.symbolic()

    if len(gate1.control_qubits) == 0:
        circuit = Circuit(nqubits, density_matrix=False)
        circuit.add(gate1.controlled_by(2))
        circuit.add(gate2.controlled_by(0))

        initial_state_full = get_full_initial_state(
            initial_state, weight, nqubits, hamming_bkd
        )
        result = numpy_bkd.execute_circuit(circuit, initial_state=initial_state_full)
        state = result.state()
        state = backend.cast(state)

        hamming_result = hamming_bkd.execute_circuit(
            circuit, weight=weight, initial_state=initial_state
        )
        hamming_full_state = hamming_result.full_state()

        backend.assert_allclose(hamming_full_state, state, atol=1e-8)

        assert result.symbolic() == hamming_result.symbolic()


@pytest.mark.parametrize("weight", [1, 2, 3])
def test_ccz(backend, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(3, weight))
    initial_state = random_statevector(dim, backend=backend, seed=1237)

    qubits1 = [0, 1, 2]
    qubits2 = [2, 1, 0]
    gate1 = gates.CCZ(*qubits1)
    gate2 = gates.CCZ(*qubits2)

    circuit = Circuit(3, density_matrix=False)
    circuit.add(gate1)
    circuit.add(gate2)
    hamming_result = hamming_bkd.execute_circuit(
        circuit, weight=weight, initial_state=initial_state
    )
    hamming_state = hamming_result.state()
    hamming_full_state = hamming_result.full_state()
    initial_state_full = get_full_initial_state(initial_state, weight, 3, hamming_bkd)
    result = backend.execute_circuit(circuit, initial_state=initial_state_full)
    state = result.state()
    state = backend.cast(state)

    backend.assert_allclose(hamming_full_state, state, atol=1e-8)

    hamming_bkd._dict_indexes = None
    state = hamming_bkd._apply_gate_CCZ(gate1, initial_state, 3, weight)
    state = hamming_bkd._apply_gate_CCZ(gate2, state, 3, weight)
    backend.assert_allclose(hamming_state, state, atol=1e-8)


@pytest.mark.parametrize("weight", [1, 2, 3, 4])
def test_apply_gate_n_qubit_single(backend, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    nqubits = 4
    dim = int(binom(nqubits, weight))
    initial_state = random_statevector(dim, backend=backend, seed=1237)
    initial_state_full = get_full_initial_state(
        initial_state, weight, nqubits, hamming_bkd
    )

    gate = gates.Z(0).controlled_by(1)

    state = backend.apply_gate(gate, initial_state_full, nqubits)
    hamming_state = hamming_bkd._apply_gate_n_qubit(
        gate, initial_state, nqubits, weight
    )

    hamming_full_state = get_full_initial_state(
        hamming_state, weight, nqubits, hamming_bkd
    )

    backend.assert_allclose(hamming_full_state, state, atol=1e-8)


@pytest.mark.parametrize("weight", [1, 2, 3, 4])
def test_n_qubit_gates(backend, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(4, weight))
    initial_state = random_statevector(dim, backend=backend, seed=1237)

    gate = gates.fSim(0, 1, 0.1, 0.3)
    id = gates.I(0).matrix(numpy_bkd)
    gate_matrix = gate.matrix(numpy_bkd)
    gate1_matrix = np.kron(id, gate_matrix)
    gate2_matrix = np.kron(gate_matrix, id)
    gate1 = gates.Unitary(backend.cast(gate1_matrix), 0, 1, 2)
    gate2 = gates.Unitary(backend.cast(gate2_matrix), 0, 1, 2)
    gate1.hamming_weight = True
    gate2.hamming_weight = True

    circuit = Circuit(4, density_matrix=False)
    circuit.add(gate1)
    circuit.add(gate2)
    hamming_result = hamming_bkd.execute_circuit(
        circuit, weight=weight, initial_state=initial_state
    )
    hamming_state = hamming_result.state()
    hamming_full_state = hamming_result.full_state()
    initial_state_full = get_full_initial_state(initial_state, weight, 4, hamming_bkd)

    result = backend.execute_circuit(circuit, initial_state=initial_state_full)
    state = result.state()
    state = backend.cast(state)

    backend.assert_allclose(hamming_full_state, state, atol=1e-8)

    hamming_bkd._dict_indexes = None
    state = hamming_bkd._apply_gate_n_qubit(gate1, initial_state, 4, weight)
    state = hamming_bkd._apply_gate_n_qubit(gate2, state, 4, weight)
    backend.assert_allclose(hamming_state, state, atol=1e-8)

    hamming_result._backend._dict_indexes = None
    assert result.symbolic() == hamming_result.symbolic()

    if len(gate1.control_qubits) == 0:
        circuit = Circuit(4, density_matrix=False)
        circuit.add(gate1.controlled_by(3))
        circuit.add(gate2.controlled_by(3))

        hamming_result = hamming_bkd.execute_circuit(
            circuit, weight=weight, initial_state=initial_state
        )
        hamming_state = hamming_result.state()
        hamming_full_state = hamming_result.full_state()
        initial_state_full = get_full_initial_state(
            initial_state, weight, 4, hamming_bkd
        )

        initial_state_full = numpy_bkd.cast(initial_state_full)
        result = backend.execute_circuit(circuit, initial_state=initial_state_full)
        state = result.state()
        state = backend.cast(state)

        backend.assert_allclose(hamming_full_state, state, atol=1e-8)
        assert result.symbolic() == hamming_result.symbolic()


@pytest.mark.parametrize("weight", [1, 2, 3])
@pytest.mark.parametrize("collapse", [False, True])
@pytest.mark.parametrize("nshots", [None, 100])
def test_measurement(backend, weight, collapse, nshots):
    backend.set_seed(2024)
    hamming_bkd = construct_hamming_weight_backend(backend)
    numpy_bkd.set_seed(2024)

    dim = int(binom(3, weight))
    initial_state = random_statevector(dim, backend=backend)
    initial_state_full = get_full_initial_state(initial_state, weight, 3, hamming_bkd)

    circuit = Circuit(3)
    circuit.add(gates.SWAP(0, 1))
    if collapse and nshots is not None:
        circuit.add(gates.M(1, collapse=collapse))
    circuit.add(gates.M(0, 2))

    result = backend.execute_circuit(
        circuit, initial_state=initial_state_full, nshots=nshots
    )
    probabilities = result.probabilities()

    hamming_result = hamming_bkd.execute_circuit(
        circuit, weight=weight, initial_state=initial_state, nshots=nshots
    )
    hamming_result._backend._dict_indexes = None
    hamming_probabilities = hamming_result.probabilities()

    if nshots is None:
        backend.assert_allclose(hamming_probabilities, probabilities, atol=1e-8)
    else:
        if collapse:
            hamming_samples = hamming_result.samples()
            samples = result.samples()
            backend.assert_allclose(hamming_samples, samples)

            nqubits = 4
            weight = 2
            dim = binom(nqubits, weight)
            initial_state = np.zeros(int(dim), complex)
            initial_state[0] = 1 / np.sqrt(2)
            initial_state[1] = 1 / np.sqrt(2)
            initial_state = backend.cast(initial_state)

            hamming_bkd._dict_indexes = None
            measure_gate = gates.M(0, collapse=True)
            measure_gate.result.backend = backend
            qubits = sorted(measure_gate.target_qubits)
            probs = hamming_bkd.calculate_probabilities(
                initial_state, qubits, weight, nqubits
            )
            shot = measure_gate.result.add_shot(probs, backend=hamming_bkd.engine)
            hamming_bkd._dict_indexes = None
            state = hamming_bkd.collapse_state(
                initial_state, qubits, shot, weight, nqubits
            )

            backend.assert_allclose(state, initial_state, atol=1e-8)

        else:
            hamming_freq = hamming_result.frequencies()
            frequencies = result.frequencies()

            probabilities = np.zeros(4)
            for key, value in frequencies.items():
                probabilities[int(key, 2)] = value / nshots

            hamming_freq_probs = np.zeros(4)
            for key, value in hamming_freq.items():
                hamming_freq_probs[int(key, 2)] = value / nshots

            backend.assert_allclose(hamming_probabilities, probabilities, atol=1e-1)
            backend.assert_allclose(
                hamming_freq_probs, hamming_probabilities, atol=1e-8
            )


@pytest.mark.parametrize("weight", [1, 2, 3])
def test_probabilities_from_samples(backend, weight):
    backend.set_seed(2024)
    hamming_bkd = construct_hamming_weight_backend(backend)

    circuit = Circuit(3)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.M(0, 2))
    result = hamming_bkd.execute_circuit(circuit, weight=weight, nshots=10)
    probs_1 = result.probabilities(qubits=[0])
    circuit = Circuit(3)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.M(0))
    result = hamming_bkd.execute_circuit(circuit, weight=weight, nshots=10)
    probs_2 = result.probabilities()

    backend.assert_allclose(probs_1, probs_2)


def test_errors(backend):
    hamming_bkd = construct_hamming_weight_backend(backend)

    circuit = Circuit(3)
    circuit.add(gates.Z(0))
    result = hamming_bkd.execute_circuit(circuit, weight=1)
    with pytest.raises(RuntimeError):
        result.samples()
    with pytest.raises(RuntimeError):
        result.frequencies()

    circuit.add(gates.M(1))
    result = hamming_bkd.execute_circuit(circuit, weight=1, nshots=10)
    with pytest.raises(RuntimeError):
        result.probabilities(qubits=[0, 1])

    circuit = Circuit(3)
    circuit.add(gates.Z(0))
    circuit.density_matrix = True
    with pytest.raises(RuntimeError):
        hamming_bkd.execute_circuit(circuit, weight=1)

    circuit.density_matrix = False
    circuit.add(gates.DepolarizingChannel(0, 0.1))
    with pytest.raises(RuntimeError):
        hamming_bkd.execute_circuit(circuit, weight=1)

    circuit = Circuit(3)
    circuit.add(gates.X(0))
    with pytest.raises(RuntimeError):
        hamming_bkd.execute_circuit(circuit, weight=1)
