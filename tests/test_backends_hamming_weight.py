"""Tests for HammingWeight backend."""

from itertools import product

import numpy as np
import pytest
from scipy.special import binom

from qibo import Circuit, gates, get_backend, set_backend
from qibo.backends import HammingWeightBackend, NumpyBackend, _get_engine_name

numpy_bkd = NumpyBackend()


def construct_hamming_weight_backend(backend):
    if backend.__class__.__name__ in (
        "TensorflowBackend",
        "PyTorchBackend",
        "CuQuantumBackend",
    ):
        with pytest.raises(NotImplementedError):
            hamming_backend = HammingWeightBackend(backend.name)
        pytest.skip("HammingWeight backend not defined for this engine.")

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


SINGLE_QUBIT_GATES = ["Z", "S", "SDG", "T", "TDG", "I"]


@pytest.mark.parametrize("gate", SINGLE_QUBIT_GATES)
@pytest.mark.parametrize("weight", [1, 2, 3])
def test_single_qubit_gates(backend, gate, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(3, weight))
    initial_state = np.random.rand(dim)
    initial_state /= np.linalg.norm(initial_state)
    initial_state = backend.cast(initial_state)
    initial_state_copy = initial_state.copy()

    qubits = [0, 2]
    gate1 = getattr(gates, gate)(qubits[0])
    gate2 = getattr(gates, gate)(qubits[1])

    c = Circuit(3, density_matrix=False)
    c.add(gate1)
    c.add(gate2)

    hamming_result = hamming_bkd.execute_circuit(
        c, weight=weight, initial_state=initial_state
    )
    hamming_state = hamming_result.state()
    hamming_full_state = hamming_result.full_state()
    initial_state_full = get_full_initial_state(
        initial_state_copy, weight, 3, hamming_bkd
    )
    initial_state_full = numpy_bkd.cast(initial_state_full)
    numpy_result = numpy_bkd.execute_circuit(c, initial_state=initial_state_full)
    numpy_state = numpy_result.state()
    numpy_state = backend.cast(numpy_state)

    backend.assert_allclose(hamming_full_state, numpy_state, atol=1e-8)
    assert numpy_result.symbolic() == hamming_result.symbolic()

    c = Circuit(3, density_matrix=False)
    c.add(gate1.controlled_by(1))
    c.add(gate2.controlled_by(0))

    hamming_result = hamming_bkd.execute_circuit(
        c, weight=weight, initial_state=initial_state
    )
    hamming_result.backend._dict_indexes = None
    hamming_state = hamming_result.state()
    hamming_full_state = hamming_result.full_state()
    initial_state_full = get_full_initial_state(
        initial_state_copy, weight, 3, hamming_bkd
    )
    initial_state_full = numpy_bkd.cast(initial_state_full)
    numpy_result = numpy_bkd.execute_circuit(c, initial_state=initial_state_full)
    numpy_state = numpy_result.state()
    numpy_state = backend.cast(numpy_state)

    backend.assert_allclose(hamming_full_state, numpy_state, atol=1e-8)
    assert numpy_result.symbolic() == hamming_result.symbolic()


TWO_QUBIT_GATES = ["CZ", "SWAP", "iSWAP", "SiSWAP", "SiSWAPDG", "FSWAP", "SYC"]


@pytest.mark.parametrize("gate", TWO_QUBIT_GATES)
@pytest.mark.parametrize("weight", [1, 2, 3])
def test_two_qubit_gates(backend, gate, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(3, weight))
    initial_state = np.random.rand(dim)
    initial_state /= np.linalg.norm(initial_state)
    initial_state = backend.cast(initial_state)
    initial_state_copy = initial_state.copy()

    qubits1 = [0, 1]
    qubits2 = [2, 1]
    gate1 = getattr(gates, gate)(*qubits1)
    gate2 = getattr(gates, gate)(*qubits2)

    c = Circuit(3, density_matrix=False)
    c.add(gate1)
    c.add(gate2)
    hamming_result = hamming_bkd.execute_circuit(
        c, weight=weight, initial_state=initial_state
    )
    hamming_state = hamming_result.state()
    hamming_full_state = hamming_result.full_state()
    initial_state_full = get_full_initial_state(
        initial_state_copy, weight, 3, hamming_bkd
    )
    initial_state_full = numpy_bkd.cast(initial_state_full)
    numpy_result = numpy_bkd.execute_circuit(c, initial_state=initial_state_full)
    numpy_state = numpy_result.state()
    numpy_state = backend.cast(numpy_state)

    backend.assert_allclose(hamming_full_state, numpy_state, atol=1e-8)
    assert numpy_result.symbolic() == hamming_result.symbolic()

    if len(gate1.control_qubits) == 0:
        c = Circuit(3, density_matrix=False)
        c.add(gate1.controlled_by(2))
        c.add(gate2.controlled_by(0))

        hamming_result = hamming_bkd.execute_circuit(
            c, weight=weight, initial_state=initial_state
        )
        hamming_state = hamming_result.state()
        hamming_full_state = hamming_result.full_state()
        initial_state_full = get_full_initial_state(
            initial_state_copy, weight, 3, hamming_bkd
        )
        initial_state_full = numpy_bkd.cast(initial_state_full)
        numpy_result = numpy_bkd.execute_circuit(c, initial_state=initial_state_full)
        numpy_state = numpy_result.state()
        numpy_state = backend.cast(numpy_state)

        backend.assert_allclose(hamming_full_state, numpy_state, atol=1e-8)
        assert numpy_result.symbolic() == hamming_result.symbolic()


@pytest.mark.parametrize("weight", [1, 2, 3])
def test_ccz(backend, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(3, weight))
    initial_state = np.random.rand(dim)
    initial_state /= np.linalg.norm(initial_state)
    initial_state = backend.cast(initial_state)
    initial_state_copy = initial_state.copy()

    qubits1 = [0, 1, 2]
    qubits2 = [2, 1, 0]
    gate1 = gates.CCZ(*qubits1)
    gate2 = gates.CCZ(*qubits2)

    c = Circuit(3, density_matrix=False)
    c.add(gate1)
    c.add(gate2)
    hamming_result = hamming_bkd.execute_circuit(
        c, weight=weight, initial_state=initial_state
    )
    hamming_state = hamming_result.state()
    hamming_full_state = hamming_result.full_state()
    initial_state_full = get_full_initial_state(
        initial_state_copy, weight, 3, hamming_bkd
    )
    initial_state_full = numpy_bkd.cast(initial_state_full)
    numpy_result = numpy_bkd.execute_circuit(c, initial_state=initial_state_full)
    numpy_state = numpy_result.state()
    numpy_state = backend.cast(numpy_state)

    backend.assert_allclose(hamming_full_state, numpy_state, atol=1e-8)


@pytest.mark.parametrize("weight", [1, 2, 3, 4])
def test_n_qubit_gates(backend, weight):
    hamming_bkd = construct_hamming_weight_backend(backend)
    dim = int(binom(4, weight))
    initial_state = np.random.rand(dim)
    initial_state /= np.linalg.norm(initial_state)
    initial_state = backend.cast(initial_state)
    initial_state_copy = initial_state.copy()

    gate = gates.fSim(0, 1, 0.1, 0.3)
    id = gates.I(0).matrix(numpy_bkd)
    gate_matrix = gate.matrix(numpy_bkd)
    gate1_matrix = np.kron(id, gate_matrix)
    gate2_matrix = np.kron(gate_matrix, id)
    gate1 = gates.Unitary(backend.cast(gate1_matrix), 0, 1, 2)
    gate2 = gates.Unitary(backend.cast(gate2_matrix), 0, 1, 2)
    gate1_numpy = gates.Unitary(gate1_matrix, 0, 1, 2)
    gate2_numpy = gates.Unitary(gate2_matrix, 0, 1, 2)
    gate1.hamming_weight = True
    gate2.hamming_weight = True

    c = Circuit(4, density_matrix=False)
    c.add(gate1)
    c.add(gate2)
    hamming_result = hamming_bkd.execute_circuit(
        c, weight=weight, initial_state=initial_state
    )
    hamming_state = hamming_result.state()
    hamming_full_state = hamming_result.full_state()
    initial_state_full = get_full_initial_state(
        initial_state_copy, weight, 4, hamming_bkd
    )

    c_numpy = Circuit(4, density_matrix=False)
    c_numpy.add(gate1_numpy)
    c_numpy.add(gate2_numpy)

    initial_state_full = numpy_bkd.cast(initial_state_full)
    numpy_result = numpy_bkd.execute_circuit(c_numpy, initial_state=initial_state_full)
    numpy_state = numpy_result.state()
    numpy_state = backend.cast(numpy_state)

    backend.assert_allclose(hamming_full_state, numpy_state, atol=1e-8)
    assert numpy_result.symbolic() == hamming_result.symbolic()

    if len(gate1.control_qubits) == 0:
        c = Circuit(4, density_matrix=False)
        c.add(gate1.controlled_by(3))
        c.add(gate2.controlled_by(3))

        hamming_result = hamming_bkd.execute_circuit(
            c, weight=weight, initial_state=initial_state
        )
        hamming_state = hamming_result.state()
        hamming_full_state = hamming_result.full_state()
        initial_state_full = get_full_initial_state(
            initial_state_copy, weight, 4, hamming_bkd
        )

        c_numpy = Circuit(4, density_matrix=False)
        c_numpy.add(gate1_numpy.controlled_by(3))
        c_numpy.add(gate2_numpy.controlled_by(3))

        initial_state_full = numpy_bkd.cast(initial_state_full)
        numpy_result = numpy_bkd.execute_circuit(
            c_numpy, initial_state=initial_state_full
        )
        numpy_state = numpy_result.state()
        numpy_state = backend.cast(numpy_state)

        backend.assert_allclose(hamming_full_state, numpy_state, atol=1e-8)
        assert numpy_result.symbolic() == hamming_result.symbolic()


@pytest.mark.parametrize("weight", [1, 2, 3])
@pytest.mark.parametrize("collapse", [False, True])
@pytest.mark.parametrize("nshots", [None, 100])
def test_measurement(backend, weight, collapse, nshots):
    backend.set_seed(2024)
    hamming_bkd = construct_hamming_weight_backend(backend)
    numpy_bkd.set_seed(2024)

    dim = int(binom(3, weight))
    initial_state = np.random.rand(dim)
    initial_state /= np.linalg.norm(initial_state)
    initial_state = backend.cast(initial_state)
    initial_state_copy = initial_state.copy()

    c = Circuit(3)
    c.add(gates.SWAP(0, 1))
    if collapse and nshots is not None:
        c.add(gates.M(1, collapse=collapse))
    c.add(gates.M(0, 2))

    hamming_result = hamming_bkd.execute_circuit(
        c, weight=weight, initial_state=initial_state, nshots=nshots
    )
    hamming_probabilities = hamming_result.probabilities()

    initial_state_full = get_full_initial_state(
        initial_state_copy, weight, 3, hamming_bkd
    )
    initial_state_full = numpy_bkd.cast(initial_state_full)
    numpy_result = numpy_bkd.execute_circuit(
        c, initial_state=initial_state_full, nshots=nshots
    )
    numpy_probabilities = numpy_result.probabilities()

    if nshots is None:
        backend.assert_allclose(hamming_probabilities, numpy_probabilities, atol=1e-8)
    else:
        if collapse:
            hamming_samples = hamming_result.samples()
            numpy_samples = numpy_result.samples()
            backend.assert_allclose(hamming_samples, numpy_samples)
        else:
            hamming_freq = hamming_result.frequencies()
            numpy_freq = numpy_result.frequencies()

            numpy_probabilities = np.zeros(4)
            for key, value in numpy_freq.items():
                numpy_probabilities[int(key, 2)] = value / nshots

            hamming_freq_probs = np.zeros(4)
            for key, value in hamming_freq.items():
                hamming_freq_probs[int(key, 2)] = value / nshots

            backend.assert_allclose(
                hamming_probabilities, numpy_probabilities, atol=1e-1
            )
            backend.assert_allclose(
                hamming_freq_probs, hamming_probabilities, atol=1e-8
            )


@pytest.mark.parametrize("weight", [1, 2, 3])
def test_probabilities_from_samples(backend, weight):
    backend.set_seed(2024)
    hamming_bkd = construct_hamming_weight_backend(backend)

    c = Circuit(3)
    c.add(gates.SWAP(0, 1))
    c.add(gates.M(0, 2))
    result = hamming_bkd.execute_circuit(c, weight=weight, nshots=10)
    probs_1 = result.probabilities(qubits=[0])
    c = Circuit(3)
    c.add(gates.SWAP(0, 1))
    c.add(gates.M(0))
    result = hamming_bkd.execute_circuit(c, weight=weight, nshots=10)
    probs_2 = result.probabilities()

    backend.assert_allclose(probs_1, probs_2)


def test_errors(backend):
    hamming_bkd = construct_hamming_weight_backend(backend)

    c = Circuit(3)
    c.add(gates.Z(0))
    result = hamming_bkd.execute_circuit(c, weight=1)
    with pytest.raises(RuntimeError):
        result.samples()
    with pytest.raises(RuntimeError):
        result.frequencies()

    c.add(gates.M(1))
    result = hamming_bkd.execute_circuit(c, weight=1, nshots=10)
    with pytest.raises(RuntimeError):
        result.probabilities(qubits=[0, 1])

    c = Circuit(3)
    c.add(gates.Z(0))
    c.density_matrix = True
    with pytest.raises(RuntimeError):
        hamming_bkd.execute_circuit(c, weight=1)

    c.density_matrix = False
    c.add(gates.DepolarizingChannel(0, 0.1))
    with pytest.raises(RuntimeError):
        hamming_bkd.execute_circuit(c, weight=1)

    c = Circuit(3)
    c.add(gates.X(0))
    with pytest.raises(RuntimeError):
        hamming_bkd.execute_circuit(c, weight=1)
