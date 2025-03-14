"""Tests for HammingWeight backend."""

from itertools import product

import numpy as np
import pytest
from scipy.special import binom

from qibo import Circuit, gates, get_backend, set_backend
from qibo.backends import HammingWeightBackend, NumpyBackend, _get_engine_name
from qibo.noise import DepolarizingError

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

    # full_state = backend.engine.cast(full_state)
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


# @pytest.mark.parametrize("weight", [1, 2, 3])
# @pytest.mark.parametrize("collapse", [False, True])
# def test_measurement(backend, weight, collapse):
