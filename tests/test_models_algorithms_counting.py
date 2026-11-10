"""Tests for the Quantum Counting circuit defined in `qibo/models/algorithms/counting.py`."""

import math

import pytest

from qibo import Circuit, gates
from qibo.models import Counting
from qibo.quantum_info import estimate_phase


def _oracle(nqubits, nmarked_qubits):
    # marks the states in which the first nmarked_qubits qubits are in the state 1, so that
    # the number of marked states is 2^(nqubits - nmarked_qubits)
    oracle = Circuit(nqubits)
    oracle.add(gates.Z(nmarked_qubits - 1).controlled_by(*range(nmarked_qubits - 1)))
    return oracle


def _estimate(backend, circuit, nqubits_counting, nstates):
    qubits = list(range(nqubits_counting))
    probabilities = backend.execute_circuit(circuit).probabilities(qubits)
    return nstates * math.sin(math.pi * estimate_phase(probabilities)) ** 2


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_counting_exact(backend, nqubits):
    # half of the states are marked, so that the phase to be estimated is exactly 1 / 4
    circuit = Counting(_oracle(nqubits, 1), 3, measure=False)
    assert circuit.nqubits == 3 + nqubits

    estimate = _estimate(backend, circuit, 3, 2**nqubits)
    backend.assert_allclose(estimate, 2 ** (nqubits - 1), atol=1e-8)


@pytest.mark.parametrize("nmarked_qubits", [2, 3])
@pytest.mark.parametrize("nqubits", [3, 4])
def test_counting_error_bound(backend, nqubits, nmarked_qubits):
    nqubits_counting = 6
    nstates = 2**nqubits
    nmarked = 2 ** (nqubits - nmarked_qubits)
    dimension = 2**nqubits_counting

    circuit = Counting(
        _oracle(nqubits, nmarked_qubits), nqubits_counting, measure=False
    )

    estimate = _estimate(backend, circuit, nqubits_counting, nstates)
    bound = (
        2 * math.pi * math.sqrt(nmarked * (nstates - nmarked)) / dimension
        + math.pi**2 * nstates / dimension**2
    )
    assert abs(estimate - nmarked) <= bound


@pytest.mark.parametrize("nmarked", [0, 8])
def test_counting_no_or_all_marked(backend, nmarked):
    # empty oracle marks no state, and an oracle with a phase of -1 marks all of them
    oracle = Circuit(3)
    if nmarked > 0:
        oracle.add(gates.GlobalPhase(math.pi))

    circuit = Counting(oracle, 3, measure=False)

    estimate = _estimate(backend, circuit, 3, 8)
    backend.assert_allclose(estimate, nmarked, atol=1e-8)


@pytest.mark.parametrize("measure", [False, True])
def test_counting_measure(backend, measure):
    circuit = Counting(_oracle(2, 1), 3, measure=measure, density_matrix=True)
    assert circuit.density_matrix
    assert len(circuit.measurements) == int(measure)

    if measure:
        frequencies = backend.execute_circuit(circuit, nshots=50).frequencies()
        assert sum(frequencies.values()) == 50
        assert set(frequencies) <= {"010", "110"}


def test_counting_errors():
    with pytest.raises(TypeError):
        Counting(gates.Z(0), 3)

    oracle = Circuit(2)
    oracle.add(gates.M(0))
    with pytest.raises(ValueError):
        Counting(oracle, 3)

    with pytest.raises(ValueError):
        Counting(_oracle(2, 1), 0)
