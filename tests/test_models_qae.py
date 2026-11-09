"""Tests for the Quantum Amplitude Estimation circuit defined in `qibo/models/qae.py`."""

import math

import pytest

from qibo import Circuit, gates
from qibo.models import QAE
from qibo.quantum_info import estimate_phase


def _one_qubit_problem(theta):
    # state cos(theta) |0> + sin(theta) |1>, with the good state |1>
    algorithm = Circuit(1)
    algorithm.add(gates.RY(0, theta=2 * theta))
    oracle = Circuit(1)
    oracle.add(gates.Z(0))
    return algorithm, oracle


def _grover_problem():
    # uniform superposition of 2 qubits, with the good state |11>, so that a = 1 / 4
    algorithm = Circuit(2)
    algorithm.add(gates.H(qubit) for qubit in range(2))
    oracle = Circuit(2)
    oracle.add(gates.CZ(0, 1))
    return algorithm, oracle


def _estimate(backend, circuit, nqubits_counting):
    qubits = list(range(nqubits_counting))
    probabilities = backend.execute_circuit(circuit).probabilities(qubits)
    return math.sin(math.pi * estimate_phase(probabilities)) ** 2


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("numerator", [0, 1, 3])
@pytest.mark.parametrize("nqubits_counting", [3, 4])
def test_qae_exact(backend, nqubits_counting, numerator, density_matrix):
    theta = math.pi * numerator / 2**nqubits_counting
    algorithm, oracle = _one_qubit_problem(theta)

    circuit = QAE(
        algorithm,
        oracle,
        nqubits_counting,
        measure=False,
        density_matrix=density_matrix,
    )
    assert circuit.nqubits == nqubits_counting + 1
    assert circuit.density_matrix == density_matrix

    estimate = _estimate(backend, circuit, nqubits_counting)
    backend.assert_allclose(estimate, math.sin(theta) ** 2, atol=1e-8)


@pytest.mark.parametrize("nqubits_counting", [4, 6])
@pytest.mark.parametrize("problem", ["one_qubit", "grover"])
def test_qae_error_bound(backend, nqubits_counting, problem):
    if problem == "one_qubit":
        theta = 0.4
        algorithm, oracle = _one_qubit_problem(theta)
    else:
        theta = math.pi / 6
        algorithm, oracle = _grover_problem()

    amplitude = math.sin(theta) ** 2
    dimension = 2**nqubits_counting

    circuit = QAE(algorithm, oracle, nqubits_counting, measure=False)
    assert circuit.nqubits == nqubits_counting + algorithm.nqubits

    estimate = _estimate(backend, circuit, nqubits_counting)
    bound = (
        2 * math.pi * math.sqrt(amplitude * (1 - amplitude)) / dimension
        + math.pi**2 / dimension**2
    )
    assert abs(estimate - amplitude) <= bound


@pytest.mark.parametrize("measure", [False, True])
def test_qae_measure(backend, measure):
    algorithm, oracle = _one_qubit_problem(math.pi / 8)
    circuit = QAE(algorithm, oracle, 3, measure=measure)

    assert len(circuit.measurements) == int(measure)
    if measure:
        frequencies = backend.execute_circuit(circuit, nshots=50).frequencies()
        assert sum(frequencies.values()) == 50
        assert set(frequencies) <= {"001", "111"}


def test_qae_errors():
    algorithm, oracle = _one_qubit_problem(0.3)

    with pytest.raises(TypeError):
        QAE(gates.H(0), oracle, 3)

    with pytest.raises(ValueError):
        QAE(algorithm, Circuit(2), 3)

    with pytest.raises(ValueError):
        QAE(algorithm, oracle, 0)
