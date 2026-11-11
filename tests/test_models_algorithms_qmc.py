"""Tests for the Quantum Monte Carlo algorithms in `qibo/models/algorithms/qmc.py`."""

import math

import pytest

from qibo import Circuit, gates
from qibo.models import QMC, binary_encoder
from qibo.models.algorithms.qmc import _repetitions

PROBABILITIES = [0.30, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.02]
VALUES = [0.0, 0.3, 0.8, 1.5, 2.5, 4.0, 6.0, 9.0]


def _algorithm(backend, probabilities=PROBABILITIES):
    amplitudes = backend.sqrt(backend.cast(probabilities, dtype=backend.float64))

    return binary_encoder(
        int(math.log2(len(probabilities))), data=amplitudes, backend=backend
    )


def _moments(probabilities, values):
    mean = sum(p * v for p, v in zip(probabilities, values))
    second = sum(p * v**2 for p, v in zip(probabilities, values))

    return mean, second


def test_qmc_bounded(backend):
    # Algorithm 1: the error of the amplitude estimation, with 5 counting qubits
    values = [0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0]
    mean, _ = _moments(PROBABILITIES, values)
    ncounting = 5

    qmc = QMC(_algorithm(backend), values)
    assert qmc.circuit is None

    backend.set_seed(0)
    estimates = qmc._bounded(values, 2**ncounting - 1, 1e-3, 4, backend)

    bound = 2 * math.pi * math.sqrt(mean * (1 - mean)) / 2**ncounting
    bound += math.pi**2 / 4**ncounting
    assert len(estimates) == 4
    assert all(abs(estimate - mean) <= bound for estimate in estimates)

    # the circuit has the counting qubits, the qubits of the algorithm, and the flag
    assert qmc.circuit.nqubits == ncounting + 3 + 1
    assert qmc.circuit.measurements[0].qubits == tuple(range(ncounting))


def test_qmc_positive(backend):
    # Algorithm 2, with an error that is bounded by epsilon (norm + 1)^2
    mean, second = _moments(PROBABILITIES, VALUES)
    epsilon = 0.1

    backend.set_seed(1)
    estimate = QMC(_algorithm(backend), VALUES).execute(
        epsilon, delta=0.05, backend=backend
    )

    assert abs(estimate - mean) <= epsilon * (math.sqrt(second) + 1) ** 2


def test_qmc_variance(backend):
    # Algorithm 3: the error is epsilon, which is smaller than 4 sigma
    mean, second = _moments(PROBABILITIES, VALUES)
    sigma = math.sqrt(second - mean**2)

    backend.set_seed(2)
    estimate = QMC(_algorithm(backend), VALUES).execute(
        2 * sigma, sigma=sigma, backend=backend
    )

    assert abs(estimate - mean) <= 2 * sigma

    # the values can be negative, and the median improves the probability of success
    values = [value - 3 for value in VALUES]
    backend.set_seed(3)
    estimate = QMC(_algorithm(backend), values).execute(
        2 * sigma, sigma=sigma, delta=0.4, backend=backend
    )

    assert abs(estimate - (mean - 3)) <= 2 * sigma


def test_qmc_relative(backend):
    # Algorithm 4: the relative error is epsilon
    mean, second = _moments(PROBABILITIES, VALUES)
    relative_variance = (second - mean**2) / mean**2
    epsilon = 0.5

    backend.set_seed(4)
    estimate = QMC(_algorithm(backend), VALUES).execute(
        epsilon, relative_variance=relative_variance, backend=backend
    )

    assert abs(estimate - mean) <= epsilon * mean


def test_qmc_zero(backend):
    # the amplitude is zero, so that no circuit is needed to estimate it
    qmc = QMC(_algorithm(backend), [0.0] * 8)

    assert qmc.execute(0.1, backend=backend) == 0.0
    assert qmc.circuit is None
    assert qmc.execute(0.5, relative_variance=1.0, backend=backend) == 0.0
    assert qmc.execute(0.5, sigma=1.0, backend=backend) == 0.0


def test_qmc_qubits(backend):
    # only two of the qubits are measured, and the first one is the least significant
    algorithm = Circuit(3)
    algorithm.add(gates.RY(0, theta=0.7))
    algorithm.add(gates.RY(1, theta=1.9))
    algorithm.add(gates.CNOT(1, 2))
    values = [0.0, 0.5, 1.5, 2.0]

    qubits = [2, 0]
    probabilities = backend.execute_circuit(algorithm).probabilities(qubits)
    mean, second = _moments([float(p) for p in probabilities], values)
    epsilon = 0.1

    backend.set_seed(5)
    estimate = QMC(algorithm, values, qubits=qubits).execute(
        epsilon, delta=0.05, backend=backend
    )

    assert abs(estimate - mean) <= epsilon * (math.sqrt(second) + 1) ** 2


def test_qmc_repetitions():
    assert _repetitions(0.8, None) == 1
    assert _repetitions(0.8, 0.2) == 1
    assert _repetitions(2 / 3, 0.34) == 1

    # odd numbers, which grow with the confidence
    repetitions = [_repetitions(0.8, delta) for delta in (0.1, 0.01, 0.001)]
    assert all(number % 2 == 1 for number in repetitions)
    assert repetitions == sorted(repetitions)
    assert repetitions[1] >= math.log(100) / (2 * 0.3**2)


def test_qmc_errors(backend):
    algorithm = _algorithm(backend)

    with pytest.raises(TypeError):
        QMC(gates.H(0), VALUES)

    for qubits in ([], [0, 0, 1], [0, 1, 3], [-1, 0, 1], [0.5, 1, 2], [True, 1, 2]):
        with pytest.raises(ValueError):
            QMC(algorithm, VALUES, qubits=qubits)

    for values in (VALUES[:7], VALUES + [1.0], VALUES[:7] + [math.nan], [math.inf] * 8):
        with pytest.raises(ValueError):
            QMC(algorithm, values)

    qmc = QMC(algorithm, VALUES)

    for epsilon in (0, -0.1, True, "a", 0.5):
        with pytest.raises(ValueError):
            qmc.execute(epsilon, backend=backend)

    for name in ("sigma", "relative_variance", "constant"):
        for number in (0, -1, True, "a"):
            with pytest.raises(ValueError):
                qmc.execute(0.1, backend=backend, **{name: number})

    for delta in (0, 1, 1.5, True, "a"):
        with pytest.raises(ValueError):
            qmc.execute(0.1, delta=delta, backend=backend)

    # sigma and the relative variance are exclusive, the relative variance is at least 1, and
    # the accuracy is smaller than the limit of each algorithm
    for arguments in (
        {"sigma": 1.0, "relative_variance": 2.0},
        {"relative_variance": 0.5},
        {"sigma": 1.0, "epsilon": 4.0},
        {"relative_variance": 1.0, "epsilon": 7.0},
    ):
        with pytest.raises(ValueError):
            qmc.execute(**{"epsilon": 0.1, **arguments}, backend=backend)

    # the algorithms for non-negative values
    negative = QMC(algorithm, [-1.0] + VALUES[1:])
    with pytest.raises(ValueError):
        negative.execute(0.1, backend=backend)
    with pytest.raises(ValueError):
        negative.execute(0.5, relative_variance=1.0, backend=backend)
