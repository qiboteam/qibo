"""Tests for the Maximum Likelihood Quantum Amplitude Estimation in `qibo/models/mlqae.py`."""

import math

import pytest

from qibo import Circuit, gates
from qibo.models import MLQAE


def _one_qubit_problem(theta):
    # state cos(theta) |0> + sin(theta) |1>, with the good state |1>
    algorithm = Circuit(1)
    algorithm.add(gates.RY(0, theta=2 * theta))
    oracle = Circuit(1)
    oracle.add(gates.Z(0))
    return algorithm, oracle, lambda bitstring: bitstring == "1"


def _grover_problem():
    # uniform superposition of 3 qubits, with the good state |111>, so that a = 1 / 8
    algorithm = Circuit(3)
    algorithm.add(gates.H(qubit) for qubit in range(3))
    oracle = Circuit(3)
    oracle.add(gates.Z(2).controlled_by(0, 1))
    return algorithm, oracle, lambda bitstring: bitstring == "111"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("theta", [0.1, 0.5, 1.2])
def test_mlqae_one_qubit(backend, theta, seed):
    algorithm, oracle, check = _one_qubit_problem(theta)

    backend.set_seed(seed)
    estimate = MLQAE(algorithm, oracle, check, backend=backend)

    assert abs(estimate - math.sin(theta) ** 2) < 0.02


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mlqae_grover(backend, seed):
    algorithm, oracle, check = _grover_problem()

    backend.set_seed(seed)
    estimate = MLQAE(algorithm, oracle, check, nlevels=4, nshots=200, backend=backend)

    assert abs(estimate - 1 / 8) < 0.02


@pytest.mark.parametrize("nlevels", [1, 2])
def test_mlqae_few_levels(backend, nlevels):
    algorithm, oracle, check = _one_qubit_problem(0.7)

    backend.set_seed(nlevels)
    estimate = MLQAE(algorithm, oracle, check, nlevels, nshots=1000, backend=backend)

    assert abs(estimate - math.sin(0.7) ** 2) < 0.05


def test_mlqae_no_or_all_good(backend):
    algorithm, oracle, _ = _one_qubit_problem(0.5)

    # no good outcome, and all of them are good
    estimate = MLQAE(algorithm, oracle, lambda bitstring: False, backend=backend)
    backend.assert_allclose(estimate, 0.0, atol=1e-8)

    estimate = MLQAE(algorithm, oracle, lambda bitstring: True, backend=backend)
    backend.assert_allclose(estimate, 1.0, atol=1e-8)


def test_mlqae_errors(backend):
    algorithm, oracle, check = _one_qubit_problem(0.5)

    with pytest.raises(TypeError):
        MLQAE(algorithm, oracle, "1", backend=backend)

    with pytest.raises(TypeError):
        MLQAE(gates.H(0), oracle, check, backend=backend)

    for nlevels in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            MLQAE(algorithm, oracle, check, nlevels=nlevels, backend=backend)

    for nshots in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            MLQAE(algorithm, oracle, check, nshots=nshots, backend=backend)
