"""Tests for the Quantum Amplitude Amplification circuits defined in `qibo/models/qaa.py`."""

import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.models import QAA, QSearch, amplification_operator
from qibo.quantum_info import random_unitary


def test_amplification_circuits_errors():
    algorithm, oracle = _grover(2)

    with pytest.raises(TypeError):
        amplification_operator(algorithm.queue, oracle)
    with pytest.raises(TypeError):
        QAA(algorithm, gates.CZ(0, 1), niterations=1)

    with pytest.raises(ValueError):
        amplification_operator(algorithm, _grover(3)[1])
    with pytest.raises(ValueError):
        QAA(_grover(3)[0], oracle, niterations=1)

    measured = Circuit(2)
    measured.add(gates.M(0))
    with pytest.raises(ValueError):
        amplification_operator(measured, oracle)
    with pytest.raises(ValueError):
        QAA(algorithm, measured, niterations=1)


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_amplification_operator(backend, nqubits):
    """The circuit is the operator -A S_0 A^{-1} S_chi, with its global phase."""
    dim = 2**nqubits
    algorithm, oracle, good = _problem(backend, nqubits, seed=nqubits)

    s_zero = np.eye(dim)
    s_zero[0, 0] = -1
    s_chi = np.eye(dim)
    s_chi[good, good] = -1

    matrix = algorithm.unitary(backend)
    target = (
        -matrix @ backend.cast(s_zero) @ backend.dagger(matrix) @ backend.cast(s_chi)
    )

    operator = amplification_operator(algorithm, oracle)
    assert operator.nqubits == nqubits
    backend.assert_allclose(operator.unitary(backend), target, atol=1e-8, rtol=0.0)


def test_amplification_operator_grover(backend):
    """With Hadamard gates, one application on 2 qubits finds the marked state."""
    algorithm, oracle = _grover(2)
    circuit = Circuit(2)
    circuit.add(algorithm)
    circuit.add(amplification_operator(algorithm, oracle))

    assert _good_probability(backend, circuit, [3]) == pytest.approx(1.0)


def test_qaa_density_matrix(backend):
    algorithm, oracle = _grover(2)
    circuit = QAA(
        algorithm, oracle, niterations=1, density_matrix=True, backend=backend
    )

    assert circuit.density_matrix
    frequencies = backend.execute_circuit(circuit, nshots=10).frequencies()
    assert frequencies == {"11": 10}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"niterations": 1, "probability": 0.5},
        {"niterations": 1, "exact": True},
        {"niterations": -1},
        {"niterations": 1.5},
        {"niterations": True},
        {"probability": 0.0},
        {"probability": 1.5},
        {"probability": "0.5"},
    ],
)
def test_qaa_errors_arguments(kwargs):
    algorithm, oracle = _grover(2)
    with pytest.raises(ValueError):
        QAA(algorithm, oracle, **kwargs)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_qaa_exact(backend, nqubits, seed):
    """A good state is found with certainty when the probability is known."""
    algorithm, oracle, good = _problem(backend, nqubits, seed=seed + 100 * nqubits)

    initial = _good_probability(backend, algorithm, good)
    circuit = QAA(
        algorithm,
        oracle,
        probability=initial,
        exact=True,
        measure=False,
        backend=backend,
    )

    # the last qubit is the auxiliary qubit, and a good state has it in the state 1
    assert circuit.nqubits == nqubits + 1
    assert _good_probability(
        backend, circuit, [2 * index + 1 for index in good]
    ) == pytest.approx(1.0, abs=1e-8)


def test_qaa_exact_integer_iterations(backend):
    """The auxiliary qubit does not spoil the cases that are exact without it."""
    # a = 1 / 4 needs exactly one iteration
    algorithm, oracle = _grover(2)
    circuit = QAA(algorithm, oracle, probability=1 / 4, exact=True, backend=backend)
    assert circuit.nqubits == 3
    frequencies = backend.execute_circuit(circuit, nshots=20).frequencies()
    assert frequencies == {"11": 20}

    # a = 1 needs no iterations
    algorithm, oracle = Circuit(1), Circuit(1)
    algorithm.add(gates.X(0))
    oracle.add(gates.Z(0))
    circuit = QAA(algorithm, oracle, probability=1.0, exact=True, backend=backend)
    assert circuit.nqubits == 2
    frequencies = backend.execute_circuit(circuit, nshots=20).frequencies()
    assert frequencies == {"1": 20}


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_qaa_grover(backend, nqubits):
    """Grover's search is recovered for a single marked state."""
    algorithm, oracle = _grover(nqubits)
    circuit = QAA(algorithm, oracle, probability=1 / 2**nqubits, backend=backend)

    backend.set_seed(0)
    frequencies = backend.execute_circuit(circuit, nshots=200).frequencies()

    assert frequencies.most_common(1)[0][0] == "1" * nqubits
    assert frequencies["1" * nqubits] > 150


def test_qaa_larger_circuit(backend):
    """The circuit can be added to a subset of the qubits of a larger one."""
    algorithm, oracle = _grover(2)
    circuit = Circuit(3)
    circuit.add(gates.X(2))
    circuit.add(
        QAA(algorithm, oracle, niterations=1, measure=False, backend=backend).on_qubits(
            0, 1
        )
    )

    state = backend.execute_circuit(circuit).state()
    backend.assert_allclose(backend.abs(state[7]) ** 2, 1.0, atol=1e-8)


def test_qaa_measure(backend):
    algorithm, oracle = _grover(2)

    circuit = QAA(algorithm, oracle, niterations=1, measure=False, backend=backend)
    assert len(circuit.measurements) == 0

    circuit = QAA(algorithm, oracle, niterations=1, backend=backend)
    assert len(circuit.measurements) == 1
    assert circuit.measurements[0].target_qubits == (0, 1)

    # the auxiliary qubit is not measured
    circuit = QAA(algorithm, oracle, probability=1 / 4, exact=True, backend=backend)
    assert circuit.nqubits == 3
    assert circuit.measurements[0].target_qubits == (0, 1)


@pytest.mark.parametrize("niterations", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_qaa_niterations(backend, nqubits, niterations):
    """After m iterations the success probability is sin^2((2m + 1) theta)."""
    algorithm, oracle, good = _problem(backend, nqubits, seed=10 * nqubits)

    initial = _good_probability(backend, algorithm, good)
    theta = backend.arcsin(backend.sqrt(initial))
    circuit = QAA(
        algorithm, oracle, niterations=niterations, measure=False, backend=backend
    )

    assert circuit.nqubits == nqubits
    assert len(circuit.queue) == len(algorithm.queue) + niterations * len(
        amplification_operator(algorithm, oracle).queue
    )
    assert _good_probability(backend, circuit, good) == pytest.approx(
        backend.sin((2 * niterations + 1) * theta) ** 2, abs=1e-8
    )


@pytest.mark.parametrize("seed", range(6))
def test_qaa_probability(backend, seed):
    """The probability is at least max(a, 1 - a) with m = floor(pi / (4 theta))."""
    algorithm, oracle, good = _problem(backend, 3, seed=seed)

    initial = _good_probability(backend, algorithm, good)
    niterations = int(
        backend.floor(math.pi / (4 * backend.arcsin(backend.sqrt(initial))))
    )
    circuit = QAA(
        algorithm, oracle, probability=initial, measure=False, backend=backend
    )

    assert len(circuit.queue) == len(algorithm.queue) + niterations * len(
        amplification_operator(algorithm, oracle).queue
    )
    assert _good_probability(backend, circuit, good) >= max(initial, 1 - initial) - 1e-8


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_qsearch(backend, nqubits):
    algorithm, oracle = _grover(nqubits)
    target = "1" * nqubits

    backend.set_seed(0)
    bitstring, napplications = QSearch(
        algorithm,
        oracle,
        check=lambda bitstring: bitstring == target,
        seed=0,
        backend=backend,
    )

    assert bitstring == target
    assert napplications >= 1


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"check": None}, TypeError),
        ({"growth": 1.0}, ValueError),
        ({"growth": 2.0}, ValueError),
        ({"growth": "1.5"}, ValueError),
        ({"max_rounds": 0}, ValueError),
        ({"max_rounds": 1.5}, ValueError),
    ],
)
def test_qsearch_errors(kwargs, error):
    algorithm, oracle = _grover(2)
    kwargs = {"check": lambda bitstring: True, **kwargs}

    with pytest.raises(error):
        QSearch(algorithm, oracle, **kwargs)

    with pytest.raises(TypeError):
        QSearch(algorithm, None, check=lambda bitstring: True)


def test_qsearch_generator_seed(backend):
    algorithm, oracle = _grover(3)

    bitstring, _ = QSearch(
        algorithm,
        oracle,
        check=lambda bitstring: bitstring == "111",
        seed=np.random.default_rng(1),
        backend=backend,
    )

    assert bitstring == "111"


def test_qsearch_no_good_state(backend):
    algorithm, oracle = _grover(2)

    bitstring, napplications = QSearch(
        algorithm,
        oracle,
        check=lambda bitstring: False,
        max_rounds=3,
        seed=0,
        backend=backend,
    )

    assert bitstring is None
    # each round applies the algorithm at least 4 times
    assert napplications >= 12


def test_qsearch_probability_one(backend):
    """The first measurement of the algorithm is already good, and uses it once."""
    algorithm, oracle = Circuit(1), Circuit(1)
    algorithm.add(gates.X(0))
    oracle.add(gates.Z(0))

    result = QSearch(
        algorithm, oracle, check=lambda bitstring: bitstring == "1", backend=backend
    )

    assert result == ("1", 1)


def _good_probability(backend, circuit, good):
    state = backend.execute_circuit(circuit).state()
    return float(sum(backend.abs(state[index]) ** 2 for index in good))


def _grover(nqubits):
    """Hadamard gates and the oracle that marks the state with all qubits equal to 1."""
    algorithm = Circuit(nqubits)
    algorithm.add(gates.H(qubit) for qubit in range(nqubits))
    oracle = Circuit(nqubits)
    oracle.add(gates.Z(nqubits - 1).controlled_by(*range(nqubits - 1)))
    return algorithm, oracle


def _problem(backend, nqubits, seed):
    """Random unitary as the algorithm and a diagonal oracle that marks random states."""
    rng = np.random.default_rng(seed)
    dim = 2**nqubits
    good = sorted(rng.choice(dim, size=rng.integers(1, dim // 2 + 1), replace=False))

    algorithm = Circuit(nqubits)
    algorithm.add(
        gates.Unitary(random_unitary(dim, seed=seed, backend=backend), *range(nqubits))
    )

    signs = np.ones(dim)
    signs[good] = -1
    oracle = Circuit(nqubits)
    oracle.add(gates.Unitary(backend.cast(np.diag(signs)), *range(nqubits)))

    return algorithm, oracle, [int(index) for index in good]
