"""Tests for the Quantum Amplitude Amplification circuits defined in `qibo/models/qaa.py`."""

import math
import random

import pytest

from qibo import Circuit, gates
from qibo.models import FPQAA, QAA, QSearch, amplification_operator
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
    with pytest.raises(ValueError):
        FPQAA(algorithm, measured, 0.5, niterations=1)


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_amplification_operator(backend, nqubits):
    """The circuit is the operator -A S_0 A^{-1} S_chi, with its global phase."""
    dim = 2**nqubits
    algorithm, oracle, good = _problem(backend, nqubits, seed=nqubits)

    s_zero = backend.diag(backend.cast([-1] + [1] * (dim - 1)))
    s_chi = backend.diag(
        backend.cast([-1 if index in good else 1 for index in range(dim)])
    )

    matrix = algorithm.unitary(backend)
    target = -matrix @ s_zero @ backend.dagger(matrix) @ s_chi

    operator = amplification_operator(algorithm, oracle)
    assert operator.nqubits == nqubits
    backend.assert_allclose(operator.unitary(backend), target, atol=1e-8, rtol=0.0)


def test_amplification_operator_grover(backend):
    """With Hadamard gates, one application on 2 qubits finds the marked state."""
    algorithm, oracle = _grover(2)
    circuit = Circuit(2)
    circuit.add(algorithm)
    circuit.add(amplification_operator(algorithm, oracle))

    backend.assert_allclose(
        _good_probability(backend, circuit, [3]), 1.0, atol=1e-8, rtol=0.0
    )


def test_fpqaa_delta_one(backend):
    """For delta equal to 1, the circuit is the one of Grover's search."""
    algorithm, oracle, good = _problem(backend, 3, seed=7)

    for niterations in range(5):
        fixed_point = FPQAA(algorithm, oracle, 1.0, niterations, measure=False)
        grover = QAA(algorithm, oracle, niterations=niterations, measure=False)

        backend.assert_allclose(
            _good_probability(backend, fixed_point, [2 * index for index in good]),
            _good_probability(backend, grover, good),
            atol=1e-8,
            rtol=0.0,
        )


@pytest.mark.parametrize("measure", [False, True])
def test_fpqaa_density_matrix(backend, measure):
    algorithm, oracle = _grover(2)
    circuit = FPQAA(algorithm, oracle, 0.1, 3, measure=measure, density_matrix=True)

    assert circuit.density_matrix
    if measure:
        frequencies = backend.execute_circuit(circuit, nshots=20).frequencies()
        assert frequencies.most_common(1)[0][0] == "11"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"niterations": 1, "probability": 0.5},
        {"niterations": -1},
        {"niterations": 1.5},
        {"niterations": True},
        {"probability": 0.0},
        {"probability": 1.5},
        {"probability": "0.5"},
        {"probability": True},
        {"probability": 0.5, "delta": 0.0},
        {"niterations": 1, "delta": -0.1},
        {"niterations": 1, "delta": 1.1},
        {"niterations": 1, "delta": "0.5"},
        {"niterations": 1, "delta": True},
    ],
)
def test_fpqaa_errors(kwargs):
    algorithm, oracle = _grover(2)
    kwargs = {"delta": 0.5, **kwargs}

    with pytest.raises(ValueError):
        FPQAA(algorithm, oracle, **kwargs)


@pytest.mark.parametrize("delta", [0.1, 0.3])
@pytest.mark.parametrize("bound", [0.04, 0.2])
def test_fpqaa_fixed_point(backend, bound, delta):
    """The failure probability stays bounded for every probability above the bound,
    even if the number of iterations is increased."""
    niterations = math.ceil(
        (math.acosh(1 / delta) / math.atanh(math.sqrt(bound)) - 1) / 2
    )

    for extra in (0, 1, 4):
        for angle in (0.1, 0.3, 0.6, 1.0, 1.4):
            initial = math.sin(angle) ** 2
            if initial < bound:
                continue

            algorithm, oracle = Circuit(1), Circuit(1)
            algorithm.add(gates.RY(0, theta=2 * angle))
            oracle.add(gates.Z(0))

            circuit = FPQAA(
                algorithm, oracle, delta, niterations + extra, measure=False
            )
            assert _good_probability(backend, circuit, [2]) >= 1 - delta**2 - 1e-8


@pytest.mark.parametrize("delta", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("niterations", [0, 1, 3])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_fpqaa_probability_equation(backend, nqubits, seed, niterations, delta):
    """The success probability is 1 - delta^2 T_L(T_{1/L}(1/delta) sqrt(1 - a))^2."""
    algorithm, oracle, good = _problem(backend, nqubits, seed=seed + 10 * nqubits)

    initial = float(_good_probability(backend, algorithm, good))
    circuit = FPQAA(algorithm, oracle, delta, niterations, measure=False)

    assert circuit.nqubits == nqubits + 1

    # the good states are measured with the auxiliary qubit, the last one, in the state 0
    backend.assert_allclose(
        _good_probability(backend, circuit, [2 * index for index in good]),
        _fixed_point_probability(initial, 2 * niterations + 1, delta),
        atol=1e-8,
        rtol=0.0,
    )
    backend.assert_allclose(
        _good_probability(backend, circuit, range(1, 2 ** (nqubits + 1), 2)),
        0.0,
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("bound", [0.02, 0.25, 1.0])
def test_fpqaa_probability_iterations(backend, bound, delta):
    """The number of iterations is the smallest one that satisfies the bound."""
    algorithm, oracle = Circuit(1), Circuit(1)
    algorithm.add(gates.RY(0, theta=2 * math.asin(math.sqrt(bound))))
    oracle.add(gates.Z(0))

    width = math.inf if bound == 1 else math.atanh(math.sqrt(bound))
    niterations = max(math.ceil((math.acosh(1 / delta) / width - 1) / 2), 0)

    circuit = FPQAA(algorithm, oracle, delta, probability=bound, measure=False)
    expected = FPQAA(algorithm, oracle, delta, niterations, measure=False)
    assert len(circuit.queue) == len(expected.queue)

    success = _good_probability(backend, circuit, [2])
    assert success >= 1 - delta**2 - 1e-8

    # one iteration less cannot satisfy the bound, unless delta is 1 and it is trivial
    if niterations > 0:
        fewer = FPQAA(algorithm, oracle, delta, niterations - 1, measure=False)
        assert _good_probability(backend, fewer, [2]) < 1 - delta**2


def test_fpqaa_qubits(backend):
    algorithm, oracle = _grover(3)

    circuit = FPQAA(algorithm, oracle, 0.2, 2, measure=False)
    assert circuit.nqubits == 4
    assert len(circuit.measurements) == 0

    # the auxiliary qubit is not measured
    circuit = FPQAA(algorithm, oracle, 0.2, 2)
    assert circuit.measurements[0].target_qubits == (0, 1, 2)

    # without iterations, the circuit is the algorithm
    circuit = FPQAA(algorithm, oracle, 0.2, 0, measure=False)
    assert len(circuit.queue) == len(algorithm.queue)


@pytest.mark.parametrize("niterations", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_qaa_niterations(backend, nqubits, niterations):
    """After m iterations the success probability is sin^2((2m + 1) theta)."""
    algorithm, oracle, good = _problem(backend, nqubits, seed=10 * nqubits)

    initial = _good_probability(backend, algorithm, good)
    theta = backend.arcsin(backend.sqrt(initial))
    circuit = QAA(algorithm, oracle, niterations=niterations, measure=False)

    assert circuit.nqubits == nqubits
    assert len(circuit.queue) == len(algorithm.queue) + niterations * len(
        amplification_operator(algorithm, oracle).queue
    )
    backend.assert_allclose(
        _good_probability(backend, circuit, good),
        backend.sin((2 * niterations + 1) * theta) ** 2,
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("measure", [False, True])
def test_qaa_density_matrix(backend, measure):
    algorithm, oracle = _grover(2)
    circuit = QAA(
        algorithm, oracle, niterations=1, measure=measure, density_matrix=True
    )

    assert circuit.density_matrix
    if measure:
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

    initial = float(_good_probability(backend, algorithm, good))
    circuit = QAA(algorithm, oracle, probability=initial, exact=True, measure=False)

    # the last qubit is the auxiliary qubit, and a good state has it in the state 1
    assert circuit.nqubits == nqubits + 1
    backend.assert_allclose(
        _good_probability(backend, circuit, [2 * index + 1 for index in good]),
        1.0,
        atol=1e-8,
        rtol=0.0,
    )


def test_qaa_exact_integer_iterations(backend):
    """The auxiliary qubit does not spoil the cases that are exact without it."""
    # a = 1 / 4 needs exactly one iteration
    algorithm, oracle = _grover(2)
    circuit = QAA(algorithm, oracle, probability=1 / 4, exact=True)
    assert circuit.nqubits == 3
    frequencies = backend.execute_circuit(circuit, nshots=20).frequencies()
    assert frequencies == {"11": 20}

    # a = 1 needs no iterations
    algorithm, oracle = Circuit(1), Circuit(1)
    algorithm.add(gates.X(0))
    oracle.add(gates.Z(0))
    circuit = QAA(algorithm, oracle, probability=1.0, exact=True)
    assert circuit.nqubits == 2
    frequencies = backend.execute_circuit(circuit, nshots=20).frequencies()
    assert frequencies == {"1": 20}


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_qaa_grover(backend, nqubits):
    """Grover's search is recovered for a single marked state."""
    algorithm, oracle = _grover(nqubits)
    circuit = QAA(algorithm, oracle, probability=1 / 2**nqubits)

    backend.set_seed(0)
    frequencies = backend.execute_circuit(circuit, nshots=200).frequencies()

    assert frequencies.most_common(1)[0][0] == "1" * nqubits
    assert frequencies["1" * nqubits] > 150


def test_qaa_larger_circuit(backend):
    """The circuit can be added to a subset of the qubits of a larger one."""
    algorithm, oracle = _grover(2)
    circuit = Circuit(3)
    circuit.add(gates.X(2))
    circuit.add(QAA(algorithm, oracle, niterations=1, measure=False).on_qubits(0, 1))

    backend.assert_allclose(
        _good_probability(backend, circuit, [7]), 1.0, atol=1e-8, rtol=0.0
    )


def test_qaa_measure(backend):
    algorithm, oracle = _grover(2)

    circuit = QAA(algorithm, oracle, niterations=1, measure=False)
    assert len(circuit.measurements) == 0

    circuit = QAA(algorithm, oracle, niterations=1)
    assert len(circuit.measurements) == 1
    assert circuit.measurements[0].target_qubits == (0, 1)

    # the auxiliary qubit is not measured
    circuit = QAA(algorithm, oracle, probability=1 / 4, exact=True)
    assert circuit.nqubits == 3
    assert circuit.measurements[0].target_qubits == (0, 1)


@pytest.mark.parametrize("seed", range(6))
def test_qaa_probability(backend, seed):
    """The probability is at least max(a, 1 - a) with m = floor(pi / (4 theta))."""
    algorithm, oracle, good = _problem(backend, 3, seed=seed)

    initial = float(_good_probability(backend, algorithm, good))
    niterations = math.floor(math.pi / (4 * math.asin(math.sqrt(initial))))
    circuit = QAA(algorithm, oracle, probability=initial, measure=False)

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


def _chebyshev(degree, x):
    """Chebyshev polynomial of the first kind, which is extended for ``x > 1``."""
    if abs(x) <= 1:
        return math.cos(degree * math.acos(x))
    return math.cosh(degree * math.acosh(x))


def _fixed_point_probability(initial, length, delta):
    """Success probability of the fixed-point amplification."""
    if delta == 0:
        return 1 - (1 - initial) ** length

    factor = math.cosh(math.acosh(1 / delta) / length)
    return 1 - delta**2 * _chebyshev(length, factor * math.sqrt(1 - initial)) ** 2


def _good_probability(backend, circuit, good):
    state = backend.execute_circuit(circuit).state()
    return backend.sum(backend.abs(state[list(good)]) ** 2)


def _grover(nqubits):
    """Hadamard gates and the oracle that marks the state with all qubits equal to 1."""
    algorithm = Circuit(nqubits)
    algorithm.add(gates.H(qubit) for qubit in range(nqubits))
    oracle = Circuit(nqubits)
    oracle.add(gates.Z(nqubits - 1).controlled_by(*range(nqubits - 1)))
    return algorithm, oracle


def _problem(backend, nqubits, seed):
    """Random unitary as the algorithm and a diagonal oracle that marks random states."""
    rng = random.Random(seed)
    dim = 2**nqubits
    good = sorted(rng.sample(range(dim), rng.randint(1, dim // 2)))

    algorithm = Circuit(nqubits)
    algorithm.add(
        gates.Unitary(random_unitary(dim, seed=seed, backend=backend), *range(nqubits))
    )

    oracle = Circuit(nqubits)
    oracle.add(
        gates.Unitary(
            backend.diag(
                backend.cast([-1 if index in good else 1 for index in range(dim)])
            ),
            *range(nqubits),
        )
    )

    return algorithm, oracle, good
