"""Tests for the Quantum Amplitude Amplification circuits defined in `qibo/models/algorithms/qaa.py`."""

import math
import random

import pytest

from qibo import Circuit, gates
from qibo.models import FPQAA, OAA, QAA, QSearch, amplification_operator
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
        fixed_point = FPQAA(
            algorithm, oracle, 1.0, niterations, measure=False, backend=backend
        )
        grover = QAA(
            algorithm, oracle, niterations=niterations, measure=False, backend=backend
        )

        backend.assert_allclose(
            _good_probability(backend, fixed_point, [2 * index for index in good]),
            _good_probability(backend, grover, good),
            atol=1e-8,
            rtol=0.0,
        )


@pytest.mark.parametrize("measure", [False, True])
def test_fpqaa_density_matrix(backend, measure):
    algorithm, oracle = _grover(2)
    circuit = FPQAA(
        algorithm, oracle, 0.1, 3, measure=measure, density_matrix=True, backend=backend
    )

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
    niterations = int(
        backend.ceil(
            (backend.arccosh(1 / delta) / backend.arctanh(backend.sqrt(bound)) - 1) / 2
        )
    )

    for extra in (0, 1, 4):
        for angle in (0.1, 0.3, 0.6, 1.0, 1.4):
            initial = backend.sin(angle) ** 2
            if initial < bound:
                continue

            algorithm, oracle = Circuit(1), Circuit(1)
            algorithm.add(gates.RY(0, theta=2 * angle))
            oracle.add(gates.Z(0))

            circuit = FPQAA(
                algorithm,
                oracle,
                delta,
                niterations + extra,
                measure=False,
                backend=backend,
            )
            assert _good_probability(backend, circuit, [2]) >= 1 - delta**2 - 1e-8


def test_fpqaa_independent_gates(backend):
    """Each application of the oracle has its own gates, so their parameters are independent."""
    algorithm, oracle = Circuit(2), Circuit(2)
    algorithm.add(gates.H(qubit) for qubit in range(2))
    oracle.add(gates.CU1(0, 1, theta=math.pi))

    circuit = FPQAA(algorithm, oracle, 0.1, 3, measure=False, backend=backend)
    assert len({id(gate) for gate in circuit.queue}) == len(circuit.queue)

    parameters = circuit.get_parameters()
    values = [
        tuple(0.1 * (index + 1) for _ in group)
        for index, group in enumerate(parameters)
    ]
    circuit.set_parameters(values)

    backend.assert_allclose(
        backend.cast(circuit.get_parameters(), dtype="float64"),
        backend.cast(values, dtype="float64"),
    )


@pytest.mark.parametrize("delta", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("niterations", [0, 1, 3])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_fpqaa_probability_equation(backend, nqubits, seed, niterations, delta):
    """The success probability is 1 - delta^2 T_L(T_{1/L}(1/delta) sqrt(1 - a))^2."""
    algorithm, oracle, good = _problem(backend, nqubits, seed=seed + 10 * nqubits)

    initial = float(_good_probability(backend, algorithm, good))
    circuit = FPQAA(
        algorithm, oracle, delta, niterations, measure=False, backend=backend
    )

    assert circuit.nqubits == nqubits + 1

    # the good states are measured with the auxiliary qubit, the last one, in the state 0
    backend.assert_allclose(
        _good_probability(backend, circuit, [2 * index for index in good]),
        _fixed_point_probability(backend, initial, 2 * niterations + 1, delta),
        atol=1e-8,
        rtol=0.0,
    )
    backend.assert_allclose(
        _good_probability(backend, circuit, range(1, 2 ** (nqubits + 1), 2)),
        0.0,
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("delta", [0.01, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("bound", [0.02, 0.25, 0.9, 1.0])
def test_fpqaa_probability_iterations(backend, bound, delta):
    """The number of iterations is the smallest one that satisfies the bound."""
    algorithm, oracle = Circuit(1), Circuit(1)
    algorithm.add(gates.RY(0, theta=float(2 * backend.arcsin(backend.sqrt(bound)))))
    oracle.add(gates.Z(0))

    width = math.inf if bound == 1 else backend.arctanh(backend.sqrt(bound))
    niterations = max(
        int(backend.ceil((backend.arccosh(1 / delta) / width - 1) / 2)), 0
    )

    circuit = FPQAA(
        algorithm, oracle, delta, probability=bound, measure=False, backend=backend
    )
    expected = FPQAA(
        algorithm, oracle, delta, niterations, measure=False, backend=backend
    )
    assert len(circuit.queue) == len(expected.queue)

    success = _good_probability(backend, circuit, [2])
    assert success >= 1 - delta**2 - 1e-8

    # one iteration less cannot satisfy the bound, unless delta is 1 and it is trivial
    if niterations > 0:
        fewer = FPQAA(
            algorithm, oracle, delta, niterations - 1, measure=False, backend=backend
        )
        assert _good_probability(backend, fewer, [2]) < 1 - delta**2


def test_fpqaa_qubits(backend):
    algorithm, oracle = _grover(3)

    circuit = FPQAA(algorithm, oracle, 0.2, 2, measure=False, backend=backend)
    assert circuit.nqubits == 4
    assert len(circuit.measurements) == 0

    # the auxiliary qubit is not measured
    circuit = FPQAA(algorithm, oracle, 0.2, 2, backend=backend)
    assert circuit.measurements[0].target_qubits == (0, 1, 2)

    # without iterations, the circuit is the algorithm
    circuit = FPQAA(algorithm, oracle, 0.2, 0, measure=False, backend=backend)
    assert len(circuit.queue) == len(algorithm.queue)


@pytest.mark.parametrize("exact", [False, True])
def test_oaa_density_matrix(backend, exact):
    theta = math.pi / 6
    unitary, _ = _oblivious_problem(backend, 1, 1, theta, seed=0)
    circuit = OAA(
        unitary, 1, probability=0.25, exact=exact, density_matrix=True, backend=backend
    )

    assert circuit.density_matrix
    assert circuit.nqubits == (3 if exact else 2)

    # the probability of the zero state of the auxiliary qubits is 1, for the input state |00>
    state = backend.execute_circuit(circuit).state()
    probabilities = backend.diag(state)
    backend.assert_allclose(
        backend.sum(probabilities[: 2 ** (circuit.nqubits - 1)]),
        1.0,
        atol=1e-8,
        rtol=0.0,
    )


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
        {"niterations": 1, "nauxiliary": 0},
        {"niterations": 1, "nauxiliary": 3},
        {"niterations": 1, "nauxiliary": 1.0},
        {"niterations": 1, "nauxiliary": True},
    ],
)
def test_oaa_errors(kwargs):
    unitary = Circuit(2)
    unitary.add(gates.RY(0, theta=0.3))
    kwargs = {"nauxiliary": 1, **kwargs}

    with pytest.raises(ValueError):
        OAA(unitary, **kwargs)

    with pytest.raises(TypeError):
        OAA(unitary.queue, 1, niterations=1)

    measured = Circuit(2)
    measured.add(gates.M(0))
    with pytest.raises(ValueError):
        OAA(measured, 1, niterations=1)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("theta", [0.1, 0.3, math.pi / 6, 0.9, math.pi / 2])
@pytest.mark.parametrize("nsystem", [1, 2])
@pytest.mark.parametrize("nauxiliary", [1, 2, 3])
def test_oaa_exact(backend, nauxiliary, nsystem, theta, seed):
    """The unitary V is implemented exactly, for any input state."""
    unitary, matrix = _oblivious_problem(backend, nauxiliary, nsystem, theta, seed)

    circuit = OAA(
        unitary,
        nauxiliary,
        probability=float(backend.sin(theta) ** 2),
        exact=True,
        backend=backend,
    )
    assert circuit.nqubits == unitary.nqubits + 1

    state, psi = _oblivious_output(backend, circuit, nauxiliary, nsystem, seed + 10)

    # the auxiliary qubits and the extra qubit, which is the last one, are in the state 0
    backend.assert_allclose(
        state[: 2 ** (nsystem + 1) : 2], matrix @ psi, atol=1e-8, rtol=0.0
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("niterations", [0, 1, 2, 3])
@pytest.mark.parametrize("theta", [0.3, math.pi / 6, 1.0])
@pytest.mark.parametrize("nsystem", [1, 2])
@pytest.mark.parametrize("nauxiliary", [1, 2, 3])
def test_oaa_niterations(backend, nauxiliary, nsystem, theta, niterations, seed):
    """The amplitude of V is sin((2l + 1) theta), for any input state."""
    unitary, matrix = _oblivious_problem(backend, nauxiliary, nsystem, theta, seed)

    circuit = OAA(unitary, nauxiliary, niterations=niterations, backend=backend)
    assert circuit.nqubits == unitary.nqubits
    assert len(circuit.queue) >= len(unitary.queue)

    state, psi = _oblivious_output(backend, circuit, nauxiliary, nsystem, seed + 10)

    amplitude = backend.sin((2 * niterations + 1) * theta)
    backend.assert_allclose(
        state[: 2**nsystem], amplitude * (matrix @ psi), atol=1e-8, rtol=0.0
    )


@pytest.mark.parametrize("theta", [0.2, 0.5, 0.8, 1.2])
def test_oaa_probability(backend, theta):
    """The probability is at least max(p, 1 - p) with l = floor(pi / (4 theta))."""
    unitary, _ = _oblivious_problem(backend, 1, 2, theta, seed=3)

    probability = float(backend.sin(theta) ** 2)
    niterations = int(backend.floor(math.pi / (4 * theta)))
    circuit = OAA(unitary, 1, probability=probability, backend=backend)
    expected = OAA(unitary, 1, niterations=niterations, backend=backend)
    assert len(circuit.queue) == len(expected.queue)

    state, _ = _oblivious_output(backend, circuit, 1, 2, seed=4)
    success = backend.sum(backend.abs(state[:4]) ** 2)

    assert success >= max(probability, 1 - probability) - 1e-8
    backend.assert_allclose(
        success,
        backend.sin((2 * niterations + 1) * theta) ** 2,
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("measure", [False, True])
def test_qaa_density_matrix(backend, measure):
    algorithm, oracle = _grover(2)
    circuit = QAA(
        algorithm,
        oracle,
        niterations=1,
        measure=measure,
        density_matrix=True,
        backend=backend,
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

    backend.assert_allclose(
        _good_probability(backend, circuit, [7]), 1.0, atol=1e-8, rtol=0.0
    )


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
    backend.assert_allclose(
        _good_probability(backend, circuit, good),
        backend.sin((2 * niterations + 1) * theta) ** 2,
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("seed", range(6))
def test_qaa_probability(backend, seed):
    """The probability is at least max(a, 1 - a) with m = floor(pi / (4 theta))."""
    algorithm, oracle, good = _problem(backend, 3, seed=seed)

    initial = float(_good_probability(backend, algorithm, good))
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


def _chebyshev(backend, degree, x):
    """Chebyshev polynomial of the first kind, which is extended for ``x > 1``."""
    if abs(x) <= 1:
        return backend.cos(degree * backend.arccos(x))
    return backend.cosh(degree * backend.arccosh(x))


def _fixed_point_probability(backend, initial, length, delta):
    """Success probability of the fixed-point amplification."""
    if delta == 0:
        return 1 - (1 - initial) ** length

    factor = backend.cosh(backend.arccosh(1 / delta) / length)
    return (
        1
        - delta**2
        * _chebyshev(backend, length, factor * backend.sqrt(1 - initial)) ** 2
    )


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


def _oblivious_output(backend, circuit, nauxiliary, nsystem, seed):
    """Final state when the system qubits are prepared in a random state, and that state.

    The auxiliary qubits are the first ones, and the system qubits are the following ones.
    """
    dim = 2**nsystem
    preparation = random_unitary(dim, seed=seed, backend=backend)

    prepared = Circuit(circuit.nqubits, density_matrix=circuit.density_matrix)
    prepared.add(gates.Unitary(preparation, *range(nauxiliary, nauxiliary + nsystem)))
    prepared.add(circuit)

    state = backend.execute_circuit(prepared).state()
    return state, preparation[:, 0]


def _oblivious_problem(backend, nauxiliary, nsystem, theta, seed):
    """Unitary U that applies V with amplitude sin(theta) when the auxiliary qubits are 0.

    The state with the auxiliary qubits different from 0 depends on the system state, since it
    is rotated by another random unitary, so it is not a reflection about the input state. If
    there are several auxiliary qubits, the first one is 0 in this state, so that all of them
    are needed to tell it from the zero state.
    """
    dim = 2**nsystem
    system = list(range(nauxiliary, nauxiliary + nsystem))
    matrix = random_unitary(dim, seed=seed, backend=backend)
    other = random_unitary(dim, seed=seed + 1, backend=backend)

    unitary = Circuit(nauxiliary + nsystem)
    unitary.add(gates.Unitary(matrix, *system))
    unitary.add(gates.RY(0, theta=math.pi - 2 * theta))
    unitary.add(gates.CNOT(0, qubit) for qubit in range(1, nauxiliary))
    unitary.add(gates.Unitary(other, *system).controlled_by(nauxiliary - 1))
    if nauxiliary > 1:
        unitary.add(gates.CNOT(1, 0))

    return unitary, matrix


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
