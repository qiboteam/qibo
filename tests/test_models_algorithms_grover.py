"""Test Grover model defined in `qibo/models/algorithms/grover.py`."""

import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.models import GAS, Grover


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"objective": None}, TypeError),
        ({"nqubits": 0}, ValueError),
        ({"nqubits": 1.5}, ValueError),
        ({"nqubits": True}, ValueError),
        ({"max_iterations": 0}, ValueError),
        ({"max_iterations": 2.5}, ValueError),
        ({"rotations": 0}, ValueError),
        ({"rotations": "1"}, ValueError),
    ],
)
def test_gas_errors(kwargs, error):
    kwargs = {
        "objective": lambda bitstring: 0,
        "nqubits": 2,
        "max_iterations": 3,
        **kwargs,
    }

    with pytest.raises(error):
        GAS(**kwargs)


def test_gas_expected_iterations(backend):
    """The mean number of iterations is the expectation of the Markov chain of the algorithm.

    The values of the function are the Hamming weights of the bitstrings, so the proportions of
    the domain with each value are 1/8, 3/8, 3/8 and 1/8. From the level :math:`j`, an
    iteration finds a better value with probability :math:`g_{r}(p_{j - 1})`, where
    :math:`p_{j - 1}` is the proportion of the domain that is better, and then it moves to the
    level :math:`i < j` with probability :math:`\\pi_{i} / p_{j - 1}`.
    """
    nqubits, rotations, nruns = 3, 1, 300
    proportions = [1 / 8, 3 / 8, 3 / 8, 1 / 8]
    cumulative = [sum(proportions[: level + 1]) for level in range(4)]

    # first and second moments of the number of iterations from each level, where the number
    # of iterations that are needed to find a better value is geometric
    first, second = [0.0], [0.0]
    for level in range(1, 4):
        better = cumulative[level - 1]
        success = math.sin((2 * rotations + 1) * math.asin(math.sqrt(better))) ** 2
        weights = [proportions[other] / better for other in range(level)]
        mean_next = sum(w * first[other] for other, w in enumerate(weights))
        second_next = sum(w * second[other] for other, w in enumerate(weights))
        first.append(1 / success + mean_next)
        second.append(
            (2 - success) / success**2 + 2 / success * mean_next + second_next
        )
    expected = sum(p * mean for p, mean in zip(proportions, first))
    variance = sum(p * moment for p, moment in zip(proportions, second)) - expected**2

    backend.set_seed(0)
    generator = backend.default_rng(0)
    iterations = []
    for _ in range(nruns):
        bitstring, value, niterations = GAS(
            lambda bitstring: bitstring.count("1"),
            nqubits,
            1000,
            rotations,
            target=0,
            seed=generator,
            backend=backend,
        )
        assert (bitstring, value) == ("000", 0)
        iterations.append(niterations)

    assert abs(sum(iterations) / nruns - expected) < 4 * math.sqrt(variance / nruns)


@pytest.mark.parametrize("max_iterations", [1, 6])
def test_gas_max_iterations(backend, max_iterations):
    """Without a target, the algorithm runs the maximum number of iterations."""
    backend.set_seed(max_iterations)

    bitstring, value, iterations = GAS(
        lambda bitstring: int(bitstring, 2),
        3,
        max_iterations,
        seed=max_iterations,
        backend=backend,
    )

    assert len(bitstring) == 3
    assert value == int(bitstring, 2)
    assert iterations == max_iterations


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_gas_pure_adaptive_search(backend, nqubits):
    """If the probability of a marked point is one, GAS is pure adaptive search.

    A quarter of the domain has the minimum value, and for one rotation
    :math:`g_{1}(1 / 4) = 1`, so a Grover run always finds a minimum, and the algorithm needs at
    most one iteration.
    """

    def objective(bitstring):
        return 0 if bitstring[:2] == "10" else 1

    backend.set_seed(nqubits)
    generator = backend.default_rng(nqubits)
    found = set()
    for _ in range(12):
        bitstring, value, iterations = GAS(
            objective, nqubits, 5, 1, target=0, seed=generator, backend=backend
        )

        assert bitstring[:2] == "10"
        assert value == 0
        assert iterations <= 1
        found.add(iterations)

    # three quarters of the initial points are not minima
    assert 1 in found


def test_gas_stalled_search(backend):
    """The algorithm cannot improve a value if :math:`g_{r}` of the improving region is zero.

    Three quarters of the domain has the minimum value, and for one rotation
    :math:`g_{1}(3 / 4) = 0`, so a run that starts at the other point never leaves it, and the
    algorithm stops because of the maximum number of iterations.
    """

    def objective(bitstring):
        return 1 if bitstring == "11" else 0

    backend.set_seed(0)
    generator = backend.default_rng(0)
    outcomes = set()
    for _ in range(12):
        _bitstring, value, iterations = GAS(
            objective, 2, 4, 1, target=0, seed=generator, backend=backend
        )
        outcomes.add((iterations, value))

    assert outcomes == {(0, 0), (4, 1)}


def test_gas_target(backend):
    """The algorithm stops without iterations if the initial point reaches the target."""
    values = {"00": 3, "01": 1, "10": 2, "11": 0}

    bitstring, value, iterations = GAS(
        values.__getitem__, 2, 5, target=3, seed=0, backend=backend
    )

    assert iterations == 0
    assert value == values[bitstring]


@pytest.mark.parametrize("num_sol", [None, 1])
def test_grover_execute(backend, num_sol):
    def check(result):
        for i in result:
            if int(i) != 1:
                return False
        return True

    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(
        oracle, superposition_qubits=5, check=check, number_solutions=num_sol
    )
    solution, iterations = grover(freq=True, logs=True, backend=backend)
    if num_sol:
        assert solution == ["11111"]
        assert iterations == 4
    else:
        assert solution == "11111"


def test_grover_init(backend):
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    superposition = Circuit(5)
    superposition.add([gates.H(i) for i in range(5)])
    grover = Grover(oracle, superposition_circuit=superposition)
    assert grover.oracle == oracle
    assert grover.superposition == superposition
    assert grover.sup_qubits == 5
    assert grover.sup_size == 32
    assert not grover.iterative
    grover = Grover(
        oracle, superposition_circuit=superposition, superposition_size=(2**5)
    )
    assert grover.oracle == oracle
    assert grover.superposition == superposition
    assert grover.sup_qubits == 5
    assert grover.sup_size == 32
    assert not grover.iterative


def test_grover_init_default_superposition(backend):
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    # try to initialize without passing `superposition_qubits`
    with pytest.raises(ValueError):
        grover = Grover(oracle)

    grover = Grover(oracle, superposition_qubits=4)
    assert grover.oracle == oracle
    assert grover.sup_qubits == 4
    assert grover.sup_size == 16
    assert grover.superposition.depth == 1
    assert grover.superposition.ngates == 4


def test_grover_initial_state(backend):
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    initial_state = Circuit(5)
    initial_state.add(gates.X(4))
    grover = Grover(
        oracle,
        superposition_qubits=5,
        initial_state_circuit=initial_state,
        number_solutions=1,
    )
    assert grover.initial_state_circuit == initial_state
    solution, _iterations = grover(logs=True, backend=backend)
    assert solution == ["11111"]


@pytest.mark.parametrize("nbits,nsol", [(4, 9), (8, 39)])
def test_grover_iterations(backend, nbits, nsol):
    # the number of iterations is floor(pi / (4 theta)), with sin^2(theta) = t / N,
    # for which the probability of failure is at most t / N
    nstates = 2**nbits
    marked = set(range(nsol))
    grover = Grover(
        _oracle(marked, nbits), superposition_qubits=nbits, number_solutions=nsol
    )
    _solution, iterations = grover(backend=backend)
    theta = math.asin(math.sqrt(nsol / nstates))
    assert iterations == math.floor(math.pi / (4 * theta))

    circuit = grover.initialize()
    for _ in range(iterations):
        circuit += grover.step()
    probabilities = backend.execute_circuit(circuit).probabilities(list(range(nbits)))
    success = sum(float(probabilities[state]) for state in marked)
    assert success >= 1 - nsol / nstates


def test_grover_iterative(backend):
    def check(result):
        for i in result:
            if int(i) != 1:
                return False
        return True

    def check_false(result):
        return False

    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(oracle, superposition_qubits=5, check=None, iterative=True)
    with pytest.raises(ValueError):
        grover(backend=backend)
    grover = Grover(oracle, superposition_qubits=5, check=check_false, iterative=True)
    solution, _iterations = grover(backend=backend)
    grover = Grover(oracle, superposition_qubits=5, check=check, iterative=True)
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        backend.set_seed(0)
        solution, _iterations = grover(logs=True, backend=backend)
    finally:
        np.random.set_state(rng_state)
    assert solution == "11111"


def test_grover_iterative_first_round(backend):
    # in the first round m = 1, hence j = 0 and the initial state is measured and checked
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(
        oracle, superposition_qubits=5, check=lambda _bitstring: True, iterative=True
    )
    solution, iterations = grover(backend=backend)
    assert iterations == 0
    assert len(solution) == 5


def test_grover_iterative_rounds(backend, monkeypatch):
    # in the round s, j is drawn among the non-negative integers smaller than
    # m = min(lambda^s, sqrt(N)), and the circuit is measured in every round
    iterations_per_round = []
    circuit = Grover.circuit

    def circuit_recorder(self, iterations):
        iterations_per_round.append(iterations)
        return circuit(self, iterations)

    monkeypatch.setattr(Grover, "circuit", circuit_recorder)
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(
        oracle, superposition_qubits=5, check=lambda _bitstring: False, iterative=True
    )
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        grover(backend=backend)
    finally:
        np.random.set_state(rng_state)

    assert iterations_per_round[0] == 0
    for round_, iterations in enumerate(iterations_per_round):
        assert iterations < math.ceil(min(1.2**round_, math.sqrt(32)))
    assert sum(iterations_per_round) > 9 / 4 * math.sqrt(32)


def test_grover_target_amplitude(backend):
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(oracle, superposition_qubits=5, target_amplitude=1 / 2 ** (5 / 2))
    solution, _iterations = grover(logs=True, backend=backend)
    assert len(solution) == 1
    assert solution == ["11111"]


def test_grover_wrong_solution(backend):
    def check(result):
        for i in result:
            if int(i) != 1:
                return False
        return True

    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(oracle, superposition_qubits=5, check=check, number_solutions=2)
    solution, _iterations = grover(logs=True, backend=backend)
    assert len(solution) == 2


def _oracle(marked, nbits):
    # oracle that flips the last qubit when the first ones are in a marked state
    dim = 2**nbits
    matrix = np.zeros((2 * dim, 2 * dim))
    for state in range(dim):
        for target in range(2):
            matrix[2 * state + (target ^ (state in marked)), 2 * state + target] = 1
    oracle = Circuit(nbits + 1)
    oracle.add(gates.Unitary(matrix, *range(nbits + 1)))
    return oracle
