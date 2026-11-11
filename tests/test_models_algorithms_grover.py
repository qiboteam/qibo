"""Test Grover model defined in `qibo/models/algorithms/grover.py`."""

import math

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.models import Grover


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
