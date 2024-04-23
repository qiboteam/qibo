"""Test Grover model defined in `qibo/models/grover.py`."""

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
        oracle, superposition_circuit=superposition, superposition_size=int(2**5)
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
    solution, iterations = grover(logs=True, backend=backend)
    assert solution == ["11111"]


def test_grover_target_amplitude(backend):
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(oracle, superposition_qubits=5, target_amplitude=1 / 2 ** (5 / 2))
    solution, iterations = grover(logs=True, backend=backend)
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
    solution, iterations = grover(logs=True, backend=backend)
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
        solution, iterations = grover(backend=backend)
    grover = Grover(oracle, superposition_qubits=5, check=check_false, iterative=True)
    solution, iterations = grover(backend=backend)
    grover = Grover(oracle, superposition_qubits=5, check=check, iterative=True)
    solution, iterations = grover(logs=True, backend=backend)
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
