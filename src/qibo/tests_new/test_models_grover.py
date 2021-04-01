"""Test Grover model defined in `qibo/models/grover.py`."""
import pytest
import qibo
from qibo import gates
from qibo.models import Circuit, Grover


def test_grover_init(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
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
    grover = Grover(oracle, superposition_circuit=superposition, superposition_size=int(2**5))
    assert grover.oracle == oracle
    assert grover.superposition == superposition
    assert grover.sup_qubits == 5
    assert grover.sup_size == 32
    assert not grover.iterative
    qibo.set_backend(original_backend)


def test_grover_init_default_superposition(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
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
    qibo.set_backend(original_backend)


def test_grover_initial_state(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    initial_state = Circuit(5)
    initial_state.add(gates.X(4))
    grover = Grover(oracle, superposition_qubits=5, initial_state_circuit=initial_state, number_solutions=1)
    assert grover.initial_state_circuit == initial_state
    solution, iterations = grover(logs=True)
    assert solution == ["11111"]
    qibo.set_backend(original_backend)


def test_grover_wrong_solution(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    def check(result):
        for i in result:
            if int(i) != 1:
                return False
        return True

    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(oracle, superposition_qubits=5, check=check, number_solutions=2)
    solution, iterations = grover()
    assert len(solution) == 2
    qibo.set_backend(original_backend)


def test_grover_iterative(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

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
        solution, iterations = grover()
    grover = Grover(oracle, superposition_qubits=5, check=check_false, iterative=True)
    with pytest.raises(TimeoutError):
        solution, iterations = grover()
    grover = Grover(oracle, superposition_qubits=5, check=check, iterative=True)
    solution, iterations = grover()
    assert solution == "11111"
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("num_sol", [None, 1])
def test_grover_execute(backend, num_sol):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    def check(result):
        for i in result:
            if int(i) != 1:
                return False
        return True

    oracle = Circuit(5 + 1)
    oracle.add(gates.X(5).controlled_by(*range(5)))
    grover = Grover(oracle, superposition_qubits=5, check=check, number_solutions=num_sol)
    solution, iterations = grover(freq=True)
    if num_sol:
        assert solution == ["11111"]
        assert iterations == 4
    else:
        assert solution == "11111"
    qibo.set_backend(original_backend)
