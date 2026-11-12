"""Tests for the Hadamard and swap tests in `qibo/models/algorithms/interference_tests.py`."""

import logging
from functools import reduce

import pytest

from qibo import Circuit, gates, hamiltonians
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models import hadamard_test, swap_test
from qibo.quantum_info import random_hermitian, random_unitary
from qibo.symbols import Symbol, X, Y, Z


def _random_circuit(nqubits, seed, backend):
    circuit = Circuit(nqubits)
    circuit.add(
        gates.Unitary(
            random_unitary(2**nqubits, seed=seed, backend=backend), *range(nqubits)
        )
    )
    return circuit


@pytest.mark.parametrize("measure", [True, False])
@pytest.mark.parametrize(
    "kind", ["none", "circuit", "dense", "tfim", "symbolic", "list", "tuple"]
)
@pytest.mark.parametrize("imaginary", [False, True])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_hadamard_test(backend, nqubits, imaginary, kind, measure, caplog):
    """The probability of the first qubit gives the part of the amplitude."""
    state_1 = _random_circuit(nqubits, 1, backend)
    state_2 = _random_circuit(nqubits, 2, backend)

    observable, matrix = None, None
    if kind == "circuit":
        observable = _random_circuit(nqubits, 3, backend)
        matrix = observable.unitary(backend)
    elif kind == "dense":
        matrix = random_hermitian(2**nqubits, seed=4, backend=backend)
        observable = Hamiltonian(nqubits, matrix, backend=backend)
    elif kind == "tfim":
        observable = hamiltonians.TFIM(nqubits, h=0.8, dense=False, backend=backend)
        matrix = observable.matrix
    elif kind == "symbolic":
        form = (
            0.25
            + sum(0.5 * X(qubit) for qubit in range(nqubits))
            + sum(Z(qubit) * Z(qubit + 1) for qubit in range(nqubits - 1))
            + 0.3 * Y(0) * Z(nqubits - 1)
        )
        observable = SymbolicHamiltonian(form, nqubits, backend=backend)
        matrix = observable.matrix
    elif kind in ("list", "tuple"):
        observable = [
            ("X" + "I" * (nqubits - 1), 0.5),
            ("Z" * nqubits, -0.3),
            ("I" * nqubits, 0.25),
            ("Y" + "Z" * (nqubits - 1), 0.4j),
        ]
        paulis = {
            "I": backend.matrices.I(2),
            "X": backend.matrices.X,
            "Y": backend.matrices.Y,
            "Z": backend.matrices.Z,
        }
        matrix = sum(
            coefficient * reduce(backend.kron, [paulis[letter] for letter in string])
            for string, coefficient in observable
        )
        if kind == "tuple":
            observable = tuple(observable)

    with caplog.at_level(logging.WARNING):
        circuit, alpha = hadamard_test(
            state_1, state_2, observable, imaginary, measure, backend=backend
        )

    ket = backend.execute_circuit(state_1).state()
    if matrix is not None:
        ket = backend.matmul(matrix, ket)
    amplitude = backend.sum(
        backend.conj(backend.execute_circuit(state_2).state()) * ket
    )
    target = float(amplitude.imag if imaginary else amplitude.real)

    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    pauli_sum = kind not in ("none", "circuit")
    assert (circuit.nqubits > nqubits + 1) == pauli_sum
    assert (alpha != 1.0) == pauli_sum
    assert len(circuit.measurements) == int(measure)
    assert ("inefficient" in caplog.text) == (kind == "dense")
    backend.assert_allclose(alpha * (2 * float(probability) - 1), target, atol=1e-10)


def test_hadamard_test_controlled_gates(backend):
    """Gates that are already controlled are supported, and so are empty circuits."""
    state_1 = Circuit(2)
    state_2 = Circuit(2)
    state_2.add(gates.H(0))
    state_2.add(gates.CRY(0, 1, 0.7))
    observable = Circuit(2)
    observable.add(gates.CNOT(0, 1))
    observable.add(gates.Z(0))

    circuit, _ = hadamard_test(state_1, state_2, observable, backend=backend)
    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    ket = backend.matmul(
        observable.unitary(backend), backend.execute_circuit(state_1).state()
    )
    bra = backend.conj(backend.execute_circuit(state_2).state())
    target = float(backend.sum(bra * ket).real)

    backend.assert_allclose(2 * float(probability) - 1, target, atol=1e-10)


def test_hadamard_test_non_pauli_symbols(backend, caplog):
    """A symbolic Hamiltonian with other symbols is converted from its matrix."""
    state_1 = _random_circuit(2, 1, backend)
    state_2 = _random_circuit(2, 2, backend)
    projector = backend.cast([[1.0, 0.0], [0.0, 0.0]], dtype="complex128")
    observable = SymbolicHamiltonian(
        Symbol(0, projector, name="P") * Z(1) + X(0), 2, backend=backend
    )

    with caplog.at_level(logging.WARNING):
        circuit, alpha = hadamard_test(state_1, state_2, observable, backend=backend)
    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    ket = backend.matmul(observable.matrix, backend.execute_circuit(state_1).state())
    bra = backend.conj(backend.execute_circuit(state_2).state())
    target = float(backend.sum(bra * ket).real)

    assert "inefficient" in caplog.text
    backend.assert_allclose(alpha * (2 * float(probability) - 1), target, atol=1e-10)


def test_hadamard_test_density_matrix(backend):
    """Keyword arguments are passed to the circuit."""
    state_1 = Circuit(1)
    state_1.add(gates.X(0))
    state_2 = Circuit(1)

    circuit, alpha = hadamard_test(
        state_1, state_2, density_matrix=True, backend=backend
    )
    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    assert circuit.density_matrix
    assert alpha == 1.0
    backend.assert_allclose(float(probability), 0.5, atol=1e-10)


def test_hadamard_test_errors(backend):
    """The inputs are validated."""
    circuit = Circuit(1)
    measured = Circuit(1)
    measured.add(gates.M(0))
    hamiltonian = hamiltonians.TFIM(2, h=1.0, dense=False, backend=backend)

    with pytest.raises(TypeError):
        hadamard_test(circuit, "state", backend=backend)
    with pytest.raises(TypeError):
        hadamard_test(circuit, circuit, "observable", backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, measured, backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, Circuit(2), backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, circuit, Circuit(2), backend=backend)
    with pytest.raises(ValueError):
        hadamard_test(circuit, circuit, hamiltonian, backend=backend)


@pytest.mark.parametrize("observable", [["X"], [("X", 1.0, 2.0)], [(1, 1.0)], [None]])
def test_hadamard_test_pauli_terms_type_errors(backend, observable):
    """The elements of a list of Pauli strings must be tuples of a string and a number."""
    with pytest.raises(TypeError):
        hadamard_test(Circuit(1), Circuit(1), observable, backend=backend)


@pytest.mark.parametrize("observable", [[("XX", 1.0)], [("A", 1.0)], [("x", 1.0)], []])
def test_hadamard_test_pauli_terms_value_errors(backend, observable):
    """The Pauli strings must have one letter of IXYZ per qubit, and there must be terms."""
    with pytest.raises(ValueError):
        hadamard_test(Circuit(1), Circuit(1), observable, backend=backend)


@pytest.mark.parametrize("measure", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_swap_test(backend, nqubits, measure):
    """The probability of the auxiliary qubit gives the squared overlap."""
    state_1 = _random_circuit(nqubits, 4, backend)
    state_2 = _random_circuit(nqubits, 5, backend)

    circuit = swap_test(state_1, state_2, measure)

    overlap = backend.sum(
        backend.conj(backend.execute_circuit(state_2).state())
        * backend.execute_circuit(state_1).state()
    )
    probability = backend.execute_circuit(circuit).probabilities(qubits=[0])[0]

    assert circuit.nqubits == 2 * nqubits + 1
    assert len(circuit.measurements) == int(measure)
    backend.assert_allclose(
        2 * float(probability) - 1, float(backend.abs(overlap) ** 2), atol=1e-10
    )


def test_swap_test_identical_and_orthogonal(backend):
    """The auxiliary qubit is always in zero for equal states and uniform if orthogonal."""
    zero = Circuit(1)
    one = Circuit(1)
    one.add(gates.X(0))

    equal = backend.execute_circuit(swap_test(one, one)).probabilities(qubits=[0])
    orthogonal = backend.execute_circuit(swap_test(zero, one)).probabilities(qubits=[0])

    backend.assert_allclose(equal, [1.0, 0.0], atol=1e-10)
    backend.assert_allclose(orthogonal, [0.5, 0.5], atol=1e-10)


def test_swap_test_density_matrix():
    """Keyword arguments are passed to the circuit."""
    circuit = swap_test(Circuit(1), Circuit(1), density_matrix=True)

    assert circuit.density_matrix


def test_swap_test_errors():
    """The inputs are validated."""
    measured = Circuit(1)
    measured.add(gates.M(0))

    with pytest.raises(TypeError):
        swap_test(Circuit(1), "state")
    with pytest.raises(ValueError):
        swap_test(Circuit(1), measured)
    with pytest.raises(ValueError):
        swap_test(Circuit(1), Circuit(2))
