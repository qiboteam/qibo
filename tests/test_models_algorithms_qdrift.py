"""Tests for the qDRIFT protocol in `qibo/models/algorithms/qdrift.py`."""

import itertools
import logging
from functools import reduce

import pytest

from qibo import Circuit, hamiltonians
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.models import qdrift
from qibo.symbols import X, Z


def _pauli_matrix(string, backend):
    paulis = {
        "I": backend.matrices.I(2),
        "X": backend.matrices.X,
        "Y": backend.matrices.Y,
        "Z": backend.matrices.Z,
    }

    return reduce(backend.kron, [paulis[letter] for letter in string])


def _exponential(string, coefficient, backend):
    # Pauli matrices square to the identity
    matrix = _pauli_matrix(string, backend)
    identity = backend.matrices.I(2 ** len(string))

    return backend.cos(coefficient) * identity - 1j * backend.sin(coefficient) * matrix


@pytest.mark.parametrize("coefficient", [0.7, -0.4])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_qdrift_single_pauli(backend, nqubits, coefficient):
    """With one term, each step is exact for every Pauli string, and so is the circuit."""
    time = 1.3

    for string in map("".join, itertools.product("IXYZ", repeat=nqubits)):
        circuit = qdrift([(string, coefficient)], time, nsteps=4, backend=backend)

        backend.assert_allclose(
            circuit.unitary(backend),
            _exponential(string, time * coefficient, backend),
            atol=1e-10,
        )


@pytest.mark.parametrize("kind", ["list", "tuple", "dense", "symbolic"])
def test_qdrift_hamiltonians(backend, kind, caplog):
    """The Hamiltonians are written as Pauli strings, with a warning if it is not direct."""
    time, coefficient = 0.9, -0.6
    matrix = coefficient * _pauli_matrix("XZ", backend)

    hamiltonian = {
        "list": [("XZ", coefficient)],
        "tuple": (("XZ", coefficient),),
        "dense": Hamiltonian(2, matrix, backend=backend),
        "symbolic": SymbolicHamiltonian(coefficient * X(0) * Z(1), backend=backend),
    }[kind]

    with caplog.at_level(logging.WARNING):
        circuit = qdrift(hamiltonian, time, epsilon=0.1, backend=backend)

    assert circuit.nqubits == 2
    assert ("inefficient" in caplog.text) == (kind == "dense")
    backend.assert_allclose(
        circuit.unitary(backend),
        _exponential("XZ", time * coefficient, backend),
        atol=1e-10,
    )


def test_qdrift_sampled_steps(backend):
    """The circuit is the product of the Pauli rotations of the sampled terms."""
    terms = [("XX", 0.5), ("ZI", -0.3), ("IY", 0.4), ("II", 0.2), ("YZ", -0.1)]
    time, nsteps, seed = 0.8, 12, 3

    circuit = qdrift(terms, time, nsteps=nsteps, seed=seed, backend=backend)

    weights = [abs(coefficient) for _, coefficient in terms]
    norm = sum(weights)
    indices = backend.random_choice(
        len(terms),
        size=nsteps,
        p=backend.cast([w / norm for w in weights], dtype=backend.float64),
        seed=backend.default_rng(seed),
        dtype=backend.int64,
    )
    steps = [
        _exponential(
            terms[int(index)][0],
            norm * time / nsteps * (1 if terms[int(index)][1] > 0 else -1),
            backend,
        )
        for index in indices
    ]

    backend.assert_allclose(
        circuit.unitary(backend), reduce(lambda a, b: b @ a, steps), atol=1e-10
    )


def test_qdrift_probabilities(backend):
    """The terms are drawn with probabilities proportional to their strengths."""
    terms = [("ZII", 1.0), ("IZI", -2.0), ("IIZ", 3.0)]
    time, nsteps = 0.5, 6000

    circuit = qdrift(terms, time, nsteps=nsteps, seed=1, backend=backend)

    # each step is a single rotation, with the same strength for every term
    assert len(circuit.queue) == nsteps
    assert all(gate.name == "rz" for gate in circuit.queue)
    angle = 2 * 6.0 * time / nsteps
    for gate in circuit.queue:
        qubit = gate.target_qubits[0]
        sign = 1 if qubit != 1 else -1
        backend.assert_allclose(gate.parameters[0], sign * angle, atol=1e-12)

    for qubit, weight in enumerate([1.0, 2.0, 3.0]):
        frequency = (
            sum(gate.target_qubits == (qubit,) for gate in circuit.queue) / nsteps
        )
        assert abs(frequency - weight / 6.0) < 0.03


def test_qdrift_nsteps(backend):
    """The number of steps follows the precision."""
    terms = [("ZI", 1.0), ("IZ", 0.5)]

    # 2 * (1.5 * 1) ** 2 / 0.4 = 11.25
    assert len(qdrift(terms, 1.0, epsilon=0.4, backend=backend).queue) == 12
    # at least one step
    assert len(qdrift(terms, 0.0, epsilon=0.4, backend=backend).queue) == 1
    assert len(qdrift(terms, 1.0, nsteps=7, backend=backend).queue) == 7


def test_qdrift_seed(backend):
    """An integer seed repeats the circuit, and a generator draws different ones."""
    terms = [("ZII", 1.0), ("IZI", 1.0), ("IIZ", 1.0)]

    def targets(circuit):
        return [gate.target_qubits for gate in circuit.queue]

    first = qdrift(terms, 1.0, nsteps=30, seed=5, backend=backend)
    second = qdrift(terms, 1.0, nsteps=30, seed=5, backend=backend)
    assert targets(first) == targets(second)

    generator = backend.default_rng(5)
    third = qdrift(terms, 1.0, nsteps=30, seed=generator, backend=backend)
    fourth = qdrift(terms, 1.0, nsteps=30, seed=generator, backend=backend)
    assert targets(third) == targets(first)
    assert targets(third) != targets(fourth)


def test_qdrift_average(backend):
    """The average of the random circuits approximates the exact evolution."""
    hamiltonian = hamiltonians.TFIM(2, h=0.7, dense=False, backend=backend)
    time, epsilon, nruns = 0.5, 0.05, 300

    exact = backend.matmul(
        hamiltonian.dense.exp(time), backend.execute_circuit(Circuit(2)).state()
    )
    observable = hamiltonians.Z(2, backend=backend)

    generator = backend.default_rng(2)
    values = []
    for _ in range(nruns):
        circuit = qdrift(hamiltonian, time, epsilon, seed=generator, backend=backend)
        values.append(
            observable.expectation_from_state(backend.execute_circuit(circuit).state())
        )

    assert abs(sum(values) / nruns - observable.expectation_from_state(exact)) < 0.15


def test_qdrift_kwargs(backend):
    circuit = qdrift([("Z", 1.0)], 1.0, nsteps=2, density_matrix=True, backend=backend)

    assert circuit.density_matrix


def test_qdrift_errors(backend):
    terms = [("ZI", 1.0), ("IX", 0.5)]

    with pytest.raises(ValueError):
        qdrift(terms, 1.0, backend=backend)
    with pytest.raises(ValueError):
        qdrift(terms, 1.0, epsilon=0.1, nsteps=3, backend=backend)
    for epsilon in (0.0, -0.1):
        with pytest.raises(ValueError):
            qdrift(terms, 1.0, epsilon=epsilon, backend=backend)
    for nsteps in (0, -2, 2.5, True):
        with pytest.raises(ValueError):
            qdrift(terms, 1.0, nsteps=nsteps, backend=backend)
    with pytest.raises(TypeError):
        qdrift("ZI", 1.0, nsteps=2, backend=backend)
    with pytest.raises(TypeError):
        qdrift([("ZI",)], 1.0, nsteps=2, backend=backend)
    with pytest.raises(ValueError):
        qdrift([("ZI", 1.0), ("X", 0.5)], 1.0, nsteps=2, backend=backend)
    with pytest.raises(ValueError):
        qdrift([("ZA", 1.0)], 1.0, nsteps=2, backend=backend)
    with pytest.raises(ValueError):
        qdrift([("ZI", 0.0)], 1.0, nsteps=2, backend=backend)
    with pytest.raises(ValueError):
        qdrift([("ZI", 1.0), ("IX", 0.5j)], 1.0, nsteps=2, backend=backend)
