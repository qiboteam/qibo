"""Tests for the Trotter product formula in `qibo/models/algorithms/trotter.py`."""

import itertools
import logging
from functools import reduce

import pytest

from qibo import Circuit, hamiltonians
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.hamiltonians._models import _multikron
from qibo.models import Trotter
from qibo.symbols import Z


def _pauli(string, backend):
    paulis = {
        "I": backend.matrices.I(2),
        "X": backend.matrices.X,
        "Y": backend.matrices.Y,
        "Z": backend.matrices.Z,
    }

    return _multikron([paulis[letter] for letter in string], backend)


def _exact(matrix, time, backend):
    eigenvalues, eigenvectors = backend.eigh(matrix)

    return backend.matrix_exp(matrix, -1j * time, eigenvectors, eigenvalues)


def _term(string, coefficient, backend):
    # Pauli matrices square to the identity
    identity = backend.matrices.I(2 ** len(string))

    return backend.cos(coefficient) * identity - 1j * backend.sin(coefficient) * _pauli(
        string, backend
    )


def _formula(terms, tau, order, backend):
    """Matrix of the symmetric product formula of an even order, which applies the first term
    first."""
    if order == 2:
        halves = [_term(string, c * tau / 2, backend) for string, c in terms]
        forward = reduce(lambda a, b: b @ a, halves)
        backward = reduce(lambda a, b: b @ a, halves[::-1])

        return backward @ forward

    p = 1 / (4 - 4 ** (1 / (order - 1)))
    short = _formula(terms, p * tau, order - 2, backend)
    long = _formula(terms, (1 - 4 * p) * tau, order - 2, backend)

    return short @ short @ long @ short @ short


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("coefficient", [0.7, -0.4])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_trotter_single_pauli(backend, nqubits, coefficient, order):
    """With one term, the circuit is exact for every Pauli string."""
    time = 1.3

    for string in map("".join, itertools.product("IXYZ", repeat=nqubits)):
        circuit = Trotter([(string, coefficient)], time, 3, order, backend=backend)

        backend.assert_allclose(
            circuit.unitary(backend),
            _term(string, time * coefficient, backend),
            atol=1e-10,
        )


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("nsteps", [1, 4])
@pytest.mark.parametrize("kind", ["list", "tuple", "dense", "symbolic"])
def test_trotter_commuting(backend, kind, nsteps, order, caplog):
    """Terms that commute have no error, with the identity as a global phase."""
    terms = [("ZZ", 0.7), ("ZI", -0.4), ("IZ", 0.2), ("II", 0.3)]
    time = 0.9
    matrix = sum(c * _pauli(string, backend) for string, c in terms)

    hamiltonian = {
        "list": terms,
        "tuple": tuple(terms),
        "dense": Hamiltonian(2, matrix, backend=backend),
        "symbolic": SymbolicHamiltonian(
            0.7 * Z(0) * Z(1) - 0.4 * Z(0) + 0.2 * Z(1) + 0.3, backend=backend
        ),
    }[kind]

    with caplog.at_level(logging.WARNING):
        circuit = Trotter(hamiltonian, time, nsteps, order, backend=backend)

    assert circuit.nqubits == 2
    assert ("inefficient" in caplog.text) == (kind == "dense")
    backend.assert_allclose(
        circuit.unitary(backend), _exact(matrix, time, backend), atol=1e-10
    )


def test_trotter_order_of_the_terms(backend):
    """The circuit is the product of the steps, where the first term is the rightmost factor."""
    terms = [("XX", 0.5), ("ZI", -0.3), ("IY", 0.4), ("YZ", -0.1)]
    time, nsteps = 0.8, 3

    circuit = Trotter(terms, time, nsteps, backend=backend)

    step = reduce(
        lambda a, b: b @ a,
        [_term(string, c * time / nsteps, backend) for string, c in terms],
    )
    expected = reduce(backend.matmul, nsteps * [step])

    backend.assert_allclose(circuit.unitary(backend), expected, atol=1e-10)
    # the reversed order of the terms is a different circuit
    reverse = Trotter(terms[::-1], time, nsteps, backend=backend)
    assert (
        backend.matrix_norm(circuit.unitary(backend) - reverse.unitary(backend)) > 1e-3
    )


@pytest.mark.parametrize("order", [2, 4, 6])
def test_trotter_suzuki(backend, order):
    """The circuit is the recursive product formula, whose merged rotations are exact."""
    terms = [("XX", 0.5), ("ZI", -0.3), ("IY", 0.4), ("YZ", -0.1)]
    time, nsteps = 0.8, 2

    circuit = Trotter(terms, time, nsteps, order, backend=backend)

    step = _formula(terms, time / nsteps, order, backend)
    expected = reduce(backend.matmul, nsteps * [step])

    backend.assert_allclose(circuit.unitary(backend), expected, atol=1e-10)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_trotter_symmetry(backend, order):
    """The even orders are the inverse of the formula of the opposite time, and the first is not."""
    terms = [("XX", 0.5), ("ZI", -0.3), ("IY", 0.4), ("YZ", -0.1)]

    forward = Trotter(terms, 0.8, 3, order, backend=backend)
    backward = Trotter(terms, -0.8, 3, order, backend=backend)
    product = backend.matmul(forward.unitary(backend), backward.unitary(backend))

    distance = backend.matrix_norm(product - backend.matrices.I(4), order=2)
    assert (distance < 1e-10) == (order > 1)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_trotter_error(backend, order):
    """The error decreases as the number of steps to the power of the order."""
    terms = [("XI", 0.9), ("ZZ", -0.6), ("IX", 0.5), ("YZ", 0.3)]
    time = 1.1
    matrix = sum(c * _pauli(string, backend) for string, c in terms)
    exact = _exact(matrix, time, backend)

    # sum of the norms of the commutators of the pairs of terms
    commutators = 0.0
    for j, (first, c) in enumerate(terms):
        for second, d in terms[j + 1 :]:
            a, b = c * _pauli(first, backend), d * _pauli(second, backend)
            commutators += float(
                backend.matrix_norm(
                    backend.matmul(a, b) - backend.matmul(b, a), order=2
                )
            )

    # the errors of the highest orders are not below the precision of the numbers
    steps = {
        1: (1, 2, 4, 32, 64),
        2: (1, 2, 4, 32, 64),
        4: (1, 2, 4, 8, 16),
        6: (1, 2, 4, 8),
    }
    errors = []
    for nsteps in steps[order]:
        circuit = Trotter(terms, time, nsteps, order, backend=backend)
        errors.append(
            float(backend.matrix_norm(circuit.unitary(backend) - exact, order=2))
        )

        if order == 1:
            assert errors[-1] <= time**2 / (2 * nsteps) * commutators * (1 + 1e-9)

    assert errors == sorted(errors, reverse=True)
    assert 0.8 * 2**-order < errors[-1] / errors[-2] < 1.2 * 2**-order


@pytest.mark.parametrize("order", [1, 2])
def test_trotter_gates(backend, order):
    """The terms with a zero coefficient have no gates, and no gate is shared."""
    terms = [("ZI", 0.0), ("IZ", 0.5), ("ZZ", 0.2)]

    circuit = Trotter(terms, 1.0, 5, order, backend=backend)

    # a Z rotation is one gate, and the ZZ one has two CNOT gates. In the second order, the first
    # term is applied before and after the second one, and the ones of the steps are merged,
    # which gives six rotations of the first term and five of the second one.
    assert len(circuit.queue) == {1: 5 * (1 + 3), 2: 6 * 1 + 5 * 3}[order]
    # no gate is shared between the steps, or inside of a rotation
    assert len({id(gate) for gate in circuit.queue}) == len(circuit.queue)


@pytest.mark.parametrize("nsteps", [1, 3])
@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_trotter_rotations(backend, order, nsteps):
    """The adjacent rotations of the same Pauli string are merged."""
    terms = [("ZII", 0.5), ("IZI", -0.3), ("IIZ", 0.2)]

    circuit = Trotter(terms, 1.0, nsteps, order, backend=backend)

    # a Z rotation is one gate, and a step of the order 2k has at most 2 (L - 1) 5^(k - 1) + 1
    # rotations, where the last of a step is merged to the first of the next one
    if order == 1:
        expected = 3 * nsteps
    else:
        expected = nsteps * (2 * 2 * 5 ** (order // 2 - 1) + 1) - (nsteps - 1)

    assert len(circuit.queue) == expected


def test_trotter_hamiltonians(backend):
    """The Hamiltonians of the models are written in the Pauli basis."""
    hamiltonian = hamiltonians.TFIM(2, h=0.7, dense=False, backend=backend)
    time = 0.6

    circuit = Trotter(hamiltonian, time, 40, backend=backend)

    backend.assert_allclose(
        circuit.unitary(backend),
        _exact(hamiltonian.matrix, time, backend),
        atol=2e-2,
    )


def test_trotter_density_matrix(backend):
    """Keyword arguments are passed to the circuit."""
    circuit = Trotter([("Z", 1.0)], 1.0, density_matrix=True, backend=backend)

    assert circuit.density_matrix
    assert isinstance(circuit, Circuit)


def test_trotter_errors(backend):
    terms = [("ZI", 1.0), ("IX", 0.5)]

    for order in (0, -1, 1.5, True, None):
        with pytest.raises(ValueError, match="``order`` must be a positive integer"):
            Trotter(terms, 1.0, order=order, backend=backend)
    for order in (3, 5, 7):
        with pytest.raises(
            ValueError, match="``order`` must be one or an even integer"
        ):
            Trotter(terms, 1.0, order=order, backend=backend)
    for order in (1, 2, 4, 6, 8):
        assert isinstance(Trotter(terms, 1.0, order=order, backend=backend), Circuit)
    for nsteps in (0, -2, 2.5, True):
        with pytest.raises(ValueError, match="``nsteps`` must be a positive integer"):
            Trotter(terms, 1.0, nsteps, backend=backend)
    with pytest.raises(TypeError, match="hamiltonian must be a list"):
        Trotter("ZI", 1.0, backend=backend)
    with pytest.raises(ValueError, match="hamiltonian must have at least one"):
        Trotter([("ZI", 0.0)], 1.0, backend=backend)
    with pytest.raises(ValueError, match="coefficients of hamiltonian must be real"):
        Trotter([("ZI", 1.0), ("IX", 0.5j)], 1.0, backend=backend)
