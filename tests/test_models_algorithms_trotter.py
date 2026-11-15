"""Tests for the product formulas in `qibo/models/algorithms/trotter.py`."""

import itertools
import logging
import math
from collections import Counter
from functools import reduce

import pytest

from qibo import Circuit, hamiltonians
from qibo.hamiltonians import Hamiltonian, SymbolicHamiltonian
from qibo.hamiltonians._models import _multikron
from qibo.models import RandomizedTrotter, Trotter
from qibo.quantum_info import trace_distance
from qibo.symbols import Z

TERMS = [("XX", 1.0), ("ZI", 0.8), ("IZ", -0.6), ("YY", 0.4)]


def test_randomized_trotter_average(backend):
    """The average of the forward and the reverse first-order formulas is closer to the exact
    evolution than each one of them, for this Hamiltonian and time."""
    time = 0.2
    matrix = sum(c * _pauli(string, backend) for string, c in TERMS)
    exact = _exact(matrix, time, backend)

    forward = _formula(TERMS, time, 1, backend)
    reverse = _formula(TERMS[::-1], time, 1, backend)

    def distance(matrix):
        return float(backend.matrix_norm(matrix - exact, order=2))

    assert distance((forward + reverse) / 2) < 0.3 * min(
        distance(forward), distance(reverse)
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("nsteps", [1, 2])
def test_randomized_trotter_average_channel(backend, nsteps, order):
    """The density matrix averaged over many seeded random circuits approaches the one of the
    ensemble channel of the segments, which is closer to the exact one than the density matrix
    of the deterministic formula of the same order, for this seeded instance."""
    time, nsamples = 0.6, 200
    rho = _density_matrix(backend)
    matrix = sum(c * _pauli(string, backend) for string, c in TERMS)
    exact = _evolve(_exact(matrix, time, backend), rho, backend)

    def error(state):
        return float(backend.matrix_norm(state - exact, order="fro"))

    deterministic = error(
        _evolve(
            Trotter(TERMS, time, nsteps, order, backend=backend).unitary(backend),
            rho,
            backend,
        )
    )

    # average of the evolved density matrices of the circuits of the public function
    average = (
        sum(
            _evolve(
                RandomizedTrotter(
                    TERMS, time, nsteps, order, seed=seed, backend=backend
                ).unitary(backend),
                rho,
                backend,
            )
            for seed in range(nsamples)
        )
        / nsamples
    )

    # exact ensemble channel of a segment, which averages over the forward and the reverse
    # orders (first order) or over all of the permutations of the terms (even orders), applied
    # to the density matrix once for each segment
    unitaries = _segment_unitaries(TERMS, order, time, nsteps, backend)
    ensemble = rho
    for _ in range(nsteps):
        ensemble = sum(_evolve(unitary, ensemble, backend) for unitary in unitaries)
        ensemble = ensemble / len(unitaries)

    assert error(ensemble) <= deterministic
    assert error(average) <= deterministic
    # the margins are large: the errors of the ensemble are less than 0.55 of the deterministic
    assert error(ensemble) < 0.6 * deterministic
    assert error(average) < 0.6 * deterministic
    assert float(backend.matrix_norm(average - ensemble, order="fro")) < 0.05


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("nsteps", [1, 3])
def test_randomized_trotter_commuting(backend, nsteps, order):
    """The terms commute, so the order of the terms does not matter."""
    terms = [("ZZI", 0.5), ("IZZ", -0.3), ("ZIZ", 0.7), ("ZII", 0.2)]
    time = 0.9

    circuit = RandomizedTrotter(
        terms, time, nsteps=nsteps, order=order, seed=2, backend=backend
    )

    matrix = sum(c * _pauli(string, backend) for string, c in terms)
    backend.assert_allclose(
        circuit.unitary(backend), _exact(matrix, time, backend), atol=1e-10
    )


@pytest.mark.parametrize("order", [1, 2, 4])
def test_randomized_trotter_error(backend, order):
    """The error vanishes as the number of steps grows, with the order of the formula."""
    time = 1.0
    matrix = sum(c * _pauli(string, backend) for string, c in TERMS)
    exact = _exact(matrix, time, backend)

    errors = []
    for nsteps in (2, 8, 32):
        circuit = RandomizedTrotter(
            TERMS, time, nsteps=nsteps, order=order, seed=3, backend=backend
        )
        errors.append(
            float(backend.matrix_norm(circuit.unitary(backend) - exact, order=2))
        )

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < {1: 5e-2, 2: 5e-3, 4: 1e-6}[order]


def test_randomized_trotter_errors(backend):
    terms = [("ZI", 1.0), ("IX", 0.5)]

    for order in (0, -2, 3, 2.0 + 0.5, True):
        with pytest.raises(ValueError):
            RandomizedTrotter(terms, 1.0, order=order, backend=backend)
    for nsteps in (0, -2, 2.5, True):
        with pytest.raises(ValueError):
            RandomizedTrotter(terms, 1.0, nsteps=nsteps, backend=backend)
    with pytest.raises(TypeError, match="hamiltonian must be a list"):
        RandomizedTrotter("ZI", 1.0, backend=backend)
    with pytest.raises(TypeError, match="the elements of hamiltonian"):
        RandomizedTrotter([("ZI",)], 1.0, backend=backend)
    with pytest.raises(ValueError):
        RandomizedTrotter([("ZI", 1.0), ("X", 0.5)], 1.0, backend=backend)
    with pytest.raises(ValueError):
        RandomizedTrotter([("ZA", 1.0)], 1.0, backend=backend)
    with pytest.raises(ValueError):
        RandomizedTrotter([("ZI", 1.0j)], 1.0, backend=backend)
    # all of the coefficients are zero, as in Trotter
    for hamiltonian in ([("ZI", 0.0)], [("ZI", 0.0), ("IX", 0.0)]):
        with pytest.raises(ValueError, match="non-zero coefficient"):
            RandomizedTrotter(hamiltonian, 1.0, backend=backend)
        with pytest.raises(ValueError, match="non-zero coefficient"):
            Trotter(hamiltonian, 1.0, backend=backend)


def test_randomized_trotter_first_order_forward_reverse(backend):
    """The first-order segment is the forward or the reverse order, with probability 1/2, and
    never another permutation of the terms."""
    terms = [("ZII", 1.0), ("IZI", 1.0), ("IIZ", 1.0)]
    generator = backend.default_rng(4)
    nruns = 600

    counts = {}
    for _ in range(nruns):
        circuit = RandomizedTrotter(terms, 1.0, seed=generator, backend=backend)
        order = tuple(gate.target_qubits[0] for gate in circuit.queue)
        counts[order] = counts.get(order, 0) + 1

    assert set(counts) == {(0, 1, 2), (2, 1, 0)}
    assert all(abs(count / nruns - 1 / 2) < 0.08 for count in counts.values())


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_randomized_trotter_gate_counts(backend, order, seed):
    """The circuit has the same number of gates of each type as the one of Trotter. The Pauli
    strings are all different and have the same cost, which is a pair of X gates, so that
    there is no merging of rotations in a single step, whose pattern only depends on the
    positions of the terms, and the gates do not depend on the permutation."""
    terms = [
        (pauli, 0.5 + 0.1 * index)
        for index, pauli in enumerate(["XXII", "IXXI", "IIXX", "XIIX", "XIXI", "IXIX"])
    ]

    randomized = RandomizedTrotter(
        terms, 0.7, nsteps=1, order=order, seed=seed, backend=backend
    )
    deterministic = Trotter(terms, 0.7, nsteps=1, order=order, backend=backend)

    assert _gate_counts(randomized) == _gate_counts(deterministic)
    assert len(randomized.queue) > 0


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("nsteps", [3, 6])
def test_randomized_trotter_gate_counts_steps(backend, order, nsteps):
    """With several steps, only the merging of rotations at the boundaries of the steps differs,
    by at most one rotation per boundary."""
    terms = [("XXI", 1.0), ("ZII", 0.8), ("IZY", -0.6), ("YYZ", 0.4)]

    deterministic = Trotter(terms, 0.7, nsteps, order, backend=backend)
    for seed in range(5):
        randomized = RandomizedTrotter(
            terms, 0.7, nsteps, order, seed=seed, backend=backend
        )

        # a merged pair of rotations of the same string has one RZ gate less
        nrz = sum(gate.name == "rz" for gate in randomized.queue)
        reference = sum(gate.name == "rz" for gate in deterministic.queue)
        assert abs(nrz - reference) <= nsteps - 1
        assert randomized.nqubits == deterministic.nqubits


def test_randomized_trotter_hamiltonians(backend, caplog):
    """The Hamiltonians are written as Pauli strings, with a warning if it is not direct."""
    time, coefficient = 0.7, 0.6
    matrix = coefficient * _pauli("ZZ", backend)
    expected = _exact(matrix, time, backend)

    for hamiltonian in (
        [("ZZ", coefficient)],
        (("ZZ", coefficient),),
        SymbolicHamiltonian(coefficient * Z(0) * Z(1), backend=backend),
    ):
        circuit = RandomizedTrotter(hamiltonian, time, 2, seed=1, backend=backend)
        backend.assert_allclose(circuit.unitary(backend), expected, atol=1e-10)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        circuit = RandomizedTrotter(
            Hamiltonian(2, matrix, backend=backend), time, 2, seed=1, backend=backend
        )
    assert "inefficient" in caplog.text
    backend.assert_allclose(circuit.unitary(backend), expected, atol=1e-10)


def test_randomized_trotter_permutations(backend):
    """The permutation of the second order is any of them, with the same probability."""
    terms = [("ZII", 1.0), ("IZI", 1.0), ("IIZ", 1.0)]
    generator = backend.default_rng(4)
    nruns = 1200

    counts = {permutation: 0 for permutation in itertools.permutations(range(3))}
    for _ in range(nruns):
        circuit = RandomizedTrotter(
            terms, 1.0, order=2, seed=generator, backend=backend
        )
        # the rotations of the terms are applied in the order sigma, then reversed
        gates_order = tuple(gate.target_qubits[0] for gate in circuit.queue)
        assert gates_order == gates_order[::-1]
        counts[gates_order[:3]] += 1

    assert all(abs(count / nruns - 1 / 6) < 0.04 for count in counts.values())


@pytest.mark.parametrize("order", [1, 2, 4])
def test_randomized_trotter_permuted_formula(backend, order):
    """Each segment is the formula with the terms in the order of its own random draw: forward
    or reverse for the first order, and a permutation, the same in the whole formula, for the
    even orders."""
    time, nsteps, seed = 0.8, 6, 11

    circuit = RandomizedTrotter(
        TERMS, time, nsteps=nsteps, order=order, seed=seed, backend=backend
    )

    # orders of the terms that the circuit draws from the seed, one for each segment
    generator = backend.default_rng(seed)
    if order == 1:
        bits = [
            int(
                backend.random_choice(2, size=1, seed=generator, dtype=backend.int64)[0]
            )
            for _ in range(nsteps)
        ]
        draws = [list(range(len(TERMS)))[:: -1 if bit else 1] for bit in bits]
    else:
        draws = [
            [
                int(index)
                for index in backend.random_choice(
                    len(TERMS),
                    size=len(TERMS),
                    replace=False,
                    seed=generator,
                    dtype=backend.int64,
                )
            ]
            for _ in range(nsteps)
        ]
    # the terms are not applied in the same order in every segment
    assert len({tuple(draw) for draw in draws}) > 1
    if order == 1:
        assert {tuple(draw) for draw in draws} == {(0, 1, 2, 3), (3, 2, 1, 0)}

    steps = [
        _formula([TERMS[index] for index in draw], time / nsteps, order, backend)
        for draw in draws
    ]
    backend.assert_allclose(
        circuit.unitary(backend), reduce(lambda a, b: b @ a, steps), atol=1e-10
    )


@pytest.mark.parametrize("order", [1, 2, 4])
def test_randomized_trotter_rotation_counts(backend, order):
    """The number of rotations, which is the one of RZ gates, does not depend on the permutation,
    even if the Pauli strings have different costs."""
    terms = [("XXI", 1.0), ("ZII", 0.8), ("IZY", -0.6), ("YYZ", 0.4), ("IIX", 0.3)]

    deterministic = Trotter(terms, 0.7, nsteps=1, order=order, backend=backend)
    for seed in range(5):
        randomized = RandomizedTrotter(
            terms, 0.7, nsteps=1, order=order, seed=seed, backend=backend
        )

        assert _gate_counts(randomized)[1]["rz"] == _gate_counts(deterministic)[1]["rz"]


def test_randomized_trotter_seed(backend):
    """An integer seed repeats the circuit, and a generator draws different ones."""

    def circuit_gates(circuit):
        return [(gate.name, gate.qubits, gate.parameters) for gate in circuit.queue]

    first = RandomizedTrotter(TERMS, 1.0, nsteps=6, order=2, seed=5, backend=backend)
    second = RandomizedTrotter(TERMS, 1.0, nsteps=6, order=2, seed=5, backend=backend)
    assert circuit_gates(first) == circuit_gates(second)
    backend.assert_allclose(first.unitary(backend), second.unitary(backend))

    generator = backend.default_rng(5)
    third = RandomizedTrotter(
        TERMS, 1.0, nsteps=6, order=2, seed=generator, backend=backend
    )
    fourth = RandomizedTrotter(
        TERMS, 1.0, nsteps=6, order=2, seed=generator, backend=backend
    )
    assert circuit_gates(third) == circuit_gates(first)
    assert circuit_gates(third) != circuit_gates(fourth)

    other = RandomizedTrotter(TERMS, 1.0, nsteps=6, order=2, seed=6, backend=backend)
    assert circuit_gates(other) != circuit_gates(first)


@pytest.mark.parametrize("order", [1, 2, 4])
@pytest.mark.parametrize("coefficient", [0.7, -0.4])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_randomized_trotter_single_pauli(backend, nqubits, coefficient, order):
    """With one term, there is only one permutation and each step is exact."""
    time = 1.3

    for string in map("".join, itertools.product("IXYZ", repeat=nqubits)):
        circuit = RandomizedTrotter(
            [(string, coefficient)],
            time,
            nsteps=3,
            order=order,
            seed=1,
            backend=backend,
        )

        backend.assert_allclose(
            circuit.unitary(backend),
            _term(string, time * coefficient, backend),
            atol=1e-10,
        )


@pytest.mark.parametrize(
    ("order", "terms", "time", "nsegments"),
    [
        (1, TERMS, 0.3, 2),
        (1, TERMS, 0.3, 3),
        (2, TERMS[:3], 0.1, 1),
        (2, TERMS[:3], 0.1, 2),
        (4, TERMS[:3], 0.03, 1),
    ],
)
def test_randomized_trotter_theorem_bound(backend, order, terms, time, nsegments):
    """The exact ensemble of all of the draws of the segments, whose circuits are the ones of
    the public function, is closer to the exact evolution than the bounds of Theorems 1 and 2.

    The bounds are on the diamond norm of the difference of the channels, which is at least the
    trace norm of the difference of the states that they give for any density matrix, and the
    trace distance is half of the trace norm. Therefore, the trace distance is at most half of
    the bound.
    """
    # Eq. (20) of Theorem 1 of Childs, Ostrander and Su, Quantum 3, 182 (2019), for the first
    # order, and Eq. (47) of Theorem 2 for the order 2k. The constant Lambda is the maximum of
    # the spectral norms of the terms, which is the maximum of the absolute values of the
    # coefficients as the Pauli strings have unit norm.
    nterms = len(terms)
    lambda_max = max(abs(coefficient) for _, coefficient in terms)
    x = lambda_max * time * nterms
    if order == 1:
        first = x**4 / nsegments**3 * math.exp(2 * x / nsegments)
        second = 2 * x**3 / (3 * nsegments**2) * math.exp(x / nsegments)
    else:
        k = order // 2
        scale = 2 * 5 ** (k - 1) * lambda_max * time
        first = (
            4
            * (scale * nterms) ** (4 * k + 2)
            / (math.factorial(2 * k + 1) ** 2 * nsegments ** (4 * k + 1))
            * math.exp(2 * scale * nterms / nsegments)
        )
        second = (
            2
            * scale ** (2 * k + 1)
            * nterms ** (2 * k)
            / (math.factorial(2 * k - 1) * nsegments ** (2 * k))
            * math.exp(scale * nterms / nsegments)
        )
    bound = first + second

    # the bound is not vacuous, as the diamond distance of two channels is at most 2
    assert bound < 2

    unitaries = _segment_unitaries(terms, order, time, nsegments, backend)
    sequences = [
        reduce(lambda a, b: b @ a, sequence)
        for sequence in itertools.product(unitaries, repeat=nsegments)
    ]
    matrix = sum(c * _pauli(string, backend) for string, c in terms)
    exact = _exact(matrix, time, backend)

    # valid density matrices: a pure state, a mixture with coherences and a pure entangled one
    zero = backend.cast([[1, 0, 0, 0]], dtype=backend.complex128)
    bell = backend.cast([[1, 0, 0, 1]], dtype=backend.complex128) / math.sqrt(2)
    states = [
        backend.matmul(backend.transpose(zero), backend.conj(zero)),
        _density_matrix(backend),
        backend.matmul(backend.transpose(bell), backend.conj(bell)),
    ]

    # the ensemble state is the average of the states of the circuits over all of the draws
    distances = [
        float(
            trace_distance(
                _evolve(exact, rho, backend),
                sum(_evolve(sequence, rho, backend) for sequence in sequences)
                / len(sequences),
                backend,
            )
        )
        for rho in states
    ]

    assert max(distances) > 0
    assert max(distances) <= bound / 2

    # the circuits of the public function are products of the unitaries of the draws, and
    # several of them are drawn with the seeds
    found = set()
    for seed in range(12):
        circuit = RandomizedTrotter(
            terms, time, nsegments, order, seed=seed, backend=backend
        )
        matrix = circuit.unitary(backend)
        matches = [
            index
            for index, sequence in enumerate(sequences)
            if float(backend.matrix_norm(matrix - sequence, order=2)) < 1e-9
        ]
        assert len(matches) >= 1
        found.add(matches[0])
    assert len(found) > 1


def test_randomized_trotter_zero_terms(backend):
    """The terms with a zero coefficient are not applied."""
    terms = [("XX", 1.0), ("ZI", 0.0), ("IZ", 0.5)]

    circuit = RandomizedTrotter(terms, 1.0, nsteps=2, order=2, seed=1, backend=backend)
    reference = RandomizedTrotter(
        [terms[0], terms[2]], 1.0, nsteps=2, order=2, seed=1, backend=backend
    )

    backend.assert_allclose(circuit.unitary(backend), reference.unitary(backend))


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


def _density_matrix(backend):
    # mixture of two states with complex amplitudes, so it has coherences
    psi = backend.cast([1, 0.5j, -0.3, 0.2 + 0.1j], dtype=backend.complex128)
    phi = backend.cast([0.1, 1, 0.4j, -0.6], dtype=backend.complex128)
    psi = psi / backend.sqrt(backend.sum(backend.abs(psi) ** 2))
    phi = phi / backend.sqrt(backend.sum(backend.abs(phi) ** 2))

    def projector(state):
        return backend.outer(state, backend.conj(state))

    return 0.7 * projector(psi) + 0.3 * projector(phi)


def _evolve(unitary, rho, backend):
    return backend.matmul(
        backend.matmul(unitary, rho), backend.conj(backend.transpose(unitary))
    )


def _exact(matrix, time, backend):
    eigenvalues, eigenvectors = backend.eigh(matrix)

    return backend.matrix_exp(matrix, -1j * time, eigenvectors, eigenvalues)


def _formula(terms, tau, order, backend):
    """Matrix of the product formula of a step, which applies the first term first."""
    if order == 1:
        return reduce(
            lambda a, b: b @ a, [_term(string, c * tau, backend) for string, c in terms]
        )

    if order == 2:
        halves = [_term(string, c * tau / 2, backend) for string, c in terms]
        forward = reduce(lambda a, b: b @ a, halves)
        backward = reduce(lambda a, b: b @ a, halves[::-1])

        return backward @ forward

    p = 1 / (4 - 4 ** (1 / (order - 1)))
    short = _formula(terms, p * tau, order - 2, backend)
    long = _formula(terms, (1 - 4 * p) * tau, order - 2, backend)

    return short @ short @ long @ short @ short


def _gate_counts(circuit):
    return len(circuit.queue), Counter(gate.name for gate in circuit.queue)


def _pauli(string, backend):
    paulis = {
        "I": backend.matrices.I(2),
        "X": backend.matrices.X,
        "Y": backend.matrices.Y,
        "Z": backend.matrices.Z,
    }

    return _multikron([paulis[letter] for letter in string], backend)


def _segment_unitaries(terms, order, time, nsegments, backend):
    """Unitaries of the segment for all of the random draws: the forward and the reverse orders
    for the first order, and all of the permutations of the terms for the even orders.
    """
    if order == 1:
        draws = [list(range(len(terms))), list(range(len(terms)))[::-1]]
    else:
        draws = [list(draw) for draw in itertools.permutations(range(len(terms)))]

    return [
        _formula([terms[index] for index in draw], time / nsegments, order, backend)
        for draw in draws
    ]


def _term(string, coefficient, backend):
    # Pauli matrices square to the identity
    identity = backend.matrices.I(2 ** len(string))

    return backend.cos(coefficient) * identity - 1j * backend.sin(coefficient) * _pauli(
        string, backend
    )
