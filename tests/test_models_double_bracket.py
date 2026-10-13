"""Tests for double-bracket iterations in `qibo/models/double_bracket.py`."""

import math
from itertools import pairwise

import pytest

from qibo import hamiltonians, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.double_bracket import DoubleBracketIteration
from qibo.quantum_info import (
    commutator,
    double_commutator,
    random_gaussian_matrix,
    random_hermitian,
)


def test_initial_norm_and_monotonicity(backend):
    """Greedy canonical steps decrease the off-diagonal norm monotonically."""
    nqubits = 3
    hamiltonian = hamiltonians.TFIM(
        nqubits, h=1.0, closed_boundary=False, backend=backend
    )
    dbi = DoubleBracketIteration(hamiltonian, backend=backend)

    # the only off-diagonal term of the TFIM is the transverse field
    off_diagonal = hamiltonians.X(nqubits, backend=backend).matrix
    expected = float(backend.matrix_norm(off_diagonal, order="fro"))
    assert math.isclose(dbi.off_diagonal_norm, expected, rel_tol=1e-10)

    norms = dbi.run(nsteps=10)

    assert len(norms) == 11
    assert all(later < earlier for earlier, later in pairwise(norms))
    assert norms[-1] < 0.9 * norms[0]


def test_random_hermitian(backend):
    """The iteration accepts raw matrices and decreases their off-diagonal norm."""
    matrix = random_hermitian(8, seed=1, backend=backend)
    dbi = DoubleBracketIteration(matrix, backend=backend)

    norms = dbi.run(nsteps=3)

    assert all(later < earlier for earlier, later in pairwise(norms))


def test_hamiltonian_input_and_unitarity(backend):
    """Hamiltonian objects are accepted and ``H_k`` is unitarily related to ``H_0``."""
    hamiltonian = hamiltonians.TFIM(3, h=0.5, backend=backend)
    dbi = DoubleBracketIteration(hamiltonian, backend=backend)
    dbi.run(nsteps=3)

    unitary = dbi.unitary
    backend.assert_allclose(
        backend.conj(unitary.T) @ unitary, backend.identity(8), atol=1e-10
    )
    backend.assert_allclose(
        backend.conj(unitary.T) @ hamiltonian.matrix @ unitary,
        dbi.hamiltonian,
        atol=1e-10,
    )


def test_short_step_derivative(backend):
    """Short canonical steps decrease the norm as ``-2 s ||[Delta(H), H]||^2``."""
    hamiltonian = hamiltonians.Ising(
        3, 1.0, (1.0, 0.7), closed_boundary=False, backend=backend
    )
    dbi = DoubleBracketIteration(hamiltonian, backend=backend)
    diagonal = backend.diag(backend.diag(hamiltonian.matrix))
    bracket = commutator(diagonal, hamiltonian.matrix)
    expected = -2 * float(backend.matrix_norm(bracket, order="fro")) ** 2

    step = 1e-5
    before = dbi.off_diagonal_norm**2
    dbi.step(step)
    after = dbi.off_diagonal_norm**2

    assert math.isclose((after - before) / step, expected, rel_tol=1e-3)


def test_heisenberg_equation(backend):
    """A short double-bracket rotation follows ``dH/ds = [[D, H], H]``."""
    hamiltonian = hamiltonians.Ising(
        3, 1.0, (1.0, 0.7), closed_boundary=False, backend=backend
    )
    dbi = DoubleBracketIteration(hamiltonian, backend=backend)
    diagonal = backend.diag(backend.diag(hamiltonian.matrix))
    derivative = double_commutator(diagonal, hamiltonian.matrix, nesting="left")

    step = 1e-6
    dbi.step(step)

    backend.assert_allclose(
        (dbi.hamiltonian - hamiltonian.matrix) / step, derivative, atol=1e-3
    )


@pytest.mark.parametrize("rotation", ["double_bracket", "group_commutator"])
def test_negative_step_flips_diagonal(backend, rotation):
    """A negative step duration is equivalent to flipping the sign of the diagonal."""
    hamiltonian = hamiltonians.Ising(
        3, 1.0, (0.3, 0.5), closed_boundary=False, backend=backend
    )
    diagonal = backend.diag(
        SymbolicHamiltonian(
            symbols.Z(0) * symbols.Z(2), nqubits=3, backend=backend
        ).matrix
    )

    first = DoubleBracketIteration(hamiltonian, backend=backend)
    second = DoubleBracketIteration(hamiltonian, backend=backend)
    first.step(-0.02, diagonal, rotation)
    second.step(0.02, -diagonal, rotation)

    backend.assert_allclose(first.hamiltonian, second.hamiltonian, atol=1e-10)


def test_variational_diagonals(backend):
    """Variational candidates can only improve on the canonical bracket step by step."""
    nqubits = 4
    hamiltonian = hamiltonians.TFIM(
        nqubits, h=1.0, closed_boundary=False, backend=backend
    )
    candidates = [
        SymbolicHamiltonian(
            symbols.Z(qubit) * symbols.Z(qubit + 1), nqubits=nqubits, backend=backend
        ).matrix
        for qubit in range(nqubits - 1)
    ]

    canonical = DoubleBracketIteration(hamiltonian, backend=backend)
    variational = DoubleBracketIteration(hamiltonian, backend=backend)

    canonical_norms = canonical.run(nsteps=1)
    variational_norms = variational.run(nsteps=1, diagonals=candidates)

    assert variational_norms[-1] <= canonical_norms[-1] + 1e-10
    assert variational_norms[-1] < variational_norms[0]


def test_diagonal_matrix_input(backend):
    """Diagonal operators can be given as vectors or as diagonal matrices."""
    hamiltonian = hamiltonians.TFIM(3, h=0.3, backend=backend)
    matrix = hamiltonians.Z(3, backend=backend).matrix

    first = DoubleBracketIteration(hamiltonian, backend=backend)
    second = DoubleBracketIteration(hamiltonian, backend=backend)
    first.step(0.05, matrix)
    second.step(0.05, backend.diag(matrix))

    backend.assert_allclose(first.hamiltonian, second.hamiltonian, atol=1e-10)


@pytest.mark.parametrize("nsteps", [1, 2, 3])
def test_circuit(backend, nsteps):
    """The circuit reproduces the unitary of the group commutator iteration."""
    hamiltonian = hamiltonians.Ising(
        3, 1.0, (1.0, 0.5), closed_boundary=False, backend=backend
    )
    candidate = SymbolicHamiltonian(symbols.Z(0), nqubits=3, backend=backend).matrix
    dbi = DoubleBracketIteration(hamiltonian, backend=backend)
    dbi.run(nsteps, diagonals=[candidate], rotation="group_commutator")

    circuit = dbi.circuit()
    unitary = backend.cast(circuit.unitary(backend=backend))

    assert circuit.nqubits == 3
    backend.assert_allclose(unitary, dbi.unitary, atol=1e-8)


def test_group_commutator_approximates_double_bracket(backend):
    """Group commutator rotations approximate double-bracket ones with error ``O(s^{3/2})``."""
    hamiltonian = hamiltonians.Ising(
        3, 1.0, (1.0, 0.5), closed_boundary=False, backend=backend
    )

    errors = []
    for step in (1e-3, 1e-4):
        exact = DoubleBracketIteration(hamiltonian, backend=backend)
        approximate = DoubleBracketIteration(hamiltonian, backend=backend)
        exact.step(step)
        approximate.step(step, rotation="group_commutator")
        errors.append(
            float(
                backend.matrix_norm(
                    exact.hamiltonian - approximate.hamiltonian, order=2
                )
            )
        )

    assert errors[1] < 1e-3
    assert errors[0] / errors[1] > 10


def test_energy_fluctuations(backend):
    """Fluctuations vanish for a diagonal Hamiltonian and decrease along the iteration."""
    diagonal = DoubleBracketIteration(
        hamiltonians.Z(2, backend=backend), backend=backend
    )
    backend.assert_allclose(diagonal.energy_fluctuations, backend.zeros(4), atol=1e-10)

    hamiltonian = hamiltonians.TFIM(3, h=1.0, closed_boundary=False, backend=backend)
    dbi = DoubleBracketIteration(hamiltonian, backend=backend)
    initial = dbi.energy_fluctuations
    assert all(float(fluctuation) > 0 for fluctuation in initial)

    dbi.run(nsteps=10)
    assert float(dbi.energy_fluctuations.min()) < float(initial.min())


def test_optimize_step_trivial(backend):
    """No step is needed if the Hamiltonian is already diagonal."""
    dbi = DoubleBracketIteration(hamiltonians.Z(2, backend=backend), backend=backend)

    assert dbi.optimize_step() == 0.0


def test_errors(backend):
    """Invalid inputs are rejected."""
    with pytest.raises(ValueError):
        DoubleBracketIteration(
            random_gaussian_matrix(4, seed=1, backend=backend), backend=backend
        )

    dbi = DoubleBracketIteration(
        hamiltonians.TFIM(2, h=1.0, backend=backend), backend=backend
    )

    with pytest.raises(ValueError):
        dbi.step(0.1, rotation="trotter")

    with pytest.raises(ValueError):
        dbi.step(0.1, diagonal=backend.ones(3))

    with pytest.raises(ValueError):
        dbi.step(0.1, diagonal=backend.ones((4, 4)))

    with pytest.raises(ValueError):
        dbi.step(0.1, diagonal=1j * backend.ones(4))

    dbi.step(0.1, rotation="double_bracket")
    with pytest.raises(ValueError):
        dbi.circuit()

    dbi = DoubleBracketIteration(
        random_hermitian(3, seed=1, backend=backend), backend=backend
    )
    with pytest.raises(ValueError):
        dbi.circuit()
