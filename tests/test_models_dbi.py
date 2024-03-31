"""Testing DoubleBracketIteration model"""

import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketCostFunction,
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    DoubleBracketScheduling,
)
from qibo.quantum_info import random_hermitian

NSTEPS = 1
seed = 10
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_canonical(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.canonical,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for _ in range(NSTEPS):
        dbi(step=np.sqrt(0.001))

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_group_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.group_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm

    # test first iteration with default d
    dbi(mode=DoubleBracketGeneratorType.group_commutator, step=0.01)
    for _ in range(NSTEPS):
        dbi(step=0.01, d=d)

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_single_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm

    # test first iteration with default d
    dbi(mode=DoubleBracketGeneratorType.single_commutator, step=0.01)

    for _ in range(NSTEPS):
        dbi(step=0.01, d=d)

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4])
def test_hyperopt_step(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(Hamiltonian(nqubits, h0, backend=backend))
    dbi.scheduling = DoubleBracketScheduling.hyperopt
    # find initial best step with look_ahead = 1
    initial_step = 0.01
    delta = 0.02
    step = dbi.choose_step(
        step_min=initial_step - delta, step_max=initial_step + delta, max_evals=100
    )

    assert step != initial_step

    # evolve following with optimized first step
    for generator in DoubleBracketGeneratorType:
        dbi(mode=generator, step=step, d=d)

    # find the following step size with look_ahead
    look_ahead = 3

    step = dbi.choose_step(
        step_min=initial_step - delta,
        step_max=initial_step + delta,
        max_evals=10,
        look_ahead=look_ahead,
    )

    # evolve following the optimized first step
    for gentype in range(look_ahead):
        dbi(mode=DoubleBracketGeneratorType(gentype + 1), step=step, d=d)


def test_energy_fluctuations(backend):
    h0 = np.array([[1, 0], [0, -1]])
    h0 = backend.cast(h0, dtype=backend.dtype)

    state = np.array([1, 0])
    state = backend.cast(state, dtype=backend.dtype)

    dbi = DoubleBracketIteration(Hamiltonian(1, matrix=h0, backend=backend))
    energy_fluctuation = dbi.energy_fluctuation(state=state)
    assert energy_fluctuation == 1.0


@pytest.mark.parametrize(
    "scheduling",
    [
        DoubleBracketScheduling.grid_search,
        DoubleBracketScheduling.hyperopt,
        DoubleBracketScheduling.simulated_annealing,
    ],
)
@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_iteration_scheduling_grid_hyperopt_annealing(
    backend, nqubits, scheduling
):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for _ in range(NSTEPS):
        step1 = dbi.choose_step(d=d, scheduling=scheduling)
        dbi(d=d, step=step1)
    step2 = dbi.choose_step()
    dbi(step=step2)
    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 6])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize(
    "cost",
    [
        DoubleBracketCostFunction.least_squares,
        DoubleBracketCostFunction.off_diagonal_norm,
    ],
)
def test_double_bracket_iteration_scheduling_polynomial(backend, nqubits, n, cost):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        scheduling=DoubleBracketScheduling.polynomial_approximation,
        cost=cost,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for _ in range(NSTEPS):
        step1 = dbi.choose_step(d=d, n=n)
        dbi(d=d, step=step1)
    assert initial_off_diagonal_norm > dbi.off_diagonal_norm
