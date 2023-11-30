"""Testing DoubleBracketIteration model"""
import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from qibo.quantum_info import random_hermitian

NSTEPS = 50
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_iteration_canonical(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    dbf = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.canonical,
    )
    initial_off_diagonal_norm = dbf.off_diagonal_norm
    for _ in range(NSTEPS):
        dbf(step=np.sqrt(0.001))

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_iteration_group_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbf = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.group_commutator,
    )
    initial_off_diagonal_norm = dbf.off_diagonal_norm

    with pytest.raises(ValueError):
        dbf(mode=DoubleBracketGeneratorType.group_commutator, step=0.01)

    for _ in range(NSTEPS):
        dbf(step=0.01, d=d)

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_iteration_single_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbf = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    initial_off_diagonal_norm = dbf.off_diagonal_norm

    with pytest.raises(ValueError):
        dbf(mode=DoubleBracketGeneratorType.single_commutator, step=0.01)

    for _ in range(NSTEPS):
        dbf(step=0.01, d=d)

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_hyperopt_step(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbf = DoubleBracketIteration(Hamiltonian(nqubits, h0, backend=backend))

    # find initial best step with look_ahead = 1
    initial_step = 0.01
    delta = 0.02

    step = dbf.hyperopt_step(
        step_min=initial_step - delta, step_max=initial_step + delta, max_evals=100
    )

    assert step != initial_step

    # evolve following the optimized first step
    for generator in DoubleBracketGeneratorType:
        dbf(mode=generator, step=step, d=d)

    # find the following step size with look_ahead
    look_ahead = 3

    step = dbf.hyperopt_step(
        step_min=initial_step - delta,
        step_max=initial_step + delta,
        max_evals=100,
        look_ahead=look_ahead,
    )

    # evolve following the optimized first step
    for gentype in range(look_ahead):
        dbf(mode=DoubleBracketGeneratorType(gentype + 1), step=step, d=d)


def test_energy_fluctuations(backend):
    h0 = np.array([[1, 0], [0, -1]])
    state = np.array([1, 0])
    dbf = DoubleBracketIteration(Hamiltonian(1, matrix=h0, backend=backend))
    energy_fluctuation = dbf.energy_fluctuation(state=state)
    assert energy_fluctuation == 0
