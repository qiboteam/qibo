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
SEED = 10
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_canonical(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
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
    h0 = random_hermitian(2**nqubits, backend=backend, seed=SEED)
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


@pytest.mark.parametrize("nqubits", [3])
def test_double_bracket_iteration_eval_dbr_unitary(backend, nqubits):
    r"""The bound is $$||e^{-[D,H]}-GC||\le s^{3/2}(||[H,[D,H]||+||[D,[D,H]]||$$"""
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.group_commutator,
    )

    for s in np.linspace(0.001, 0.01, NSTEPS):
        u = dbi.eval_dbr_unitary(
            s, d=d, mode=DoubleBracketGeneratorType.single_commutator
        )
        v = dbi.eval_dbr_unitary(
            s, d=d, mode=DoubleBracketGeneratorType.group_commutator
        )

        assert np.linalg.norm(u - v) < 10 * s**1.49 * (
            np.linalg.norm(h0) + np.linalg.norm(d)
        ) * np.linalg.norm(h0) * np.linalg.norm(d)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_single_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
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
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(Hamiltonian(nqubits, h0, backend=backend))

    # find initial best step with look_ahead = 1
    initial_step = 0.01
    delta = 0.02

    step = dbi.hyperopt_step(
        step_min=initial_step - delta, step_max=initial_step + delta, max_evals=10
    )

    assert step != initial_step

    # evolve following the optimized first step
    for generator in DoubleBracketGeneratorType:
        dbi(mode=generator, step=step, d=d)

    # find the following step size with look_ahead
    look_ahead = 3

    step = dbi.hyperopt_step(
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

    assert energy_fluctuation == 0
