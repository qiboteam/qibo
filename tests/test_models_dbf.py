"""Testing DoubleBracketFlow model"""
import hyperopt
import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.double_bracket import DoubleBracketFlow, FlowGeneratorType
from qibo.quantum_info import random_hermitian

NSTEPS = 50
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_flow_canonical(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    dbf = DoubleBracketFlow(Hamiltonian(nqubits, h0, backend=backend))
    initial_off_diagonal_norm = dbf.off_diagonal_norm
    for _ in range(NSTEPS):
        dbf(mode=FlowGeneratorType.canonical, step=np.sqrt(0.001))

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_flow_group_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbf = DoubleBracketFlow(Hamiltonian(nqubits, h0, backend=backend))
    initial_off_diagonal_norm = dbf.off_diagonal_norm

    with pytest.raises(ValueError):
        dbf(mode=FlowGeneratorType.group_commutator, step=0.01)

    for _ in range(NSTEPS):
        dbf(mode=FlowGeneratorType.group_commutator, step=0.01, d=d)

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_double_bracket_flow_single_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbf = DoubleBracketFlow(Hamiltonian(nqubits, h0, backend=backend))
    initial_off_diagonal_norm = dbf.off_diagonal_norm

    with pytest.raises(ValueError):
        dbf(mode=FlowGeneratorType.single_commutator, step=0.01)

    for _ in range(NSTEPS):
        dbf(mode=FlowGeneratorType.single_commutator, step=0.01, d=d)

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_hyperopt_step(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbf = DoubleBracketFlow(Hamiltonian(nqubits, h0, backend=backend))

    # find initial best step with look_ahead = 1
    initial_step = 0.01
    delta = 0.02

    step = dbf.hyperopt_step(
        step_min=initial_step - delta, step_max=initial_step + delta, max_evals=100
    )

    assert step != initial_step

    # evolve following the optimized first step
    for generator in FlowGeneratorType:
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
        dbf(mode=FlowGeneratorType(gentype + 1), step=step, d=d)
