"""Testing DoubleBracketIteration model"""

import numpy as np
import pytest

from qibo import set_backend
from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketCostFunction,
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    DoubleBracketScheduling,
)
from qibo.models.dbi.utils import *
from qibo.models.dbi.utils_gradients import gradient_descent_dbr_d_ansatz
from qibo.models.dbi.utils_scheduling import polynomial_step
from qibo.models.dbi.utils_strategies import (
    gradient_descent_pauli,
    select_best_dbr_generator,
)
from qibo.quantum_info import random_hermitian

NSTEPS = 1
seed = 10
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [1, 2])
def test_double_bracket_iteration_canonical(backend, nqubits):
    """Check default (canonical) mode."""
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
    """Check group commutator mode."""
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
    """Check single commutator mode."""
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
@pytest.mark.parametrize(
    "scheduling",
    [
        DoubleBracketScheduling.grid_search,
        DoubleBracketScheduling.hyperopt,
        DoubleBracketScheduling.simulated_annealing,
    ],
)
def test_variational_scheduling(backend, nqubits, scheduling):
    """Check schduling options."""
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend), scheduling=scheduling
    )
    # find initial best step with look_ahead = 1
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for _ in range(NSTEPS):
        step = dbi.choose_step()
        dbi(step=step)
    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize(
    "cost",
    [
        DoubleBracketCostFunction.off_diagonal_norm,
        DoubleBracketCostFunction.least_squares,
    ],
)
def test_polynomial_cost_function(backend, cost):
    nqubits = 4
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        cost=cost,
        scheduling=DoubleBracketScheduling.polynomial_approximation,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    for i in range(NSTEPS):
        s = dbi.choose_step(d=dbi.diagonal_h_matrix, n=5)
        dbi(step=s, d=dbi.off_diag_h)
    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


def test_polynomial_energy_fluctuation():
    set_backend("numpy")
    nqubits = 4
    h0 = random_hermitian(2**nqubits, seed=seed)
    state = np.zeros(2**nqubits)
    state[3] = 1
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0),
        mode=DoubleBracketGeneratorType.single_commutator,
        cost=DoubleBracketCostFunction.energy_fluctuation,
        scheduling=DoubleBracketScheduling.polynomial_approximation,
        ref_state=state,
    )
    for i in range(NSTEPS):
        s = dbi.choose_step(d=dbi.diagonal_h_matrix, n=5)
        dbi(step=s, d=dbi.off_diag_h)
    assert dbi.energy_fluctuation(state=state) == 0.0


@pytest.mark.parametrize("nqubits", [5, 6])
def test_polynomial_fail_cases(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        scheduling=DoubleBracketScheduling.polynomial_approximation,
    )
    with pytest.raises(ValueError):
        polynomial_step(dbi, n=2, n_max=1)
    assert polynomial_step(dbi, n=1) == None


def test_energy_fluctuations(backend):
    """Check energy fluctuation cost function."""
    nqubits = 3
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(Hamiltonian(nqubits, h0, backend=backend))
    # define the state
    state = np.zeros(2**nqubits)
    state[3] = 1
    assert dbi.energy_fluctuation(state=state) < 1e-5


def test_least_squares(backend):
    """Check least squares cost function."""
    nqubits = 4
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        cost=DoubleBracketCostFunction.least_squares,
    )
    d = np.diag(np.linspace(1, 2**nqubits, 2**nqubits)) / 2**nqubits
    initial_potential = dbi.least_squares(d=d)
    step = dbi.choose_step(d=d)
    dbi(d=d, step=step)
    assert dbi.least_squares(d=d) < initial_potential


@pytest.mark.parametrize("nqubits", [2, 3])
def test_select_best_dbr_generator(backend, nqubits):
    scheduling = DoubleBracketScheduling.grid_search
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        scheduling=scheduling,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    generate_local_Z = generate_Z_operators(nqubits)
    Z_ops = list(generate_local_Z.values())
    for _ in range(NSTEPS):
        dbi, idx, step, flip_sign = select_best_dbr_generator(
            dbi, Z_ops, scheduling=scheduling, compare_canonical=True
        )
    assert dbi.off_diagonal_norm < initial_off_diagonal_norm


def test_params_to_diagonal_operator(backend):
    nqubits = 3
    pauli_operator_dict = generate_pauli_operator_dict(
        nqubits, parameterization_order=1
    )
    params = [1, 2, 3]
    operator_pauli = [
        params[i] * list(pauli_operator_dict.values())[i] for i in range(nqubits)
    ]
    assert (
        operator_pauli
        == params_to_diagonal_operator(
            params, nqubits=nqubits, parameterization=ParameterizationTypes.pauli
        )
    ).all()
    operator_element = params_to_diagonal_operator(
        params, nqubits=nqubits, parameterization=ParameterizationTypes.element
    )
    assert (operator_element.diag() == params).all()


@pytest.mark.parametrize("nqubits", [2, 3])
def test_gradient_descent_pauli(backend, nqubits):
    scheduling = DoubleBracketScheduling.grid_search
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        scheduling=scheduling,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    pauli_operator_dict = generate_pauli_operator_dict(
        nqubits=nqubits, parameterization_order=2
    )
    d_coef = decompose_into_Pauli_basis(
        dbi.h.matrix, list(pauli_operator_dict.values())
    )
    d = sum([d_coef[i] * list(pauli_operator_dict.values())[i] for i in range(nqubits)])
    step, d_coef, d = gradient_descent_pauli(dbi, d_coef, d)
    dbi(d=d, step=step)
    assert dbi.off_diagonal_norm < initial_off_diagonal_norm


@pytest.mark.parametrize("nqubits", [2, 3])
def test_gradient_descent_d_ansatz(backend, nqubits):
    scheduling = DoubleBracketScheduling.polynomial_approximation
    cost = DoubleBracketCostFunction.least_squares
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        cost=cost,
        scheduling=scheduling,
    )
    params = np.linspace(1, 2**nqubits, 2**nqubits)
    step = 1e-1

    d, loss, grad, diags = gradient_descent_dbr_d_ansatz(dbi, params, 25, step)

    assert loss[-1] < loss[0]
