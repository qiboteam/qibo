"""Testing DoubleBracketIteration model"""

import numpy as np
import pytest

from qibo import hamiltonians, set_backend
from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketCostFunction,
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    DoubleBracketScheduling,
)
from qibo.models.dbi.utils import *
from qibo.models.dbi.utils_dbr_strategies import (
    gradient_descent,
    select_best_dbr_generator,
)
from qibo.models.dbi.utils_scheduling import polynomial_step
from qibo.quantum_info import random_hermitian

NSTEPS = 3
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
def test_double_bracket_iteration_group_commutator_3rd_order(backend, nqubits):
    """Check 3rd order group commutator mode."""
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.group_commutator_third_order,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm

    # test first iteration with default d
    dbi(mode=DoubleBracketGeneratorType.group_commutator_third_order, step=0.01)
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


@pytest.mark.parametrize("nqubits", [2, 3])
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
    h = 2

    # define the hamiltonian
    h0 = hamiltonians.TFIM(nqubits=nqubits, h=h)
    dbi = DoubleBracketIteration(h0, scheduling=scheduling)
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
    nqubits = 2
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


def test_polynomial_energy_fluctuation(backend):
    nqubits = 4
    h0 = random_hermitian(2**nqubits, seed=seed, backend=backend)
    state = np.zeros(2**nqubits)
    state[0] = 1
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        cost=DoubleBracketCostFunction.energy_fluctuation,
        scheduling=DoubleBracketScheduling.polynomial_approximation,
        ref_state=state,
    )
    for i in range(NSTEPS):
        s = dbi.choose_step(d=dbi.diagonal_h_matrix, n=5)
        dbi(step=s, d=dbi.diagonal_h_matrix)
    assert dbi.energy_fluctuation(state=state) < dbi.h0.energy_fluctuation(state=state)


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


@pytest.mark.parametrize("compare_canonical", [True, False])
@pytest.mark.parametrize("step", [None, 1e-3])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_select_best_dbr_generator(backend, nqubits, step, compare_canonical):
    h0 = random_hermitian(2**nqubits, backend=backend, seed=seed)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    generate_local_Z = generate_Z_operators(nqubits, backend=backend)
    Z_ops = list(generate_local_Z.values())
    for _ in range(NSTEPS):
        dbi, idx, step, flip_sign = select_best_dbr_generator(
            dbi,
            Z_ops,
            compare_canonical=compare_canonical,
            step=step,
        )
    assert dbi.off_diagonal_norm < initial_off_diagonal_norm


@pytest.mark.parametrize("step", [None, 1e-3])
def test_params_to_diagonal_operator(backend, step):
    nqubits = 2
    pauli_operator_dict = generate_pauli_operator_dict(
        nqubits, parameterization_order=1, backend=backend
    )
    params = [1, 2, 3]
    operator_pauli = sum(
        [params[i] * list(pauli_operator_dict.values())[i] for i in range(nqubits)]
    )
    backend.assert_allclose(
        operator_pauli,
        params_to_diagonal_operator(
            params,
            nqubits=nqubits,
            parameterization=ParameterizationTypes.pauli,
            pauli_operator_dict=pauli_operator_dict,
        ),
    )
    operator_element = params_to_diagonal_operator(
        params,
        nqubits=nqubits,
        parameterization=ParameterizationTypes.computational,
    )
    for i in range(len(params)):
        backend.assert_allclose(
            backend.cast(backend.to_numpy(operator_element).diagonal())[i], params[i]
        )


@pytest.mark.parametrize("order", [1, 2])
def test_gradient_descent(backend, order):
    nqubits = 2
    h0 = random_hermitian(2**nqubits, seed=seed, backend=backend)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        scheduling=DoubleBracketScheduling.hyperopt,
        cost=DoubleBracketCostFunction.off_diagonal_norm,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    pauli_operator_dict = generate_pauli_operator_dict(
        nqubits,
        parameterization_order=order,
        backend=backend,
    )
    pauli_operators = list(pauli_operator_dict.values())
    # let initial d be approximation of $\Delta(H)
    d_coef_pauli = decompose_into_pauli_basis(
        dbi.diagonal_h_matrix, pauli_operators=pauli_operators
    )
    d_pauli = sum([d_coef_pauli[i] * pauli_operators[i] for i in range(nqubits)])
    loss_hist_pauli, d_params_hist_pauli, s_hist_pauli = gradient_descent(
        dbi,
        NSTEPS,
        d_coef_pauli,
        ParameterizationTypes.pauli,
        pauli_operator_dict=pauli_operator_dict,
        pauli_parameterization_order=order,
    )
    assert loss_hist_pauli[-1] < initial_off_diagonal_norm

    # computational basis
    d_coef_computational_partial = backend.cast(backend.to_numpy(d_pauli).diagonal())
    (
        loss_hist_computational_partial,
        _,
        _,
    ) = gradient_descent(
        dbi, NSTEPS, d_coef_computational_partial, ParameterizationTypes.computational
    )
    assert loss_hist_computational_partial[-1] < initial_off_diagonal_norm
