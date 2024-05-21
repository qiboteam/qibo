"""Testing DoubleBracketIteration strategies"""

import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketCostFunction,
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    DoubleBracketScheduling,
)
from qibo.models.dbi.utils import *
from qibo.models.dbi.utils_strategies import (
    gradient_descent_pauli,
    select_best_dbr_generator,
)
from qibo.quantum_info import random_hermitian

NSTEPS = 1
seed = 5
"""Number of steps for evolution."""


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
