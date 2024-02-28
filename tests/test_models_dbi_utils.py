""""Testing utils for DoubleBracketIteration model"""

import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from qibo.models.dbi.utils import *
from qibo.quantum_info import random_hermitian

NSTEPS = 5
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [2, 3])
def test_generate_Z_operators(nqubits):
    h0 = random_hermitian(2**nqubits)
    dbi = DoubleBracketIteration(Hamiltonian(nqubits=nqubits, matrix=h0))
    generate_Z = generate_Z_operators(nqubits)
    Z_ops = list(generate_Z.values())

    delta_h0 = dbi.diagonal_h_matrix
    dephasing_channel = (sum([Z_op @ h0 @ Z_op for Z_op in Z_ops]) + h0) / 2**nqubits
    norm_diff = np.linalg.norm(delta_h0 - dephasing_channel)

    assert norm_diff < 1e-3


@pytest.mark.parametrize("nqubits", [2, 3])
@pytest.mark.parametrize("step", [0.1, None])
def test_select_best_dbr_generator(backend, nqubits, step):
    h0 = random_hermitian(2**nqubits, seed=1, backend=backend)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    generate_Z = generate_Z_operators(nqubits)
    Z_ops = list(generate_Z.values())
    initial_off_diagonal_norm = dbi.off_diagonal_norm

    for _ in range(NSTEPS):
        dbi, idx, step_optimize, flip = select_best_dbr_generator(
            dbi, Z_ops, step=step, compare_canonical=True
        )

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
@pytest.mark.parametrize("d_option", ["delta_H", "min_max"])
def test_gradient_descent_onsite_Z(backend, nqubits, d_option):
    h0 = random_hermitian(2**nqubits, seed=1, backend=backend)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    initial_off_diagonal_norm = dbi.off_diagonal_norm
    onsite_Z_ops = generate_onsite_Z_ops(nqubits)
    if d_option == "delta_H":
        d_coef = onsite_Z_decomposition(dbi.h.matrix, onsite_Z_ops)
    if d_option == "min_max":
        d_min_max = diagonal_min_max(dbi.h.matrix)
        d_coef = onsite_Z_decomposition(d_min_max, onsite_Z_ops)
    d = sum([d_coef[i] * onsite_Z_ops[i] for i in range(nqubits)])
    iters = 15
    for _ in range(iters):
        # calculate elements of gradient descent
        s, d_coef, d = gradient_descent_onsite_Z(
            dbi, d_coef, d, onsite_Z_ops=onsite_Z_ops, max_evals=100
        )
        # double bracket rotation with the results
        dbi(step=s, d=d)
    # when onsite_Z_ops not given
    s, d_coef, d = gradient_descent_onsite_Z(dbi, d_coef, max_evals=100)
    dbi(step=s, d=d)
    assert initial_off_diagonal_norm > dbi.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_dGamma_di_onsite_Z(nqubits):
    h0 = random_hermitian(2**nqubits, seed=1)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0),
        mode=DoubleBracketGeneratorType.single_commutator,
    )
    onsite_Z_ops = generate_onsite_Z_ops(nqubits)
    d_coef = onsite_Z_decomposition(dbi.h.matrix, onsite_Z_ops)
    d = sum([d_coef[i] * onsite_Z_ops[i] for i in range(nqubits)])
    # provide onsite_Z_ops or not gives the same result
    dGamma_di_onsite_Z_with_Z_ops = dGamma_di_onsite_Z(dbi, 3, 1, d, onsite_Z_ops)
    assert (
        dGamma_di_onsite_Z_with_Z_ops[-1] == dGamma_di_onsite_Z(dbi, 3, 1, d)[-1]
    ).all()
