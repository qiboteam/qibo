""""Testing utils for DoubleBracketIteration model"""

import numpy as np
import pytest

from qibo import set_backend
from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
)
from qibo.models.dbi.utils import *
from qibo.quantum_info import random_hermitian

NSTEPS = 5
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_generate_Z_operators(backend, nqubits):
    h0 = random_hermitian(2**nqubits)
    dbi = DoubleBracketIteration(Hamiltonian(nqubits=nqubits, matrix=h0))
    generate_Z = generate_Z_operators(nqubits)
    Z_ops = list(generate_Z.values())

    delta_h0 = dbi.diagonal_h_matrix
    dephasing_channel = (sum([Z_op @ h0 @ Z_op for Z_op in Z_ops]) + h0) / 2**nqubits
    norm_diff = np.linalg.norm(delta_h0 - dephasing_channel)

    assert norm_diff < 1e-3


@pytest.mark.parametrize("nqubits", [3, 4, 5])
@pytest.mark.parametrize("step", [0.1, None])
def test_select_best_dbr_generator(backend, nqubits, step):
    h0 = random_hermitian(2**nqubits, seed=1, backend=backend)
    dbi = DoubleBracketIteration(
        Hamiltonian(nqubits, h0, backend=backend),
        mode=DoubleBracketGeneratorType.single_commutator,
        scheduling=DoubleBracketScheduling.grid_search,
    )
    generate_Z = generate_Z_operators(nqubits)
    Z_ops = list(generate_Z.values())
    initial_off_diagonal_norm = dbi.off_diagonal_norm

    for _ in range(NSTEPS):
        dbi, idx, step_optimize, flip = select_best_dbr_generator(
            dbi, Z_ops, step=step, compare_canonical=True
        )

    assert initial_off_diagonal_norm > dbi.off_diagonal_norm
