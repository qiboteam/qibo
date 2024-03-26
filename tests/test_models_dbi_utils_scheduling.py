"""Unit testing for utils_scheduling.py for Double Bracket Iteration"""

import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.dbi.double_bracket import (
    DoubleBracketGeneratorType,
    DoubleBracketIteration,
    DoubleBracketScheduling,
)
from qibo.models.dbi.utils_scheduling import polynomial_step
from qibo.quantum_info import random_hermitian

NSTEPS = 1
seed = 10
"""Number of steps for evolution."""


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
