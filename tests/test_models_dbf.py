"""Testing DoubleBracketFlow model"""
import numpy as np
import pytest

from qibo.hamiltonians import Hamiltonian
from qibo.models.double_bracket import DoubleBracketFlow, DoubleBracketFlowMode
from qibo.quantum_info import random_hermitian

NSTEPS = 100
"""Number of steps for evolution."""


@pytest.mark.parametrize("nqubits", [4, 5, 6])
def test_double_bracket_flow_canonical(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    dbf = DoubleBracketFlow(Hamiltonian(nqubits, h0, backend=backend))
    initial_off_diagonal_norm = dbf.off_diagonal_norm
    for _ in range(NSTEPS):
        dbf(mode=DoubleBracketFlowMode.canonical, step=np.sqrt(0.01))

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm


@pytest.mark.parametrize("nqubits", [4, 5, 6])
def test_double_bracket_flow_group_commutator(backend, nqubits):
    h0 = random_hermitian(2**nqubits, backend=backend)
    d = backend.cast(np.diag(np.diag(backend.to_numpy(h0))))
    dbf = DoubleBracketFlow(Hamiltonian(nqubits, h0, backend=backend))
    initial_off_diagonal_norm = dbf.off_diagonal_norm

    with pytest.raises(ValueError):
        dbf(mode=DoubleBracketFlowMode.group_commutator, step=0.1)

    for _ in range(NSTEPS):
        dbf(mode=DoubleBracketFlowMode.group_commutator, step=0.1, d=d)

    assert initial_off_diagonal_norm > dbf.off_diagonal_norm
