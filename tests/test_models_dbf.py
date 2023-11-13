"""Testing DoubleBracketFlow model"""
import numpy as np
import pytest

from qibo.backends import GlobalBackend
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


def test_energy_fluctuations(backend):
    h0 = np.array([[1, 0], [0, -1]])
    state = np.array([1, 0])
    dbf = DoubleBracketFlow(Hamiltonian(1, matrix=h0, backend=backend))
    energy_fluctuation = dbf.energy_fluctuation(state=state)
    assert energy_fluctuation == 0
