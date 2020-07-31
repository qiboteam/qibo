import pytest
import numpy as np
from qibo import hamiltonians, models
from scipy.linalg import expm


def assert_states_equal(state, target_state):
    """Asserts that two state vectors are equal up to a phase."""
    phase = state[0] / target_state[0]
    np.testing.assert_allclose(state, phase * target_state)


def test_adiabatic_evolution_initial_state():
    h0 = hamiltonians.OneBodyPauli(3)
    h1 = hamiltonians.Ising(3)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, 1)
    target_psi = np.ones(8) / np.sqrt(8)
    init_psi = adev._cast_initial_state()
    assert_states_equal(init_psi, target_psi)


@pytest.mark.parametrize("dt", [1e-2, 1e-3])
def test_adiabatic_evolution(dt):
    h0 = hamiltonians.OneBodyPauli(2)
    h1 = hamiltonians.Ising(2)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, 1)
    ham = lambda t: np.array([[t, 0, 1 - t, 0],
                              [0, -t, 0, 1 - t],
                              [1 - t, 0, -t, 0],
                              [0, 1 - t, 0, t]])
    target_psi = np.ones(4) / 2
    nsteps = int(1 / dt)
    for n in range(nsteps):
        target_psi = expm(-1j * dt * ham(n * dt)).dot(target_psi)
    final_psi = adev(dt)
    assert_states_equal(final_psi, target_psi)
