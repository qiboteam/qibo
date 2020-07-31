import pytest
import numpy as np
from qibo import callbacks, hamiltonians, models
from scipy.linalg import expm


def assert_states_equal(state, target_state, atol=0):
    """Asserts that two state vectors are equal up to a phase."""
    phase = state[0] / target_state[0]
    np.testing.assert_allclose(state, phase * target_state, atol=atol)


class TimeStepChecker(callbacks.Callback):
    """Callback that checks each evolution time step."""

    def __init__(self, target_states, atol=0):
        super(TimeStepChecker, self).__init__()
        self.target_states = iter(target_states)
        self.atol = atol

    def __call__(self, state):
        assert_states_equal(state, next(self.target_states), atol=self.atol)


def test_initial_state():
    h0 = hamiltonians.OneBodyPauli(3)
    h1 = hamiltonians.TFIM(3)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t)
    target_psi = np.ones(8) / np.sqrt(8)
    init_psi = adev._cast_initial_state()
    assert_states_equal(init_psi, target_psi)


@pytest.mark.parametrize(("solver", "atol"), [("exp", 0),
                                              ("rk4", 1e-2)])
def test_state_evolution(solver, atol):
    """Check state evolution under H = Z1 + Z2."""
    ham = hamiltonians.OneBodyPauli(2, p=(0.0, 0.0, 1.0))
    evolution = models.StateEvolution(ham)
    # Analytical solution
    t = np.linspace(0, 1, 1001)
    phase = np.exp(2j * t)[:, np.newaxis]
    ones = np.ones((1001, 2))
    target_psi = np.concatenate([phase, ones, phase.conj()], axis=1)

    dt = t[1] - t[0]
    checker = TimeStepChecker(target_psi, atol=atol)
    final_psi = evolution(dt, 1, initial_state=target_psi[0], solver=solver,
                          callbacks=[checker])

@pytest.mark.parametrize("t", [0, 0.3, 0.7, 1.0])
def test_hamiltonian_t(t):
    h0 = hamiltonians.OneBodyPauli(2)
    h1 = hamiltonians.TFIM(2)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t)

    m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1],
                   [1, 0, 0, 1], [0, 1, 1, 0]])
    m2 = np.diag([2, -2, -2, 2])
    ham = lambda t: - (1 - t) * m1 - t * m2

    matrix = adev.hamiltonian(t).hamiltonian
    np.testing.assert_allclose(matrix, ham(t))


@pytest.mark.parametrize("dt", [1e-1, 1e-2])
def test_adiabatic_evolution(dt):
    h0 = hamiltonians.OneBodyPauli(2)
    h1 = hamiltonians.TFIM(2)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t)

    m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1],
                   [1, 0, 0, 1], [0, 1, 1, 0]])
    m2 = np.diag([2, -2, -2, 2])
    ham = lambda t: - (1 - t) * m1 - t * m2

    target_psi = np.ones(4) / 2
    nsteps = int(1 / dt)
    for n in range(nsteps):
        target_psi = expm(-1j * dt * ham(n * dt)).dot(target_psi)
    final_psi = adev(dt, 1)
    assert_states_equal(final_psi, target_psi)


def test_energy_callback(dt=1e-2):
    h0 = hamiltonians.OneBodyPauli(2)
    h1 = hamiltonians.TFIM(2)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t)
    energy = callbacks.Energy(h1)

    target_psi = np.ones(4) / 2
    calc_energy = lambda psi: psi.conj().dot(h1.hamiltonian.numpy().dot(psi))
    target_energies = [calc_energy(target_psi)]
    nsteps = int(1 / dt)
    for n in range(nsteps):
        prop = expm(-1j * dt * adev.hamiltonian(n * dt).hamiltonian.numpy())
        target_psi = prop.dot(target_psi)
        target_energies.append(calc_energy(target_psi))

    final_psi = adev(dt, 1, callbacks=[energy])
    assert_states_equal(final_psi, target_psi)
    np.testing.assert_allclose(energy[:], target_energies, atol=1e-10)


def test_rk4_evolution(dt=1e-3):
    h0 = hamiltonians.OneBodyPauli(3)
    h1 = hamiltonians.TFIM(3)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t)

    nsteps = int(1 / dt)
    target_psi = [np.ones(8) / np.sqrt(8)]
    for n in range(nsteps):
        prop = expm(-1j * dt * adev.hamiltonian(n * dt).hamiltonian.numpy())
        target_psi.append(prop.dot(target_psi[-1]))

    checker = TimeStepChecker(target_psi, atol=dt)
    final_psi = adev(dt, 1, solver="rk4", callbacks=[checker])
