import pathlib
import pytest
import numpy as np
from qibo import callbacks, hamiltonians, models
from scipy.linalg import expm

REGRESSION_FOLDER = pathlib.Path(__file__).with_name('regressions')


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


def assert_regression_fixture(array, filename):
    """Check array matches data inside filename.

    Args:
        array: numpy array/
        filename: fixture filename

    If filename does not exists, this function
    creates the missing file otherwise it loads
    from file and compare.
    """
    def load(filename):
        return np.loadtxt(filename)
    try:
        array_fixture = load(filename)
    except: # pragma: no cover
        # case is not tested in GitHub workflows because files exist
        np.savetxt(filename, array)
        array_fixture = load(filename)
    np.testing.assert_allclose(array, array_fixture, rtol=1e-5)


def test_initial_state():
    """Test that adiabatic evolution initial state is the ground state of H0."""
    h0 = hamiltonians.X(3)
    h1 = hamiltonians.TFIM(3)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)
    target_psi = np.ones(8) / np.sqrt(8)
    init_psi = adev._cast_initial_state()
    assert_states_equal(init_psi, target_psi)


@pytest.mark.parametrize(("solver", "atol", "normalize"), [("exp", 0, False),
                                              ("rk4", 1e-2, True),
                                              ("rk45", 1e-2, True)])
def test_state_evolution(solver, atol, normalize):
    """Check state evolution under H = Z1 + Z2."""
    # Analytical solution
    t = np.linspace(0, 1, 1001)
    phase = np.exp(2j * t)[:, np.newaxis]
    ones = np.ones((1001, 2))
    target_psi = np.concatenate([phase, ones, phase.conj()], axis=1)

    dt = t[1] - t[0]
    checker = TimeStepChecker(target_psi, atol=atol)
    evolution = models.StateEvolution(hamiltonians.Z(2), dt=dt, solver=solver,
                                      callbacks=[checker])
    final_psi = evolution(final_time=1, initial_state=target_psi[0], normalize=normalize)


def test_state_evolution_final_state():
    """Check time-independent Hamiltonian state evolution without giving dt."""
    evolution = models.StateEvolution(hamiltonians.Z(2), dt=1)
    # Analytical solution
    phase = np.exp(2j)
    initial_psi = np.ones(4) / 2
    target_psi = np.array([phase, 1, 1, phase.conj()])
    final_psi = evolution(final_time=1, initial_state=initial_psi)
    assert_states_equal(final_psi, target_psi)


@pytest.mark.parametrize("t", [0, 0.3, 0.7, 1.0])
def test_hamiltonian_t(t):
    """Test adiabatic evolution hamiltonian as a function of time."""
    h0 = hamiltonians.X(2)
    h1 = hamiltonians.TFIM(2)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)

    m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1],
                   [1, 0, 0, 1], [0, 1, 1, 0]])
    m2 = np.diag([2, -2, -2, 2])
    ham = lambda t: - (1 - t) * m1 - t * m2

    adev.set_hamiltonian(final_time=1)
    matrix = adev.solver.hamiltonian(t).matrix
    np.testing.assert_allclose(matrix, ham(t))


@pytest.mark.parametrize("dt", [1e-1, 1e-2])
def test_adiabatic_evolution(dt):
    """Test adiabatic evolution with exponential solver."""
    h0 = hamiltonians.X(2)
    h1 = hamiltonians.TFIM(2)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt)

    m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1],
                   [1, 0, 0, 1], [0, 1, 1, 0]])
    m2 = np.diag([2, -2, -2, 2])
    ham = lambda t: - (1 - t) * m1 - t * m2

    target_psi = np.ones(4) / 2
    nsteps = int(1 / dt)
    for n in range(nsteps):
        target_psi = expm(-1j * dt * ham(n * dt)).dot(target_psi)
    final_psi = adev(final_time=1)
    assert_states_equal(final_psi, target_psi)


def test_state_evolution_errors():
    """Test ``ValueError``s for ``StateEvolution`` model."""
    ham = hamiltonians.Z(2)
    evolution = models.StateEvolution(ham, dt=1)
    # execute without initial state
    with pytest.raises(ValueError):
        final_state = evolution(final_time=1)
    # dt < 0
    with pytest.raises(ValueError):
        adev = models.StateEvolution(ham, dt=-1e-2)


def test_adiabatic_evolution_errors():
    """Test errors of ``AdiabaticEvolution`` model."""
    # Hamiltonians of bad type
    h0 = hamiltonians.X(3)
    s = lambda t: t
    with pytest.raises(TypeError):
        adev = models.AdiabaticEvolution(h0, lambda t: h0, s, dt=1e-2)
    h1 = hamiltonians.TFIM(2)
    with pytest.raises(TypeError):
        adev = models.AdiabaticEvolution(lambda t: h1, h1, s, dt=1e-2)
    # Hamiltonians with different number of qubits
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, s, dt=1e-2)
    # s with three arguments
    h0 = hamiltonians.X(2)
    s = lambda t, a, b: t + a + b
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, s, dt=1e-2)
    # s(0) != 0
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, lambda t: t + 1, dt=1e-2)
    # s(T) != 0
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, lambda t: t / 2, dt=1e-2)
    # Non-zero ``start_time``
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)
    with pytest.raises(NotImplementedError):
        final_state = adev(final_time=2, start_time=1)
    # execute without specifying variational parameters
    sp = lambda t, p: (1 - p) * np.sqrt(t) + p * t
    adevp = models.AdiabaticEvolution(h0, h1, sp, dt=1e-1)
    with pytest.raises(ValueError):
        final_state = adevp(final_time=1)


def test_energy_callback(dt=1e-2):
    """Test using energy callback in adiabatic evolution."""
    h0 = hamiltonians.X(2)
    h1 = hamiltonians.TFIM(2)
    energy = callbacks.Energy(h1)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt,
                                     callbacks=[energy])
    final_psi = adev(final_time=1)

    target_psi = np.ones(4) / 2
    calc_energy = lambda psi: psi.conj().dot(h1.matrix.numpy().dot(psi))
    target_energies = [calc_energy(target_psi)]
    ham = lambda t: h0 * (1 - t) + h1 * t
    for n in range(int(1 / dt)):
        prop = ham(n * dt).exp(-1j * dt).numpy()
        target_psi = prop.dot(target_psi)
        target_energies.append(calc_energy(target_psi))

    assert_states_equal(final_psi, target_psi)
    np.testing.assert_allclose(energy[:], target_energies, atol=1e-10)


def test_rk4_evolution(dt=1e-3):
    """Test adiabatic evolution with Runge-Kutta solver."""
    h0 = hamiltonians.X(3)
    h1 = hamiltonians.TFIM(3)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt, solver="rk4")

    target_psi = [np.ones(8) / np.sqrt(8)]
    ham = lambda t: h0 * (1 - t) + h1 * t
    for n in range(int(1 / dt)):
        prop = ham(n * dt).exp(-1j * dt).numpy()
        target_psi.append(prop.dot(target_psi[-1]))

    checker = TimeStepChecker(target_psi, atol=dt)
    adev.callbacks = [checker]
    final_psi = adev(final_time=1, initial_state=target_psi[0])


def test_set_scheduling_parameters():
    h0 = hamiltonians.X(3)
    h1 = hamiltonians.TFIM(3)
    sp = lambda t, p: (1 - p[0]) * np.sqrt(t) + p[0] * t
    adevp = models.AdiabaticEvolution(h0, h1, sp, 1e-2)
    adevp.set_parameters([0.5, 1])
    final_psi = adevp(final_time=1)

    s = lambda t: 0.5 * np.sqrt(t) + 0.5 * t
    adev = models.AdiabaticEvolution(h0, h1, s, 1e-2)
    target_psi = adev(final_time=1)
    np.testing.assert_allclose(final_psi, target_psi)


test_names = "method,options,messages,filename"
test_values = [("BFGS", {'maxiter': 1}, True, "adiabatic_bfgs.out"),
               ("sgd", {"nepochs": 5}, False, None)]
@pytest.mark.parametrize(test_names, test_values)
def test_scheduling_optimization(method, options, messages, filename):
    h0 = hamiltonians.X(3)
    h1 = hamiltonians.TFIM(3)
    sp = lambda t, p: (1 - p) * np.sqrt(t) + p * t
    adevp = models.AdiabaticEvolution(h0, h1, sp, dt=1e-1)
    best, params = adevp.minimize([0.5, 1], method=method, options=options,
                                  messages=messages)
    if filename is not None:
        assert_regression_fixture(params, REGRESSION_FOLDER/filename)
