import pytest
import numpy as np
from qibo import callbacks, hamiltonians, models
from qibo.config import raise_error
from scipy.linalg import expm


def assert_states_equal(state, target_state, atol=0):
    """Asserts that two state vectors are equal up to a phase."""
    phase = state[0] / target_state[0]
    np.testing.assert_allclose(state, phase * target_state, atol=atol)


class TimeStepChecker(callbacks.BackendCallback):
    """Callback that checks each evolution time step."""

    def __init__(self, target_states, atol=0):
        super(TimeStepChecker, self).__init__()
        self.target_states = iter(target_states)
        self.atol = atol

    def state_vector_call(self, state):
        assert_states_equal(state, next(self.target_states), atol=self.atol)

    def density_matrix_call(self, state): # pragma: no cover
        raise_error(NotImplementedError)


def test_initial_state():
    """Test that adiabatic evolution initial state is the ground state of H0."""
    h0 = hamiltonians.X(3)
    h1 = hamiltonians.TFIM(3)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)
    target_psi = np.ones(8) / np.sqrt(8)
    init_psi = adev.get_initial_state()
    assert_states_equal(init_psi, target_psi)


@pytest.mark.parametrize(("solver", "atol"), [("exp", 0),
                                              ("rk4", 1e-2),
                                              ("rk45", 1e-2)])
def test_state_evolution(solver, atol):
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
    final_psi = evolution(final_time=1, initial_state=target_psi[0])


def test_state_evolution_final_state():
    """Check time-independent Hamiltonian state evolution."""
    evolution = models.StateEvolution(hamiltonians.Z(2), dt=1)
    # Analytical solution
    phase = np.exp(2j)
    initial_psi = np.ones(4) / 2
    target_psi = np.array([phase, 1, 1, phase.conj()])
    final_psi = evolution(final_time=1, initial_state=initial_psi)
    assert_states_equal(final_psi, target_psi)


def test_state_time_dependent_evolution_final_state(nqubits=2, dt=1e-2):
    """Check time-dependent Hamiltonian state evolution."""
    ham = lambda t: np.cos(t) * hamiltonians.Z(nqubits)
    # Analytical solution
    target_psi = [np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)]
    for n in range(int(1 / dt)):
        prop = expm(-1j * dt * ham(n * dt).matrix.numpy())
        target_psi.append(prop.dot(target_psi[-1]))

    checker = TimeStepChecker(target_psi, atol=1e-8)
    evolution = models.StateEvolution(ham, dt=dt, callbacks=[checker])
    final_psi = evolution(final_time=1, initial_state=np.copy(target_psi[0]))


@pytest.mark.parametrize("nqubits,solver,dt",
                         [(3, "exp", 1e-3),
                          (4, "exp", 1e-3),
                          (4, "rk45", 1e-3)])
def test_trotterized_evolution(nqubits, solver, dt, accel=None, h=1.0):
    """Test state evolution using trotterization of ``TrotterHamiltonian``."""
    atol = 1e-4 if solver == "exp" else 1e-2
    target_psi = [np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)]
    ham_matrix = np.array(hamiltonians.TFIM(nqubits, h=h).matrix)
    prop = expm(-1j * dt * ham_matrix)
    for n in range(int(1 / dt)):
        target_psi.append(prop.dot(target_psi[-1]))

    ham = hamiltonians.TFIM(nqubits, h=h, trotter=True)
    checker = TimeStepChecker(target_psi, atol=atol)
    evolution = models.StateEvolution(ham, dt, solver=solver,
                                      callbacks=[checker],
                                      accelerators=accel)
    final_psi = evolution(final_time=1, initial_state=np.copy(target_psi[0]))

    # Change dt
    evolution = models.StateEvolution(ham, dt / 10, accelerators=accel)
    final_psi = evolution(final_time=1, initial_state=np.copy(target_psi[0]))
    assert_states_equal(final_psi, target_psi[-1], atol=atol)


def test_trotterized_evolution_distributed():
    import qibo
    if qibo.get_backend() != "custom": # pragma: no cover
        pytest.skip("Distributed circuit works only with custom backend.")
    test_trotterized_evolution(4, "exp", 1e-2, accel={"/GPU:0": 2})


def test_hamiltonian_t():
    """Test adiabatic evolution hamiltonian as a function of time."""
    h0 = hamiltonians.X(2)
    h1 = hamiltonians.TFIM(2)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)
    # try accessing hamiltonian before setting it
    with pytest.raises(RuntimeError):
        adev.hamiltonian()

    m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1],
                   [1, 0, 0, 1], [0, 1, 1, 0]])
    m2 = np.diag([2, -2, -2, 2])
    ham = lambda t, T: - (1 - t / T) * m1 - (t / T) * m2

    adev.set_hamiltonian(total_time=1)
    for t in [0, 0.3, 0.7, 1.0]:
        matrix = adev.hamiltonian(t).matrix
        np.testing.assert_allclose(matrix, ham(t, 1))
    #try using a different total time
    for t in [0, 0.3, 0.7, 1.0]:
        matrix = adev.hamiltonian(t, total_time=2).matrix
        np.testing.assert_allclose(matrix, ham(t, 2))


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
    # time-dependent Hamiltonian bad type
    with pytest.raises(TypeError):
        evol = models.StateEvolution(lambda t: "abc", dt=1e-2)
    # execute without initial state
    with pytest.raises(ValueError):
        final_state = evolution(final_time=1)
    # dt < 0
    with pytest.raises(ValueError):
        adev = models.StateEvolution(ham, dt=-1e-2)
    # pass accelerators without trotter Hamiltonian
    with pytest.raises(NotImplementedError):
        adev = models.StateEvolution(ham, dt=1e-2, accelerators={"/GPU:0": 2})


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


@pytest.mark.parametrize(("solver", "atol"),
                         [("exp", 1e-10),
                          ("rk4", 1e-2),
                          ("rk45", 1e-2)])
def test_energy_callback(solver, atol, dt=1e-2):
    """Test using energy callback in adiabatic evolution."""
    h0 = hamiltonians.X(2)
    h1 = hamiltonians.TFIM(2)
    energy = callbacks.Energy(h1)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt,
                                     callbacks=[energy], solver=solver)
    final_psi = adev(final_time=1)

    target_psi = np.ones(4) / 2
    calc_energy = lambda psi: psi.conj().dot(h1.matrix.numpy().dot(psi))
    target_energies = [calc_energy(target_psi)]
    ham = lambda t: h0 * (1 - t) + h1 * t
    for n in range(int(1 / dt)):
        prop = ham(n * dt).exp(dt).numpy()
        target_psi = prop.dot(target_psi)
        target_energies.append(calc_energy(target_psi))

    assert_states_equal(final_psi, target_psi, atol=atol)
    np.testing.assert_allclose(energy[:], target_energies, atol=atol)


@pytest.mark.parametrize("solver", ["rk4", "rk45"])
@pytest.mark.parametrize("trotter", [False, True])
def test_rk4_evolution(solver, trotter, dt=1e-3):
    """Test adiabatic evolution with Runge-Kutta solver."""
    h0 = hamiltonians.X(3, trotter=trotter)
    h1 = hamiltonians.TFIM(3, trotter=trotter)
    if trotter:
        h0 = h1.make_compatible(h0)

    target_psi = [np.ones(8) / np.sqrt(8)]
    ham = lambda t: h0 * (1 - t) + h1 * t
    for n in range(int(1 / dt)):
        prop = ham(n * dt).exp(dt).numpy()
        target_psi.append(prop.dot(target_psi[-1]))

    checker = TimeStepChecker(target_psi, atol=dt)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt, solver="rk4",
                                     callbacks=[checker])
    final_psi = adev(final_time=1, initial_state=np.copy(target_psi[0]))


@pytest.mark.parametrize("nqubits", [3, 4])
def test_trotter_hamiltonian_t(nqubits, h=1.0, dt=1e-3):
    """Test using ``TrotterHamiltonian`` in adiabatic evolution model."""
    dense_h0 = hamiltonians.X(nqubits)
    dense_h1 = hamiltonians.TFIM(nqubits, h=h)
    dense_adev = models.AdiabaticEvolution(dense_h0, dense_h1, lambda t: t, dt)

    local_h0 = hamiltonians.X(nqubits, trotter=True)
    local_h1 = hamiltonians.TFIM(nqubits, h=h, trotter=True)
    local_adev = models.AdiabaticEvolution(local_h0, local_h1, lambda t: t, dt)

    for t in np.random.random(10):
        local_matrix = local_adev.hamiltonian(t, total_time=1).dense.matrix
        target_matrix = dense_adev.hamiltonian(t, total_time=1).matrix
        np.testing.assert_allclose(local_matrix, target_matrix)


@pytest.mark.parametrize("nqubits,dt", [(3, 1e-3), (4, 1e-2)])
def test_trotterized_adiabatic_evolution(accelerators, nqubits, dt):
    """Test adiabatic evolution using trotterization of ``TrotterHamiltonian``."""
    dense_h0 = hamiltonians.X(nqubits)
    dense_h1 = hamiltonians.TFIM(nqubits)

    target_psi = [np.ones(2 ** nqubits) / np.sqrt(2 ** nqubits)]
    ham = lambda t: dense_h0 * (1 - t) + dense_h1 * t
    for n in range(int(1 / dt)):
        prop = ham(n * dt).exp(dt).numpy()
        target_psi.append(prop.dot(target_psi[-1]))

    local_h0 = hamiltonians.X(nqubits, trotter=True)
    local_h1 = hamiltonians.TFIM(nqubits, trotter=True)
    checker = TimeStepChecker(target_psi, atol=dt)
    adev = models.AdiabaticEvolution(local_h0, local_h1, lambda t: t, dt,
                                     callbacks=[checker],
                                     accelerators=accelerators)
    final_psi = adev(final_time=1)


def test_set_scheduling_parameters():
    """Test ``AdiabaticEvolution.set_parameters``."""
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


test_names = "method,options,messages,trotter,filename"
test_values = [
    ("BFGS", {'maxiter': 1}, True, False, "adiabatic_bfgs.out"),
    ("BFGS", {'maxiter': 1}, True, True, "trotter_adiabatic_bfgs.out"),
    ("sgd", {"nepochs": 5}, False, False, None)
    ]
@pytest.mark.parametrize(test_names, test_values)
def test_scheduling_optimization(method, options, messages, trotter, filename):
    """Test optimization of s(t)."""
    from qibo.tests_new.test_models_variational import assert_regression_fixture
    h0 = hamiltonians.X(3, trotter=trotter)
    h1 = hamiltonians.TFIM(3, trotter=trotter)
    sp = lambda t, p: (1 - p) * np.sqrt(t) + p * t
    adevp = models.AdiabaticEvolution(h0, h1, sp, dt=1e-1)
    best, params, _ = adevp.minimize([0.5, 1], method=method, options=options,
                                     messages=messages)
    if filename is not None:
        assert_regression_fixture(params, filename)
