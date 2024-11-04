import numpy as np
import pytest
from scipy.linalg import expm

from qibo import callbacks, hamiltonians, models
from qibo.config import raise_error


def assert_states_equal(backend, state, target_state, atol=0):
    """Asserts that two state vectors are equal up to a phase."""
    state = backend.to_numpy(state)
    target_state = backend.to_numpy(target_state)
    phase = state[0] / target_state[0]
    backend.assert_allclose(state, phase * target_state, atol=atol)


class TimeStepChecker(callbacks.Callback):
    """Callback that checks each evolution time step."""

    def __init__(self, target_states, atol=0):
        super().__init__()
        self.target_states = iter(target_states)
        self.atol = atol

    def apply(self, backend, state):
        assert_states_equal(backend, state, next(self.target_states), atol=self.atol)

    def apply_density_matrix(self, backend, state):  # pragma: no cover
        raise_error(NotImplementedError)


def test_state_evolution_init(backend):
    ham = hamiltonians.Z(2, backend=backend)
    evolution = models.StateEvolution(ham, dt=1)
    assert evolution.nqubits == 2
    # time-dependent Hamiltonian bad type
    with pytest.raises(TypeError):
        evol = models.StateEvolution(lambda t: "abc", dt=1e-2)
    # dt < 0
    with pytest.raises(ValueError):
        adev = models.StateEvolution(ham, dt=-1e-2)
    # pass accelerators without trotter Hamiltonian
    with pytest.raises(NotImplementedError):
        adev = models.StateEvolution(ham, dt=1e-2, accelerators={"/GPU:0": 2})


def test_state_evolution_get_initial_state(backend):
    ham = hamiltonians.Z(2, backend=backend)
    evolution = models.StateEvolution(ham, dt=1)
    # execute without initial state
    with pytest.raises(ValueError):
        final_state = evolution(final_time=1)


@pytest.mark.parametrize(
    ("solver", "atol"), [("exp", 0), ("rk4", 1e-2), ("rk45", 1e-1)]
)
def test_state_evolution_constant_hamiltonian(backend, solver, atol):
    nsteps = 200
    t = np.linspace(0, 1, nsteps + 1)
    phase = np.exp(2j * t)[:, np.newaxis]
    ones = np.ones((nsteps + 1, 2))
    target_psi = np.concatenate([phase, ones, phase.conj()], axis=1)

    dt = t[1] - t[0]
    checker = TimeStepChecker(target_psi, atol=atol)
    ham = hamiltonians.Z(2, backend=backend)
    evolution = models.StateEvolution(ham, dt=dt, solver=solver, callbacks=[checker])
    final_psi = evolution(final_time=1, initial_state=target_psi[0])


@pytest.mark.parametrize("nqubits,dt", [(2, 1e-2)])
def test_state_evolution_time_dependent_hamiltonian(backend, nqubits, dt):
    ham = lambda t: np.cos(t) * hamiltonians.Z(nqubits, backend=backend)
    # Analytical solution
    target_psi = [np.ones(2**nqubits) / np.sqrt(2**nqubits)]
    for n in range(int(1 / dt)):
        prop = expm(-1j * dt * backend.to_numpy(ham(n * dt).matrix))
        target_psi.append(prop.dot(target_psi[-1]))

    checker = TimeStepChecker(target_psi, atol=1e-8)
    evolution = models.StateEvolution(ham, dt=dt, callbacks=[checker])
    final_psi = evolution(final_time=1, initial_state=np.copy(target_psi[0]))


@pytest.mark.parametrize("nqubits", [5])
@pytest.mark.parametrize("solver,dt,atol", [("exp", 1e-1, 1e-2), ("rk45", 1e-2, 1e-1)])
def test_state_evolution_trotter_hamiltonian(
    backend, accelerators, nqubits, solver, dt, atol
):
    if accelerators is not None and solver != "exp":  # pragma: no cover
        pytest.skip("Distributed evolution is supported only with exp solver.")
    h = 1.0

    target_psi = [np.ones(2**nqubits) / np.sqrt(2**nqubits)]
    ham_matrix = backend.to_numpy(
        hamiltonians.TFIM(nqubits, h=h, backend=backend).matrix
    )
    prop = expm(-1j * dt * ham_matrix)
    for n in range(int(1 / dt)):
        target_psi.append(prop.dot(target_psi[-1]))

    ham = hamiltonians.TFIM(nqubits, h=h, dense=False, backend=backend)
    checker = TimeStepChecker(target_psi, atol=atol)
    evolution = models.StateEvolution(
        ham, dt, solver=solver, callbacks=[checker], accelerators=accelerators
    )
    final_psi = evolution(final_time=1, initial_state=np.copy(target_psi[0]))

    # Change dt
    if solver == "exp":
        evolution = models.StateEvolution(ham, dt / 10, accelerators=accelerators)
        final_psi = evolution(final_time=1, initial_state=np.copy(target_psi[0]))
        assert_states_equal(backend, final_psi, target_psi[-1], atol=atol)


def test_adiabatic_evolution_init(backend):
    # Hamiltonians of bad type
    h0 = hamiltonians.X(3, backend=backend)
    s = lambda t: t
    with pytest.raises(TypeError):
        adev = models.AdiabaticEvolution(h0, lambda t: h0, s, dt=1e-2)
    h1 = hamiltonians.TFIM(2, backend=backend)
    with pytest.raises(TypeError):
        adev = models.AdiabaticEvolution(lambda t: h1, h1, s, dt=1e-2)
    # Hamiltonians with different number of qubits
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, s, dt=1e-2)
    # Adiabatic Hamiltonian with bad hamiltonian types
    from qibo.hamiltonians.adiabatic import AdiabaticHamiltonian

    with pytest.raises(TypeError):
        h = AdiabaticHamiltonian("a", "b")  # pylint: disable=E0110
    # s with three arguments
    h0 = hamiltonians.X(2, backend=backend)
    s = lambda t, a, b: t + a + b
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, s, dt=1e-2)


def test_adiabatic_evolution_schedule(backend):
    h0 = hamiltonians.X(3, backend=backend)
    h1 = hamiltonians.TFIM(3, backend=backend)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)
    assert adev.schedule(0.2) == 0.2  # pylint: disable=E1102
    assert adev.schedule(0.8) == 0.8  # pylint: disable=E1102
    # s(0) != 0
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, lambda t: t + 1, dt=1e-2)
    # s(T) != 0
    with pytest.raises(ValueError):
        adev = models.AdiabaticEvolution(h0, h1, lambda t: t / 2, dt=1e-2)


def test_set_scheduling_parameters(backend):
    """Test ``AdiabaticEvolution.set_parameters``."""
    h0 = hamiltonians.X(3, backend=backend)
    h1 = hamiltonians.TFIM(3, backend=backend)
    sp = lambda t, p: (1 - p[0]) * np.sqrt(t) + p[0] * t
    adevp = models.AdiabaticEvolution(h0, h1, sp, 1e-2)
    # access parametrized scheduling before setting parameters
    with pytest.raises(ValueError):
        s = adevp.schedule

    adevp.set_parameters([0.5, 1])

    target_s = lambda t: 0.5 * np.sqrt(t) + 0.5 * t
    for t in np.random.random(10):
        assert adevp.schedule(t) == target_s(t)  # pylint: disable=E1102


@pytest.mark.parametrize("dense", [False, True])
def test_adiabatic_evolution_hamiltonian(backend, dense):
    """Test adiabatic evolution hamiltonian as a function of time."""
    h0 = hamiltonians.X(2, dense=dense, backend=backend)
    h1 = hamiltonians.TFIM(2, dense=dense, backend=backend)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)
    # try accessing hamiltonian before setting it
    with pytest.raises(RuntimeError):
        adev.hamiltonian(0.1)

    m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    m2 = np.diag([2, -2, -2, 2])
    ham = lambda t, T: -(1 - t / T) * m1 - (t / T) * m2

    adev.hamiltonian.total_time = 1
    for t in [0, 0.3, 0.7, 1.0]:
        if dense:
            matrix = adev.hamiltonian(t).matrix
        else:
            matrix = adev.hamiltonian(t).dense.matrix
        backend.assert_allclose(matrix, ham(t, 1))

    # try using a different total time
    adev.hamiltonian.total_time = 2
    for t in [0, 0.3, 0.7, 1.0]:
        if dense:
            matrix = adev.hamiltonian(t).matrix
        else:
            matrix = adev.hamiltonian(t).dense.matrix
        backend.assert_allclose(matrix, ham(t, 2))


@pytest.mark.parametrize("dt", [1e-1])
def test_adiabatic_evolution_execute_exp(backend, dt):
    """Test adiabatic evolution with exponential solver."""
    h0 = hamiltonians.X(2, backend=backend)
    h1 = hamiltonians.TFIM(2, backend=backend)
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt)

    m1 = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
    m2 = np.diag([2, -2, -2, 2])
    ham = lambda t: -(1 - t) * m1 - t * m2

    target_psi = np.ones(4) / 2
    nsteps = int(1 / dt)
    for n in range(nsteps):
        target_psi = expm(-1j * dt * ham(n * dt)).dot(target_psi)
    final_psi = adev(final_time=1)
    assert_states_equal(backend, final_psi, target_psi)


@pytest.mark.parametrize("nqubits,dt", [(4, 1e-1)])
def test_trotterized_adiabatic_evolution(backend, accelerators, nqubits, dt):
    """Test adiabatic evolution using Trotterization."""
    dense_h0 = hamiltonians.X(nqubits, backend=backend)
    dense_h1 = hamiltonians.TFIM(nqubits, backend=backend)

    target_psi = [np.ones(2**nqubits) / np.sqrt(2**nqubits)]
    ham = lambda t: dense_h0 * (1 - t) + dense_h1 * t
    for n in range(int(1 / dt)):
        prop = backend.to_numpy(ham(n * dt).exp(dt))
        target_psi.append(prop.dot(target_psi[-1]))

    local_h0 = hamiltonians.X(nqubits, dense=False, backend=backend)
    local_h1 = hamiltonians.TFIM(nqubits, dense=False, backend=backend)
    checker = TimeStepChecker(target_psi, atol=dt)
    adev = models.AdiabaticEvolution(
        local_h0,
        local_h1,
        lambda t: t,
        dt,
        callbacks=[checker],
        accelerators=accelerators,
    )
    final_psi = adev(final_time=1)


@pytest.mark.parametrize("solver", ["rk4", "rk45"])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("dt", [0.1])
def test_adiabatic_evolution_execute_rk(backend, solver, dense, dt):
    """Test adiabatic evolution with Runge-Kutta solver."""
    h0 = hamiltonians.X(3, dense=dense, backend=backend)
    h1 = hamiltonians.TFIM(3, dense=dense, backend=backend)

    target_psi = [np.ones(8) / np.sqrt(8)]
    ham = lambda t: h0 * (1 - t) + h1 * t
    for n in range(int(1 / dt)):
        prop = backend.to_numpy(ham(n * dt).exp(dt))
        target_psi.append(prop.dot(target_psi[-1]))

    checker = TimeStepChecker(target_psi, atol=dt)
    adev = models.AdiabaticEvolution(
        h0, h1, lambda t: t, dt, solver="rk4", callbacks=[checker]
    )
    final_psi = adev(final_time=1, initial_state=np.copy(target_psi[0]))


def test_adiabatic_evolution_execute_errors(backend):
    h0 = hamiltonians.X(3, backend=backend)
    h1 = hamiltonians.TFIM(3, backend=backend)
    # Non-zero ``start_time``
    adev = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=1e-2)
    with pytest.raises(NotImplementedError):
        final_state = adev(final_time=2, start_time=1)
    # execute without specifying variational parameters
    sp = lambda t, p: (1 - p) * np.sqrt(t) + p * t
    adevp = models.AdiabaticEvolution(h0, h1, sp, dt=1e-1)
    with pytest.raises(RuntimeError):
        final_state = adevp(final_time=1)


@pytest.mark.parametrize("solver,dt,atol", [("exp", 1e-1, 1e-10), ("rk45", 1e-2, 1e-2)])
def test_energy_callback(backend, solver, dt, atol):
    """Test using energy callback in adiabatic evolution."""
    h0 = hamiltonians.X(2, backend=backend)
    h1 = hamiltonians.TFIM(2, backend=backend)
    energy = callbacks.Energy(h1)
    adev = models.AdiabaticEvolution(
        h0, h1, lambda t: t, dt=dt, callbacks=[energy], solver=solver
    )
    final_psi = adev(final_time=1)

    target_psi = np.ones(4) / 2
    calc_energy = lambda psi: psi.conj().dot(backend.to_numpy(h1.matrix).dot(psi))
    target_energies = [calc_energy(target_psi)]
    ham = lambda t: h0 * (1 - t) + h1 * t
    for n in range(int(1 / dt)):
        prop = backend.to_numpy(ham(n * dt).exp(dt))
        target_psi = prop.dot(target_psi)
        target_energies.append(calc_energy(target_psi))

    assert_states_equal(backend, final_psi, target_psi, atol=atol)
    target_energies = backend.cast(target_energies)
    final_energies = np.array([backend.to_numpy(x) for x in energy[:]])
    backend.assert_allclose(final_energies, target_energies, atol=atol)


test_names = "method,options,messages,dense,filename"
test_values = [
    ("BFGS", {"maxiter": 1}, True, True, "adiabatic_bfgs.out"),
    ("BFGS", {"maxiter": 1}, True, False, "trotter_adiabatic_bfgs.out"),
    ("sgd", {"nepochs": 5}, False, True, None),
]


@pytest.mark.parametrize(test_names, test_values)
def test_scheduling_optimization(backend, method, options, messages, dense, filename):
    """Test optimization of s(t)."""
    from .test_models_variational import assert_regression_fixture

    h0 = hamiltonians.X(3, dense=dense, backend=backend)
    h1 = hamiltonians.TFIM(3, dense=dense, backend=backend)
    sp = lambda t, p: (1 - p) * np.sqrt(t) + p * t
    adevp = models.AdiabaticEvolution(h0, h1, sp, dt=1e-1)

    if method == "sgd":
        if backend.platform != "tensorflow":
            with pytest.raises(RuntimeError):
                best, params, _ = adevp.minimize(
                    [0.5, 1], method=method, options=options, messages=messages
                )
    else:
        best, params, _ = adevp.minimize(
            [0.5, 1], method=method, options=options, messages=messages
        )

    if filename is not None:
        assert_regression_fixture(backend, params, filename)
