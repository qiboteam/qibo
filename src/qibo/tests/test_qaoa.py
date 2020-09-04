import pytest
import numpy as np
from scipy.linalg import expm
from qibo import models, hamiltonians
from qibo.tests import utils


@pytest.mark.parametrize("accelerators", [None, {"/GPU:0": 2}])
def test_initial_state(accelerators):
    h = hamiltonians.TFIM(3, h=1.0, trotter=True)
    qaoa = models.QAOA(h, accelerators=accelerators)
    qaoa.set_parameters(np.random.random(4))
    target_state = np.ones(2 ** 3) / np.sqrt(2 ** 3)
    final_state = qaoa.get_initial_state()
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("solver,trotter,accelerators",
                         [("exp", False, None),
                          ("rk4", False, None),
                          ("rk45", False, None),
                          ("exp", True, None),
                          ("rk4", True, None),
                          ("rk45", True, None),
                          ("exp", True, {"/GPU:0": 1, "/GPU:1": 1})])
def test_qaoa_execution(solver, trotter, accelerators):
    h = hamiltonians.TFIM(4, h=1.0, trotter=trotter)
    m = hamiltonians.X(4, trotter=trotter)
    # Trotter and RK require small p's!
    params = 0.01 * (1 - 2 * np.random.random(4))
    state = utils.random_numpy_state(4)
    # set absolute test tolerance according to solver
    if "rk" in solver:
        atol = 1e-2
    elif trotter:
        atol = 1e-5
    else:
        atol = 0

    target_state = np.copy(state)
    h_matrix = h.matrix.numpy()
    m_matrix = m.matrix.numpy()
    for i, p in enumerate(params):
        if i % 2:
            u = expm(-1j * p * m_matrix)
        else:
            u = expm(-1j * p * h_matrix)
        target_state = u @ target_state

    qaoa = models.QAOA(h, mixer=m, solver=solver, accelerators=accelerators)
    qaoa.set_parameters(params)
    final_state = qaoa(np.copy(state))
    np.testing.assert_allclose(final_state, target_state, atol=atol)


@pytest.mark.parametrize("accelerators", [None, {"/GPU:0": 2}])
def test_qaoa_callbacks(accelerators):
    from qibo import callbacks
    # use ``Y`` Hamiltonian so that there are no errors
    # in the Trotter decomposition
    h = hamiltonians.Y(3)
    energy = callbacks.Energy(h)
    params = 0.1 * np.random.random(4)
    state = utils.random_numpy_state(3)

    ham = hamiltonians.Y(3, trotter=True)
    qaoa = models.QAOA(ham, callbacks=[energy], accelerators=accelerators)
    qaoa.set_parameters(params)
    final_state = qaoa(np.copy(state))

    h_matrix = h.matrix.numpy()
    m_matrix = qaoa.mixer.matrix.numpy()
    calc_energy = lambda s: (s.conj() * h_matrix.dot(s)).sum()
    target_state = np.copy(state)
    target_energy = [calc_energy(target_state)]
    for i, p in enumerate(params):
        if i % 2:
            u = expm(-1j * p * m_matrix)
        else:
            u = expm(-1j * p * h_matrix)
        target_state = u @ target_state
        target_energy.append(calc_energy(target_state))
    np.testing.assert_allclose(energy[:], target_energy)


def test_qaoa_errors():
    # Invalid Hamiltonian type
    with pytest.raises(TypeError):
        qaoa = models.QAOA("test")
    # Hamiltonians of different type
    h = hamiltonians.TFIM(4, h=1.0, trotter=True)
    m = hamiltonians.X(4, trotter=False)
    with pytest.raises(TypeError):
        qaoa = models.QAOA(h, mixer=m)
    # distributed execution with RK solver
    with pytest.raises(NotImplementedError):
        qaoa = models.QAOA(h, solver="rk4", accelerators={"/GPU:0": 2})
    # minimize with odd number of parameters
    qaoa = models.QAOA(h)
    with pytest.raises(ValueError):
        qaoa.minimize(np.random.random(5))


test_names = "method,options,trotter,filename"
test_values = [
    ("BFGS", {'maxiter': 1}, False, "qaoa_bfgs.out"),
    ("BFGS", {'maxiter': 1}, True, "trotter_qaoa_bfgs.out"),
    ("Powell", {'maxiter': 1}, True, "trotter_qaoa_powell.out"),
    ("sgd", {"nepochs": 5}, False, None)
    ]
@pytest.mark.parametrize(test_names, test_values)
def test_qaoa_optimization(method, options, trotter, filename):
    h = hamiltonians.XXZ(3, trotter=trotter)
    qaoa = models.QAOA(h)
    initial_p = [0.05, 0.06, 0.07, 0.08]
    best, params = qaoa.minimize(initial_p, method=method, options=options)
    if filename is not None:
        utils.assert_regression_fixture(params, filename)
