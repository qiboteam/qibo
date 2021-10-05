"""
Testing Variational Quantum Circuits.
"""
import numpy as np
import pathlib
import pytest
import qibo
from qibo import gates, models, hamiltonians, K
from qibo.tests.utils import random_state
from scipy.linalg import expm

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(array, filename, rtol=1e-5):
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

    filename = REGRESSION_FOLDER/filename
    try:
        array_fixture = load(filename)
    except: # pragma: no cover
        # case not tested in GitHub workflows because files exist
        np.savetxt(filename, array)
        array_fixture = load(filename)
    K.assert_allclose(array, array_fixture, rtol=rtol)


test_names = "method,options,compile,filename"
test_values = [("Powell", {'maxiter': 1}, True, 'vqc_powell.out'),
               ("Powell", {'maxiter': 1}, False, 'vqc_powell.out'),
               ("BFGS", {'maxiter': 1}, True, 'vqc_bfgs.out'),
               ("BFGS", {'maxiter': 1}, False, 'vqc_bfgs.out')]
@pytest.mark.parametrize(test_names, test_values)
def test_vqc(backend, method, options, compile, filename):
    """Performs a variational circuit minimization test."""
    from qibo.optimizers import optimize
    def myloss(parameters, circuit, target):
        circuit.set_parameters(parameters)
        state = K.to_numpy(circuit().tensor)
        return 1 - np.abs(np.dot(np.conj(target), state))

    nqubits = 6
    nlayers  = 4

    # Create variational circuit
    c = models.Circuit(nqubits)
    for l in range(nlayers):
        c.add((gates.RY(q, theta=0) for q in range(nqubits)))
        c.add((gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
        c.add((gates.RY(q, theta=0) for q in range(nqubits)))
        c.add((gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
        c.add(gates.CZ(0, nqubits-1))
    c.add((gates.RY(q, theta=0) for q in range(nqubits)))

    # Optimize starting from a random guess for the variational parameters
    np.random.seed(0)
    x0 = np.random.uniform(0, 2*np.pi, 2*nqubits*nlayers + nqubits)
    data = np.random.normal(0, 1, size=2**nqubits)

    # perform optimization
    best, params, _ = optimize(myloss, x0, args=(c, data), method=method,
                            options=options, compile=compile)
    if filename is not None:
        assert_regression_fixture(params, filename)


test_names = "method,options,compile,filename"
test_values = [("Powell", {'maxiter': 1}, True, 'vqe_powell.out'),
               ("Powell", {'maxiter': 1}, False, 'vqe_powell.out'),
               ("BFGS", {'maxiter': 1}, True, 'vqe_bfgs.out'),
               ("BFGS", {'maxiter': 1}, False, 'vqe_bfgs.out'),
               ("parallel_L-BFGS-B", {'maxiter': 1}, True, None),
               ("parallel_L-BFGS-B", {'maxiter': 1}, False, None),
               ("cma", {"maxfevals": 2}, False, None),
               ("sgd", {"nepochs": 5}, False, None),
               ("sgd", {"nepochs": 5}, True, None)]
@pytest.mark.parametrize(test_names, test_values)
def test_vqe(backend, method, options, compile, filename):
    """Performs a VQE circuit minimization test."""
    original_threads = qibo.get_threads()
    if (method == "sgd" or compile) and qibo.get_backend() != "tensorflow":
        pytest.skip("Skipping SGD test for unsupported backend.")

    if method == 'parallel_L-BFGS-B':
        device = qibo.get_device()
        backend = qibo.get_backend()
        if backend == "tensorflow" or backend == "qibojit" or "GPU" in device:
            pytest.skip("unsupported configuration")
        import sys
        if sys.platform == 'win32' or sys.platform == 'darwin': # pragma: no cover
            pytest.skip("Parallel L-BFGS-B only supported on linux.")
        qibo.set_threads(1)

    nqubits = 6
    layers  = 4
    circuit = models.Circuit(nqubits)
    for l in range(layers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=1.0))
        for q in range(0, nqubits-1, 2):
            circuit.add(gates.CZ(q, q+1))
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=1.0))
        for q in range(1, nqubits-2, 2):
            circuit.add(gates.CZ(q, q+1))
        circuit.add(gates.CZ(0, nqubits-1))
    for q in range(nqubits):
        circuit.add(gates.RY(q, theta=1.0))

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits)
    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 2*np.pi, 2*nqubits*layers + nqubits)
    v = models.VQE(circuit, hamiltonian)
    best, params, _ = v.minimize(initial_parameters, method=method,
                                 options=options, compile=compile)
    if method == "cma":
        # remove `outcmaes` folder
        import shutil
        shutil.rmtree("outcmaes")
    if filename is not None:
        assert_regression_fixture(params, filename)
    qibo.set_threads(original_threads)


def test_vqe_custom_gates_errors():
    """Check that ``RuntimeError``s is raised when using custom gates."""
    if "qibotf" not in qibo.K.available_backends: # pragma: no cover
        pytest.skip("Custom backend not available.")

    original_backend = qibo.get_backend()
    qibo.set_backend("qibotf")

    nqubits = 6
    circuit = models.Circuit(nqubits)
    for q in range(nqubits):
        circuit.add(gates.RY(q, theta=0))
    for q in range(0, nqubits-1, 2):
        circuit.add(gates.CZ(q, q+1))

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits)
    initial_parameters = np.random.uniform(0, 2*np.pi, 2*nqubits + nqubits)
    v = models.VQE(circuit, hamiltonian)
    # compile with custom gates
    with pytest.raises(RuntimeError):
        best, params, _ = v.minimize(initial_parameters, method="BFGS",
                                     options={'maxiter': 1}, compile=True)
    # use SGD with custom gates
    with pytest.raises(RuntimeError):
        best, params, _ = v.minimize(initial_parameters, method="sgd",
                                     compile=False)
    qibo.set_backend(original_backend)


def test_initial_state(backend, accelerators):
    h = hamiltonians.TFIM(5, h=1.0, dense=False)
    qaoa = models.QAOA(h, accelerators=accelerators)
    qaoa.set_parameters(np.random.random(4))
    target_state = np.ones(2 ** 5) / np.sqrt(2 ** 5)
    final_state = qaoa.get_initial_state()
    K.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("solver,dense",
                         [("exp", False), ("exp", True),
                          ("rk4", False),  ("rk4", True),
                          ("rk45", False), ("rk45", True)])
def test_qaoa_execution(backend, solver, dense, accel=None):
    h = hamiltonians.TFIM(6, h=1.0, dense=dense)
    m = hamiltonians.X(6, dense=dense)
    # Trotter and RK require small p's!
    params = 0.01 * (1 - 2 * np.random.random(4))
    state = random_state(6)
    # set absolute test tolerance according to solver
    if "rk" in solver:
        atol = 1e-2
    elif not dense:
        atol = 1e-5
    else:
        atol = 0

    target_state = np.copy(state)
    h_matrix = K.to_numpy(h.matrix)
    m_matrix = K.to_numpy(m.matrix)
    for i, p in enumerate(params):
        if i % 2:
            u = expm(-1j * p * m_matrix)
        else:
            u = expm(-1j * p * h_matrix)
        target_state = u @ target_state

    qaoa = models.QAOA(h, mixer=m, solver=solver, accelerators=accel)
    qaoa.set_parameters(params)
    final_state = qaoa(np.copy(state))
    K.assert_allclose(final_state, target_state, atol=atol)


def test_qaoa_distributed_execution(backend, accelerators):
    test_qaoa_execution(backend, "exp", False, accelerators)


def test_qaoa_callbacks(backend, accelerators):
    from qibo import callbacks
    # use ``Y`` Hamiltonian so that there are no errors
    # in the Trotter decomposition
    if accelerators:
        with K.on_cpu():
            h = hamiltonians.Y(5)
    else:
        h = hamiltonians.Y(5)
    energy = callbacks.Energy(h)
    params = 0.1 * np.random.random(4)
    state = random_state(5)

    ham = hamiltonians.Y(5, dense=False)
    qaoa = models.QAOA(ham, callbacks=[energy], accelerators=accelerators)
    qaoa.set_parameters(params)
    final_state = qaoa(np.copy(state))

    h_matrix = K.to_numpy(h.matrix)
    m_matrix = K.to_numpy(qaoa.mixer.matrix)
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
    K.assert_allclose(energy[:], target_energy)


def test_qaoa_errors():
    # Invalid Hamiltonian type
    with pytest.raises(TypeError):
        qaoa = models.QAOA("test")
    # Hamiltonians of different type
    h = hamiltonians.TFIM(4, h=1.0, dense=False)
    m = hamiltonians.X(4, dense=True)
    with pytest.raises(TypeError):
        qaoa = models.QAOA(h, mixer=m)
    # distributed execution with RK solver
    with pytest.raises(NotImplementedError):
        qaoa = models.QAOA(h, solver="rk4", accelerators={"/GPU:0": 2})
    # minimize with odd number of parameters
    qaoa = models.QAOA(h)
    with pytest.raises(ValueError):
        qaoa.minimize(np.random.random(5))


test_names = "method,options,dense,filename"
test_values = [
    ("BFGS", {'maxiter': 1}, True, "qaoa_bfgs.out"),
    ("BFGS", {'maxiter': 1}, False, "trotter_qaoa_bfgs.out"),
    ("Powell", {'maxiter': 1}, False, "trotter_qaoa_powell.out"),
    ("sgd", {"nepochs": 5}, True, None)
    ]
@pytest.mark.parametrize(test_names, test_values)
def test_qaoa_optimization(backend, method, options, dense, filename):
    if method == "sgd" and qibo.get_backend() != "tensorflow":
        pytest.skip("Skipping SGD test for unsupported backend.")
    h = hamiltonians.XXZ(3, dense=dense)
    qaoa = models.QAOA(h)
    initial_p = [0.05, 0.06, 0.07, 0.08]
    best, params, _ = qaoa.minimize(initial_p, method=method, options=options)
    if filename is not None:
        assert_regression_fixture(params, filename)


test_names = "delta_t,max_layers,tolerance,filename"
test_values = [
    (0.1, 5, None, "falqon1.out"),
    (0.01, 2, None, "falqon2.out"),
    (0.01, 2, 1e-5, "falqon3.out"),
    (0.01, 5, 1, "falqon4.out")
    ]
@pytest.mark.parametrize(test_names, test_values)
def test_falqon_optimization(backend, delta_t, max_layers, tolerance, filename):
    h = hamiltonians.XXZ(3)
    falqon = models.FALQON(h)
    best, params, extra = falqon.minimize(delta_t, max_layers, tol=tolerance)
    if filename is not None:
        assert_regression_fixture(params, filename)


def test_falqon_optimization_callback(backend):
    class TestCallback:
        def __call__(self, x):
            return K.sum(x)

    callback = TestCallback()
    h = hamiltonians.XXZ(3)
    falqon = models.FALQON(h)
    best, params, extra = falqon.minimize(0.1, 5, callback=callback)
    assert len(extra["callbacks"]) == 5


test_names = "method,options,compile,filename"
test_values = [("Powell", {'maxiter': 1}, True, 'aavqe_powell.out'),
               ("Powell", {'maxiter': 1}, False, 'aavqe_powell.out'),
               ("BFGS", {'maxiter': 1}, True, 'aavqe_bfgs.out'),
               ("BFGS", {'maxiter': 1}, False, 'aavqe_bfgs.out'),
               ("cma", {"maxfevals": 2}, False, None)]
@pytest.mark.parametrize(test_names, test_values)
def test_aavqe(backend, method, options, compile, filename):
    """Performs a AAVQE circuit minimization test."""
    original_threads = qibo.get_threads()

    nqubits = 6
    layers  = 4
    circuit = models.Circuit(nqubits)

    for l in range(layers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=1.0))
        for q in range(0, nqubits-1, 2):
            circuit.add(gates.CZ(q, q+1))
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=1.0))
        for q in range(1, nqubits-2, 2):
            circuit.add(gates.CZ(q, q+1))
        circuit.add(gates.CZ(0, nqubits-1))
    for q in range(nqubits):
        circuit.add(gates.RY(q, theta=1.0))

    easy_hamiltonian=hamiltonians.X(nqubits)
    problem_hamiltonian=hamiltonians.XXZ(nqubits)
    s = lambda t: t
    aavqe = models.AAVQE(circuit, easy_hamiltonian, problem_hamiltonian, 
                        s, nsteps=10, t_max=1)
    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 2*np.pi, 2*nqubits*layers + nqubits)
    best, params = aavqe.minimize(params=initial_parameters, method=method,
                                 options=options, compile=compile)
    if method == "cma":
        # remove `outcmaes` folder
        import shutil
        shutil.rmtree("outcmaes")
    if filename is not None:
        assert_regression_fixture(params, filename)
    qibo.set_threads(original_threads)