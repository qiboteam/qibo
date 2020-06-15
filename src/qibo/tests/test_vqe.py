"""
Testing Variational Quantum Eigensolver.
"""
import pathlib
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit, VQE
from qibo.hamiltonians import XXZ


REGRESSION_FOLDER = pathlib.Path(__file__).with_name('regressions')

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
    except:
        np.savetxt(filename, array)
        array_fixture = load(filename)
    np.testing.assert_allclose(array, array_fixture, rtol=1e-5)


test_names = "method,options,compile,filename"
test_values = [("BFGS", {'maxiter': 1}, True, 'vqe.out'),
               ("BFGS", {'maxiter': 1}, False, 'vqe.out'),
               ("sgd", {"nepochs": 5}, False, None),
               ("sgd", {"nepochs": 5}, True, None)]
@pytest.mark.parametrize(test_names, test_values)
def test_vqe(method, options, compile, filename):
    """Performs a VQE circuit minimization test."""
    import qibo
    if method == "sgd" or compile:
        qibo.set_backend("matmuleinsum")
    else:
        qibo.set_backend("custom")

    nqubits = 6
    layers  = 4

    def ansatz(theta):
        c = Circuit(nqubits)
        index = 0
        for l in range(layers):
            for q in range(nqubits):
                c.add(gates.RY(q, theta[index]))
                index+=1
            for q in range(0, nqubits-1, 2):
                c.add(gates.CZ(q, q+1))
            for q in range(nqubits):
                c.add(gates.RY(q, theta[index]))
                index+=1
            for q in range(1, nqubits-2, 2):
                c.add(gates.CZ(q, q+1))
            c.add(gates.CZ(0, nqubits-1))
        for q in range(nqubits):
            c.add(gates.RY(q, theta[index]))
            index+=1
        return c

    hamiltonian = XXZ(nqubits=nqubits)
    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 2*np.pi, 2*nqubits*layers + nqubits)
    v = VQE(ansatz, hamiltonian)
    best, params = v.minimize(initial_parameters, method=method,
                              options=options, compile=compile)
    if filename is not None:
        assert_regression_fixture(params, REGRESSION_FOLDER/filename)


def test_vqe_compile_error():
    """Check that ``RuntimeError`` is raised when compiling custom gates."""
    import qibo
    qibo.set_backend("custom")

    nqubits = 6
    def ansatz(theta):
        c = Circuit(nqubits)
        index = 0
        for q in range(nqubits):
            c.add(gates.RY(q, theta[index]))
            index+=1
        for q in range(0, nqubits-1, 2):
            c.add(gates.CZ(q, q+1))
        return c

    hamiltonian = XXZ(nqubits=nqubits)
    initial_parameters = np.random.uniform(0, 2*np.pi, 2*nqubits + nqubits)
    v = VQE(ansatz, hamiltonian)
    with pytest.raises(RuntimeError):
        best, params = v.minimize(initial_parameters, method="BFGS",
                                  options={'maxiter': 1}, compile=True)
