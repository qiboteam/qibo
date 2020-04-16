"""
Testing Variational Quantum Eigensolver.
"""
import pathlib
import numpy as np
from qibo.models import Circuit, VQE
from qibo import gates
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
    np.testing.assert_allclose(array, array_fixture)


def test_vqe():
    """Performs a VQE circuit minimization test."""

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
                c.add(gates.CRZ(q, q+1, 1))
            for q in range(nqubits):
                c.add(gates.RY(q, theta[index]))
                index+=1
            for q in range(1, nqubits-2, 2):
                c.add(gates.CRZ(q, q+1, 1))
            c.add(gates.CRZ(0, nqubits-1, 1))
        for q in range(nqubits):
            c.add(gates.RY(q, theta[index]))
            index+=1
        return c

    hamiltonian = XXZ(nqubits=nqubits)
    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 2,
                                           2*nqubits*layers + nqubits)
    v = VQE(ansatz, hamiltonian)
    best, params = v.minimize(initial_parameters, method='BFGS', options={'maxiter': 1})
    assert_regression_fixture(params, REGRESSION_FOLDER/'vqe.out')
