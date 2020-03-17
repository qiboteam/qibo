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
        array_fixture = load(filaneme)
    except:
        np.savetxt(filename, array)
        array_fixture = load(filename)
    np.testing.assert_array_equal(array, array_fixture)


def test_vqe():
    """Performs a VQE circuit minimization test."""

    nqubits = 6
    layers  = 4

    class Ansatz:

        def __init__(self, nqubits, layers):
            self.circuit = Circuit(nqubits)
            self.parametrized_gates = []
            for l in range(layers):
                for q in range(nqubits):
                    self.parametrized_gates.append(
                        self.circuit.add(gates.RY(q, 0)))
                for q in range(0, nqubits-1, 2):
                    self.circuit.add(gates.CRZ(q, q+1, 1))
                for q in range(nqubits):
                    self.parametrized_gates.append(
                        self.circuit.add(gates.RY(q, 0)))
                for q in range(1, nqubits-2, 2):
                    self.circuit.add(gates.CRZ(q, q+1, 1))
                self.circuit.add(gates.CRZ(0, nqubits-1, 1))
            for q in range(nqubits):
                self.parametrized_gates.append(self.circuit.add(gates.RY(q, 0)))

            # Set initial state in numpy because circuit's internal state
            # will change every time this is called
            self.initial_state = np.zeros(2 ** nqubits)
            self.initial_state[0] = 1.0

        def __call__(self, theta):
            for i, gate in enumerate(self.parametrized_gates):
                gate.update(theta[i])
            return self.circuit(self.initial_state)


    hamiltonian = XXZ(nqubits=nqubits)
    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 2*np.pi,
                                           2*nqubits*layers + nqubits)
    ansatz = Ansatz(nqubits, layers)
    v = VQE(ansatz, hamiltonian)
    best, params = v.minimize(initial_parameters, method='BFGS', options={'maxiter': 1})
    assert_regression_fixture(params, REGRESSION_FOLDER/'vqe.out')
