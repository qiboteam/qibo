"""
Testing Variational Quantum Eigensolver.
"""
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit, VQE
from qibo.hamiltonians import XXZ
from qibo.tests import utils


test_names = "method,options,compile,filename"
test_values = [("Powell", {'maxiter': 1}, True, 'vqe_powell.out'),
               ("Powell", {'maxiter': 1}, False, 'vqe_powell.out'),
               ("BFGS", {'maxiter': 1}, True, 'vqe_bfgs.out'),
               ("BFGS", {'maxiter': 1}, False, 'vqe_bfgs.out'),
               ("cma", {"maxfevals": 2}, False, None),
               ("sgd", {"nepochs": 5}, False, None),
               ("sgd", {"nepochs": 5}, True, None)]
@pytest.mark.parametrize(test_names, test_values)
def test_vqe(method, options, compile, filename):
    """Performs a VQE circuit minimization test."""
    import qibo
    original_backend = qibo.get_backend()
    if method == "sgd" or compile:
        qibo.set_backend("matmuleinsum")
    else:
        qibo.set_backend("custom")

    nqubits = 6
    layers  = 4

    circuit = Circuit(nqubits)
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

    hamiltonian = XXZ(nqubits=nqubits)
    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 2*np.pi, 2*nqubits*layers + nqubits)
    v = VQE(circuit, hamiltonian)
    best, params = v.minimize(initial_parameters, method=method,
                              options=options, compile=compile)
    if method == "cma":
        # remove `outcmaes` folder
        import shutil
        shutil.rmtree("outcmaes")
    if filename is not None:
        utils.assert_regression_fixture(params, filename)
    qibo.set_backend(original_backend)


def test_vqe_custom_gates_errors():
    """Check that ``RuntimeError``s is raised when using custom gates."""
    import qibo
    original_backend = qibo.get_backend()
    qibo.set_backend("custom")

    nqubits = 6
    circuit = Circuit(nqubits)
    for q in range(nqubits):
        circuit.add(gates.RY(q, theta=0))
    for q in range(0, nqubits-1, 2):
        circuit.add(gates.CZ(q, q+1))

    hamiltonian = XXZ(nqubits=nqubits)
    initial_parameters = np.random.uniform(0, 2*np.pi, 2*nqubits + nqubits)
    v = VQE(circuit, hamiltonian)
    # compile with custom gates
    with pytest.raises(RuntimeError):
        best, params = v.minimize(initial_parameters, method="BFGS",
                                  options={'maxiter': 1}, compile=True)
    # use SGD with custom gates
    with pytest.raises(RuntimeError):
        best, params = v.minimize(initial_parameters, method="sgd",
                                  compile=False)
    qibo.set_backend(original_backend)
