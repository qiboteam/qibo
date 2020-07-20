"""
Testing Variational Quantum Eigensolver.
"""
import argparse
import numpy as np
from qibo import gates, models, hamiltonians


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=6, help="Number of qubits.", type=int)
parser.add_argument("--nlayers", default=4, help="Number of layers.", type=int)
parser.add_argument("--method", default="Powell", help="Optimization method.", type=str)
parser.add_argument("--maxiter", default=None, help="Maximum optimization iterations.", type=int)
parser.add_argument("--varlayer", action="store_true", help="Use VariationalLayer gate.")


def standard_circuit(nqubits, nlayers):
    """Creates variational circuit using normal gates."""
    circuit = models.Circuit(nqubits)
    for l in range(nlayers):
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
        circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
        circuit.add((gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
        circuit.add(gates.CZ(0, nqubits-1))
    circuit.add((gates.RY(q, theta=0) for q in range(nqubits)))
    return circuit


def varlayer_circuit(nqubits, nlayers):
    """Creates variational circuit using ``VariationalLayer`` gate."""
    circuit = models.Circuit(nqubits)
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    theta = np.zeros(nqubits)
    for l in range(nlayers):
        circuit.add(gates.VariationalLayer(range(nqubits), pairs,
                                           gates.RY, gates.CZ,
                                           theta, theta))
        circuit.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add((gates.RY(i, theta) for i in range(nqubits)))
    return circuit


def main(nqubits, nlayers, varlayer=False, method="Powell", maxiter=None):
    """Performs a VQE circuit minimization test."""

    print("Number of qubits:", nqubits)
    print("Number of layers:", nlayers)

    if varlayer:
        circuit = varlayer_circuit(nqubits, nlayers)
    else:
        circuit = standard_circuit(nqubits, nlayers)

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits)
    target = np.real(np.min(hamiltonian.eigenvalues().numpy()))

    print('Target state =', target)

    np.random.seed(0)
    nparams = 2 * nqubits * nlayers + nqubits
    initial_parameters = np.random.uniform(0, 2 * np.pi, nparams)
    vqe = models.VQE(circuit, hamiltonian)
    options = {'disp': True, 'maxiter': maxiter}
    best, params = vqe.minimize(initial_parameters, method=method,
                                options=options, compile=False)
    epsilon = np.log10(1/np.abs(best-target))
    print('Found state =', best)
    print('Final eps =', epsilon)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
