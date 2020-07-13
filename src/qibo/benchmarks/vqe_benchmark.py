"""
Testing Variational Quantum Eigensolver.
"""
import argparse
import numpy as np
from qibo.models import Circuit, VQE
from qibo import gates
from qibo.hamiltonians import XXZ


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=6, help="Number of qubits.", type=int)
parser.add_argument("--layers", default=4, help="Number of layers.", type=int)


def main(nqubits, layers):
    """Performs a VQE circuit minimization test."""

    print("Number of qubits:", nqubits)
    print("Number of layers:", layers)

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
    target = np.real(np.min(hamiltonian.eigenvalues().numpy()))

    print('Target state =', target)

    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 2*np.pi,
                                           2*nqubits*layers + nqubits)
    v = VQE(ansatz, hamiltonian)
    best, params = v.minimize(initial_parameters, method='Powell',
                              options={'disp': True}, compile=False)
    epsilon = np.log10(1/np.abs(best-target))
    print('Found state =', best)
    print('Final eps =', epsilon)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
