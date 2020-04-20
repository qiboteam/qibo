"""
Testing Variational Quantum Eigensolver.
"""
import time
import argparse
import numpy as np
from qibo.models import Circuit, VQE
from qibo import gates
from qibo.hamiltonians import XXZ


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=6, help="Number of qubits.", type=int)
parser.add_argument("--max-layers", default=5, help="Number of layers.", type=int)


def main(nqubits, max_layers):
    """Performs a VQE circuit minimization test."""

    def ansatz(theta):
        c = Circuit(nqubits)
        index = 0
        for l in range(layers):
            for q in range(nqubits):
                c.add(gates.RY(q, theta[index]))
                index+=1
            for q in range(0, nqubits-1, 2):
                c.add(gates.CZPow(q, q+1, np.pi))
            for q in range(nqubits):
                c.add(gates.RY(q, theta[index]))
                index+=1
            for q in range(1, nqubits-2, 2):
                c.add(gates.CZPow(q, q+1, np.pi))
            c.add(gates.CZPow(0, nqubits-1, np.pi))
        for q in range(nqubits):
            c.add(gates.RY(q, theta[index]))
            index+=1
        return c

    hamiltonian = XXZ(nqubits=nqubits)
    target = np.real(np.min(hamiltonian.eigenvalues().numpy()))
    v = VQE(ansatz, hamiltonian)

    np.random.seed(0)
    params = []
    print('nqubits layers target best epsilon time(s)')
    for layers in range(1, max_layers+1):

        initial_parameters = np.array(params)
        initial_parameters = np.append(initial_parameters, np.random.uniform(0, 2*np.pi,
                                       2*nqubits*layers + nqubits - len(params)))
        t0 = time.time()
        best, params = v.minimize(initial_parameters)
        epsilon = np.log10(1/np.abs(best-target))
        print(f'{nqubits} {layers} {target} {best} {epsilon} {time.time()-t0}')


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
