"""
Testing Quantum Approximate Optimization Algorithm model.
"""
import argparse
import time
import numpy as np
from qibo import models, hamiltonians


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, help="Number of qubits.", type=int)
parser.add_argument("--nangles", default=4, help="Number of layers.", type=int)
parser.add_argument("--dense", action="store_true", help="Use dense Hamiltonian or terms.")
parser.add_argument("--solver", default="exp", help="Solver to apply gates.", type=str)
parser.add_argument("--method", default="Powell", help="Optimization method.", type=str)
parser.add_argument("--maxiter", default=None, help="Maximum optimization iterations.", type=int)


def main(nqubits, nangles, dense=True, solver="exp",
         method="Powell", maxiter=None):
    """Performs a QAOA minimization test."""

    print("Number of qubits:", nqubits)
    print("Number of angles:", nangles)

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, dense=dense)
    start_time = time.time()
    qaoa = models.QAOA(hamiltonian, solver=solver)
    creation_time = time.time() - start_time

    target = np.real(np.min(hamiltonian.eigenvalues()))
    print("\nTarget state =", target)

    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 0.1, nangles)

    start_time = time.time()
    options = {'disp': True, 'maxiter': maxiter}
    best, params, _ = qaoa.minimize(initial_parameters, method=method,
                                    options=options)
    minimization_time = time.time() - start_time

    epsilon = np.log10(1/np.abs(best - target))
    print("Found state =", best)
    print("Final eps =", epsilon)

    print("\nCreation time =", creation_time)
    print("Minimization time =", minimization_time)
    print("Total time =", minimization_time + creation_time)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
