"""
Benchmark Quantum Approximate Optimization Algorithm model.
"""

import argparse
import time

import numpy as np
from utils import BenchmarkLogger

import qibo
from qibo import hamiltonians, models

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, help="Number of qubits.", type=int)
parser.add_argument("--nangles", default=4, help="Number of layers.", type=int)
parser.add_argument(
    "--dense", action="store_true", help="Use dense Hamiltonian or terms."
)
parser.add_argument("--solver", default="exp", help="Solver to apply gates.", type=str)
parser.add_argument("--method", default="Powell", help="Optimization method.", type=str)
parser.add_argument(
    "--maxiter", default=None, help="Maximum optimization iterations.", type=int
)
parser.add_argument(
    "--filename", default=None, help="Name of file to save logs.", type=str
)


def main(
    nqubits,
    nangles,
    dense=True,
    solver="exp",
    method="Powell",
    maxiter=None,
    filename=None,
):
    """Performs a QAOA minimization test."""

    print("Number of qubits:", nqubits)
    print("Number of angles:", nangles)
    logs = BenchmarkLogger(filename)
    logs.append(
        {
            "nqubits": nqubits,
            "nangles": nangles,
            "dense": dense,
            "solver": solver,
            "backend": qibo.get_backend(),
            "precision": qibo.get_precision(),
            "device": qibo.get_device(),
            "threads": qibo.get_threads(),
            "method": method,
            "maxiter": maxiter,
        }
    )

    hamiltonian = hamiltonians.XXZ(nqubits=nqubits, dense=dense)
    start_time = time.time()
    qaoa = models.QAOA(hamiltonian, solver=solver)
    logs[-1]["creation_time"] = time.time() - start_time

    target = np.real(np.min(hamiltonian.eigenvalues()))
    print("\nTarget state =", target)

    np.random.seed(0)
    initial_parameters = np.random.uniform(0, 0.1, nangles)

    start_time = time.time()
    options = {"disp": True, "maxiter": maxiter}
    best, params, _ = qaoa.minimize(initial_parameters, method=method, options=options)
    logs[-1]["minimization_time"] = time.time() - start_time

    logs[-1]["best_energy"] = float(best)
    logs[-1]["target_energy"] = float(target)
    logs[-1]["epsilon"] = np.log10(1 / np.abs(best - target))
    print("Found state =", best)
    print("Final eps =", logs[-1]["epsilon"])

    print("\nCreation time =", logs[-1]["creation_time"])
    print("Minimization time =", logs[-1]["minimization_time"])
    print("Total time =", logs[-1]["creation_time"] + logs[-1]["minimization_time"])
    logs.dump()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
