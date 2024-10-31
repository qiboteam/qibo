"""
Benchmark Variational Quantum Eigensolver.
"""

import argparse
import time

import numpy as np
from utils import BenchmarkLogger

from qibo import (
    Circuit,
    gates,
    get_backend,
    get_device,
    get_precision,
    get_threads,
    set_backend,
)
from qibo.hamiltonians import XXZ
from qibo.models import VQE

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=6, help="Number of qubits.", type=int)
parser.add_argument("--nlayers", default=4, help="Number of layers.", type=int)
parser.add_argument("--method", default="Powell", help="Optimization method.", type=str)
parser.add_argument(
    "--maxiter", default=None, help="Maximum optimization iterations.", type=int
)
parser.add_argument(
    "--backend", default="qibojit", help="Qibo backend to use.", type=str
)
parser.add_argument("--fuse", action="store_true", help="Use gate fusion.")
parser.add_argument(
    "--filename", default=None, help="Name of file to save logs.", type=str
)


def create_circuit(nqubits, nlayers):
    """Creates variational circuit."""
    circuit = Circuit(nqubits)
    for l in range(nlayers):
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(0, nqubits - 1, 2))
        circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
        circuit.add(gates.CZ(q, q + 1) for q in range(1, nqubits - 2, 2))
        circuit.add(gates.CZ(0, nqubits - 1))
    circuit.add(gates.RY(q, theta=0) for q in range(nqubits))
    return circuit


def main(
    nqubits, nlayers, backend, fuse=False, method="Powell", maxiter=None, filename=None
):
    """Performs a VQE circuit minimization test."""
    set_backend(backend)
    logs = BenchmarkLogger(filename)
    logs.append(
        {
            "nqubits": nqubits,
            "nlayers": nlayers,
            "fuse": fuse,
            "backend": get_backend(),
            "precision": get_precision(),
            "device": get_device(),
            "threads": get_threads(),
            "method": method,
            "maxiter": maxiter,
        }
    )
    print("Number of qubits:", nqubits)
    print("Number of layers:", nlayers)
    print("Backend:", logs[-1]["backend"])

    start_time = time.time()
    circuit = create_circuit(nqubits, nlayers)
    if fuse:
        circuit = circuit.fuse()
    hamiltonian = XXZ(nqubits=nqubits)
    vqe = VQE(circuit, hamiltonian)
    logs[-1]["creation_time"] = time.time() - start_time

    target = np.real(np.min(hamiltonian.eigenvalues()))
    print("\nTarget state =", target)

    np.random.seed(0)
    nparams = 2 * nqubits * nlayers + nqubits
    initial_parameters = np.random.uniform(0, 2 * np.pi, nparams)

    start_time = time.time()
    options = {"disp": False, "maxiter": maxiter}
    best, params, _ = vqe.minimize(
        initial_parameters, method=method, options=options, compile=False
    )
    logs[-1]["minimization_time"] = time.time() - start_time
    epsilon = np.log10(1 / np.abs(best - target))
    print("Found state =", best)
    print("Final eps =", epsilon)

    logs[-1]["best_energy"] = float(best)
    logs[-1]["epsilon_energy"] = float(epsilon)

    print("\nCreation time =", logs[-1]["creation_time"])
    print("Minimization time =", logs[-1]["minimization_time"])
    print("Total time =", logs[-1]["minimization_time"] + logs[-1]["creation_time"])
    logs.dump()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
