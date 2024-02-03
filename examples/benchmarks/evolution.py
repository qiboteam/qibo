"""Adiabatic evolution for the Ising Hamiltonian using linear scaling."""

import argparse
import time

from utils import BenchmarkLogger, parse_accelerators

import qibo
from qibo import callbacks, hamiltonians, models

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)
parser.add_argument("--dense", action="store_true")
parser.add_argument("--accelerators", default=None, type=str)
parser.add_argument("--backend", default="qibojit", type=str)
parser.add_argument("--filename", default=None, type=str)


def main(nqubits, dt, solver, backend, dense=False, accelerators=None, filename=None):
    """Performs adiabatic evolution with critical TFIM as the "hard" Hamiltonian."""
    qibo.set_backend(backend)
    if accelerators is not None:
        dense = False
        solver = "exp"

    logs = BenchmarkLogger(filename)
    logs.append(
        {
            "nqubits": nqubits,
            "dt": dt,
            "solver": solver,
            "dense": dense,
            "backend": qibo.get_backend(),
            "precision": qibo.get_precision(),
            "device": qibo.get_device(),
            "threads": qibo.get_threads(),
            "accelerators": accelerators,
        }
    )
    print(f"Using {solver} solver and dt = {dt}.")
    print(f"Accelerators: {accelerators}")
    print("Backend:", logs[-1]["backend"])

    start_time = time.time()
    h0 = hamiltonians.X(nqubits, dense=dense)
    h1 = hamiltonians.TFIM(nqubits, h=1.0, dense=dense)
    logs[-1]["hamiltonian_creation_time"] = time.time() - start_time
    print(f"\nnqubits = {nqubits}, solver = {solver}")
    print(f"dense = {dense}, accelerators = {accelerators}")
    print("Hamiltonians created in:", logs[-1]["hamiltonian_creation_time"])

    start_time = time.time()
    evolution = models.AdiabaticEvolution(
        h0, h1, lambda t: t, dt=dt, solver=solver, accelerators=accelerators
    )
    logs[-1]["creation_time"] = time.time() - start_time
    print("Evolution model created in:", logs[-1]["creation_time"])

    start_time = time.time()
    final_psi = evolution(final_time=1.0)
    logs[-1]["simulation_time"] = time.time() - start_time
    print("Simulation time:", logs[-1]["simulation_time"])
    logs.dump()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["accelerators"] = parse_accelerators(args.pop("accelerators"))
    main(**args)
