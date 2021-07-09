"""Adiabatic evolution for the Ising Hamiltonian using linear scaling."""
import os
import argparse
import json
import time
import qibo
from qibo import callbacks, hamiltonians, models

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=4, type=int)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)
parser.add_argument("--dense", action="store_true")
parser.add_argument("--accelerators", default=None, type=str)
parser.add_argument("--backend", default="qibotf", type=str)
parser.add_argument("--filename", default=None, type=str)


def parse_accelerators(accelerators):
    """Transforms string that specifies accelerators to dictionary.

    The string that is parsed has the following format:
        n1device1,n2device2,n3device3,...
    and is transformed to the dictionary:
        {'device1': n1, 'device2': n2, 'device3': n3, ...}

    Example:
        2/GPU:0,2/GPU:1 --> {'/GPU:0': 2, '/GPU:1': 2}
    """
    if accelerators is None:
        return None

    def read_digit(x):
        i = 0
        while x[i].isdigit():
            i += 1
        return x[i:], int(x[:i])

    acc_dict = {}
    for entry in accelerators.split(","):
        device, n = read_digit(entry)
        if device in acc_dict:
            acc_dict[device] += n
        else:
            acc_dict[device] = n
    return acc_dict


def main(nqubits, dt, solver, backend, trotter=False, accelerators=None,
         filename=None):
    """Performs adiabatic evolution with critical TFIM as the "hard" Hamiltonian."""
    qibo.set_backend(backend)
    if accelerators is not None:
        dense = False
        solver = "exp"

    if filename is not None:
        if os.path.isfile(filename):
            with open(filename, "r") as file:
                logs = json.load(file)
            print("Extending existing logs from {}.".format(filename))
        else:
            print("Creating new logs in {}.".format(filename))
            logs = []
    else:
        logs = []

    logs.append({
        "nqubits": nqubits, "dt": dt, "solver": solver, "trotter": trotter,
        "backend": qibo.get_backend(), "precision": qibo.get_precision(),
        "device": qibo.get_device(), "threads": qibo.get_threads(),
        "accelerators": accelerators
        })
    print(f"Using {solver} solver and dt = {dt}.")
    print(f"Accelerators: {accelerators}")
    print("Backend:", logs[-1]["backend"])

    start_time = time.time()
    h0 = hamiltonians.X(nqubits, trotter=trotter)
    h1 = hamiltonians.TFIM(nqubits, h=1.0, trotter=trotter)
    logs[-1]["hamiltonian_creation_time"] = time.time() - start_time
    print(f"\nnqubits = {nqubits}, solver = {solver}")
    print(f"trotter = {trotter}, accelerators = {accelerators}")
    print("Hamiltonians created in:", logs[-1]["hamiltonian_creation_time"])

    start_time = time.time()
    evolution = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt, solver=solver,
                                          accelerators=accelerators)
    logs[-1]["creation_time"] = time.time() - start_time
    print("Evolution model created in:", logs[-1]["creation_time"])

    start_time = time.time()
    final_psi = evolution(final_time=1.0)
    logs[-1]["simulation_time"] = time.time() - start_time
    print("Simulation time:", logs[-1]["simulation_time"])

    if filename is not None:
        with open(filename, "w") as file:
            json.dump(logs, file)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["accelerators"] = parse_accelerators(args.pop("accelerators"))
    main(**args)
