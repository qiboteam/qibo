"""Adiabatic evolution for the Ising Hamiltonian using linear scaling."""
import argparse
import time
import utils
from qibo import callbacks, hamiltonians, models

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default="4", type=str)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)
parser.add_argument("--trotter", action="store_true")
parser.add_argument("--accelerators", default=None, type=str)


def main(nqubits_list, dt, solver, trotter=False, accelerators=None):
    """Performs adiabatic evolution with critical TFIM as the "hard" Hamiltonian."""
    if accelerators is not None:
        trotter = True
        solver = "exp"

    print(f"Using {solver} solver and dt = {dt}.")
    print(f"Accelerators: {accelerators}")

    for nqubits in nqubits_list:
        start_time = time.time()
        h0 = hamiltonians.X(nqubits, trotter=trotter)
        h1 = hamiltonians.TFIM(nqubits, h=1.0, trotter=trotter)
        ham_creation_time = time.time() - start_time
        print(f"\nnqubits = {nqubits}, solver = {solver}")
        print(f"trotter = {trotter}, accelerators = {accelerators}")
        print("Hamiltonians created in:", ham_creation_time)

        start_time = time.time()
        evolution = models.AdiabaticEvolution(h0, h1, lambda t: t, dt=dt,
                                              solver=solver,
                                              accelerators=accelerators)
        creation_time = time.time() - start_time
        print("Evolution model created in:", creation_time)

        start_time = time.time()
        final_psi = evolution(final_time=1.0)
        simulation_time = time.time() - start_time
        print("Simulation time:", simulation_time)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["nqubits_list"] = utils.parse_nqubits(args.pop("nqubits"))
    args["accelerators"] = utils.parse_accelerators(args.pop("accelerators"))
    main(**args)
