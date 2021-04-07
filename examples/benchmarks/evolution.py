"""Adiabatic evolution for the Ising Hamiltonian using linear scaling."""
import argparse
import time
from qibo import callbacks, hamiltonians, models

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default="4", type=str)
parser.add_argument("--dt", default=1e-2, type=float)
parser.add_argument("--solver", default="exp", type=str)
parser.add_argument("--trotter", action="store_true")
parser.add_argument("--accelerators", default=None, type=str)


def parse_nqubits(nqubits_str):
    """Transforms a string that specifies number of qubits to list.

    Supported string formats are the following:
        * 'a-b' with a and b integers.
            Then the returned list is range(a, b + 1).
        * 'a,b,c,d,...' with a, b, c, d, ... integers.
            Then the returned list is [a, b, c, d]
    """
    # TODO: Support usage of both `-` and `,` in the same string.
    if "-" in nqubits_str:
        if "," in nqubits_str:
            raise ValueError("String that specifies qubits cannot contain "
                             "both , and -.")

        nqubits_split = nqubits_str.split("-")
        if len(nqubits_split) != 2:
            raise ValueError("Invalid string that specifies nqubits "
                             "{}.".format(nqubits_str))

        n_start, n_end = nqubits_split
        return list(range(int(n_start), int(n_end) + 1))

    return [int(x) for x in nqubits_str.split(",")]


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
    args["nqubits_list"] = parse_nqubits(args.pop("nqubits"))
    args["accelerators"] = parse_accelerators(args.pop("accelerators"))
    main(**args)
