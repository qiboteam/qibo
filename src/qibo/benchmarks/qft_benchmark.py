import argparse
import os
import time
import numpy as np
import h5py
from qibo import gates, models
from typing import Dict, List, Optional


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default="3-10", type=str)
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--log-dir", default=None, type=str)
parser.add_argument("--compile", action="store_true")


def update_file(file_path: str, logs: Dict[str, List]):
    file = h5py.File(file_path, "w")
    for k, v in logs.items():
        file[k] = np.array(v)
    file.close()


def random_state(nqubits: int) -> np.ndarray:
    """Generates a random state."""
    x = np.random.random(2**nqubits) + 1j * np.random.random(2**nqubits)
    return x / np.sqrt((np.abs(x)**2).sum())


def parse_nqubits(nqubits_str: str) -> List[int]:
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


def main(nqubits_list: List[int],
         name: Optional[str] = None,
         log_dir: Optional[str] = None,
         compile: bool = False):
    if log_dir is not None:
        if name is None:
            raise ValueError("A run name should be given in order to save "
                             "logs.")

        # Generate log file name
        log_name = [__file__, name]
        if compile:
            log_name.append("compiled")
        log_name = "{}.h5".format("_".join(log_name))
        # Generate log file path
        file_path = os.path.join(log_dir, log_name)
        if os.path.exists(file_path):
            raise FileExistsError("File {} already exists in {}."
                                  "".format(log_name, log_dir))

        print("Saving logs in {}.".format(file_path))

    # Create log dict
    logs = {"nqubits": [], "simulation_time": []}
    if compile:
        logs["compile_time"] = []

    for nqubits in nqubits_list:
        # Generate a random initial state
        initial_state = random_state(nqubits)

        init_circuit = models.Circuit(nqubits)
        init_circuit.add(gates.Flatten(initial_state))
        circuit = init_circuit + models.QFTCircuit(nqubits)

        print("\nSimulating {} qubits...".format(nqubits))

        if compile:
            start_time = time.time()
            circuit.compile()
            logs["compile_time"].append(time.time() - start_time)

        start_time = time.time()
        final_state = circuit.execute()
        logs["simulation_time"].append(time.time() - start_time)

        logs["nqubits"].append(nqubits)

        # Write updated logs in file
        if log_dir is not None:
            update_file(file_path, logs)

        # Print results during run
        if compile:
            print("Compile time:", logs["compile_time"][-1])
        print("Simulation time:", logs["simulation_time"][-1])


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["nqubits_list"] = parse_nqubits(args.pop("nqubits"))
    main(**args)