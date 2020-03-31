"""
Benchmark script for the Quantum Fourier Transform using `models.QFTCircuit`.
"""
import argparse
import os
import time
import utils
import numpy as np
np.random.seed(1234)
from qibo import models, gates


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default="3-10", type=str)
parser.add_argument("--nlayers", default=5, type=int)
parser.add_argument("--nshots", default=None, type=int)
parser.add_argument("--directory", default=None, type=str)
parser.add_argument("--name", default=None, type=str)
parser.add_argument("--compile", action="store_true")


def SupremacyLikeCircuit(nqubits, nlayers):
    one_qubit_gates = ["RX", "RY", "RZ"]
    circuit = models.Circuit(nqubits)
    d = 1
    for l in range(nlayers):
        for i in range(nqubits):
            gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
            circuit.add(gate(i, 0.5))
        for i in range(nqubits):
            circuit.add(gates.CRZ(i, (i + d) % nqubits, 1.0/6.0))
        d += 1
        if d > nqubits - 1:
            d = 1
    for i in range(nqubits):
        gate = getattr(gates, one_qubit_gates[np.random.randint(0, len(one_qubit_gates))])
        circuit.add(gate(i, 0.5))
        circuit.add(gates.M(i))
    return circuit


def main(nqubits_list, nlayers, nshots=None, directory=None, name=None,
         compile=False):
    """Runs benchmarks for the Quantum Fourier Transform.

    If `directory` is specified this saves an `.h5` file that contains the
    following keys:
        * nqubits: List with the number of qubits that were simulated.
        * simulation_time: List with simulation times for each number of qubits.

    Args:
        nqubits_list: List with the number of qubits to run for.
        directory: Directory to save the log files.
            If `None` then logs are not saved.
        name: Name of the run to be used when saving logs.
            This should be specified if a directory in given. Otherwise it
            is ignored.

    Raises:
        FileExistsError if the file with the `name` specified exists in the
        given `directory`.
    """
    if directory is not None:
        if name is None:
            raise ValueError("A run name should be given in order to save "
                             "logs.")

        # Generate log file name
        log_name = [name]
        if compile:
            log_name.append("compiled")
        log_name = "{}.h5".format("_".join(log_name))
        # Generate log file path
        file_path = os.path.join(directory, log_name)
        if os.path.exists(file_path):
            raise FileExistsError("File {} already exists in {}."
                                  "".format(log_name, directory))

        print("Saving logs in {}.".format(file_path))

    # Create log dict
    logs = {"nqubits": [], "simulation_time": []}
    if compile:
        logs["compile_time"] = []
    for nqubits in nqubits_list:
        print("\nSimulating {} qubits with {} layers...".format(nqubits, nlayers))
        if nshots is not None:
            print("Performing measurements...")
        circuit = SupremacyLikeCircuit(nqubits, nlayers)

        if compile:
            start_time = time.time()
            circuit.compile()
            # Try executing here so that compile time is not included
            # in the simulation time
            results = circuit(nshots=nshots)
            logs["compile_time"].append(time.time() - start_time)

        start_time = time.time()
        results = circuit(nshots=nshots)
        logs["simulation_time"].append(time.time() - start_time)
        logs["nqubits"].append(nqubits)

        # Write updated logs in file
        if directory is not None:
            utils.update_file(file_path, logs)

        # Print results during run
        if compile:
            print("Compile time:", logs["compile_time"][-1])
        print("Simulation time:", logs["simulation_time"][-1])


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["nqubits_list"] = utils.parse_nqubits(args.pop("nqubits"))
    main(**args)
