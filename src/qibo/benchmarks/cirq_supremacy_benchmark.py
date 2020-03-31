"""
Benchmark script for the Quantum Fourier Transform using `models.QFTCircuit`.
"""
import argparse
import os
import time
import utils
import cirq
import numpy as np
np.random.seed(1234)


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default="3-10", type=str)
parser.add_argument("--nlayers", default=5, type=int)
parser.add_argument("--nshots", default=10000, type=int)
parser.add_argument("--directory", default=None, type=str)
parser.add_argument("--name", default=None, type=str)


def SupremacyLikeCircuit(nqubits, nlayers):
    one_qubit_gates = [(cirq.X)**0.5, (cirq.Y)**0.5, (cirq.Z)**0.5]
    two_qubit_gate = cirq.CZPowGate(exponent=1.0/6.0)
    qubits = [cirq.LineQubit(i) for i in range(nqubits)]
    circuit = cirq.Circuit()
    d = 1
    for l in range(nlayers):
        for i in range(nqubits):
            gate = one_qubit_gates[np.random.randint(0, len(one_qubit_gates))]
            circuit.append(gate(qubits[i]))
        for i in range(nqubits):
            circuit.append(two_qubit_gate(qubits[i], qubits[(i + d) % nqubits]))
        d += 1
        if d > nqubits - 1:
            d = 1
    for i in range(nqubits):
        gate = one_qubit_gates[np.random.randint(0, len(one_qubit_gates))]
        circuit.append(gate(qubits[i]))
        circuit.append(cirq.measure(qubits[i]))
    return circuit


def main(nqubits_list, nlayers, nshots, directory=None, name=None):
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
        log_name = "{}.h5".format("_".join(log_name))
        # Generate log file path
        file_path = os.path.join(directory, log_name)
        if os.path.exists(file_path):
            raise FileExistsError("File {} already exists in {}."
                                  "".format(log_name, directory))

        print("Saving logs in {}.".format(file_path))

    # Create log dict
    logs = {"nqubits": [], "simulation_time": []}
    simulator = cirq.Simulator()
    for nqubits in nqubits_list:
        print("\nSimulating {} qubits with {} layers...".format(nqubits, nlayers))
        circuit = SupremacyLikeCircuit(nqubits, nlayers)

        start_time = time.time()
        results = simulator.run(circuit, repetitions=nshots)
        logs["simulation_time"].append(time.time() - start_time)
        logs["nqubits"].append(nqubits)

        # Write updated logs in file
        if directory is not None:
            utils.update_file(file_path, logs)

        # Print results during run
        print("Simulation time:", logs["simulation_time"][-1])


if __name__ == "__main__":
    args = vars(parser.parse_args())
    args["nqubits_list"] = utils.parse_nqubits(args.pop("nqubits"))
    main(**args)
