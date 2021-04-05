"""
Generic benchmark script that runs circuits defined in `benchmark_models.py`.

The type of the circuit is selected using the ``--type`` flag.
"""
import argparse
import os
import time
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # disable Tensorflow warnings


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=20, type=int)
parser.add_argument("--type", default="qft", type=str)
parser.add_argument("--backend", default="custom", type=str)
parser.add_argument("--precision", default="double", type=str)

parser.add_argument("--device", default=None, type=str)
parser.add_argument("--accelerators", default=None, type=str)

parser.add_argument("--nshots", default=None, type=int)
parser.add_argument("--fuse", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--nlayers", default=None, type=int)
parser.add_argument("--gate-type", default=None, type=str)

parser.add_argument("--memory", default=None, type=int)
parser.add_argument("--filename", default=None, type=str)

# params
_PARAM_NAMES = {"theta", "phi"}
parser.add_argument("--theta", default=None, type=float)
parser.add_argument("--phi", default=None, type=float)
args = vars(parser.parse_args())


def limit_gpu_memory(memory_limit=None):
    """Limits GPU memory that is available to Tensorflow.

    Args:
        memory_limit: Memory limit in MBs.
    """
    import tensorflow as tf
    if memory_limit is None:
        print("\nNo GPU memory limiter used.\n")
        return

    print("\nAttempting to limit GPU memory to {}.\n".format(memory_limit))
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in tf.config.list_physical_devices("GPU"):
        config = tf.config.experimental.VirtualDeviceConfiguration(
                      memory_limit=memory_limit)
        tf.config.experimental.set_virtual_device_configuration(gpu, [config])
        print("Limiting memory of {} to {}.".format(gpu.name, memory_limit))
    print()

limit_gpu_memory(args.pop("memory"))

import qibo
import circuits


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


def main(nqubits, type,
         backend="custom", precision="double",
         device=None, accelerators=None,
         nshots=None, fuse=False, compile=False,
         nlayers=None, gate_type=None, params={},
         filename=None):
    """Runs benchmarks for different circuit types.

    Args:
        nqubits (int): Number of qubits in the circuit.
        type (str): Type of Circuit to use.
            See ``benchmark_models.py`` for available types.
        device (str): Tensorflow logical device to use for the benchmark.
            If ``None`` the first available device is used.
        accelerators (dict): Dictionary that specifies the accelarator devices
            for multi-GPU setups.
        nshots (int): Number of measurement shots.
            Logs the time required to sample frequencies (no samples).
            If ``None`` no measurements are performed.
        fuse (bool): If ``True`` gate fusion is used for faster circuit execution.
        compile: If ``True`` then the Tensorflow graph is compiled using
            ``circuit.compile()``. Compilation time is logged in this case.
        nlayers (int): Number of layers for supremacy-like or gate circuits.
            If a different circuit is used ``nlayers`` is ignored.
        gate_type (str): Type of gate for gate circuits.
            If a different circuit is used ``gate_type`` is ignored.
        params (dict): Gate parameter for gate circuits.
            If a non-parametrized circuit is used then ``params`` is ignored.
        filename (str): Name of file to write logs.
            If ``None`` logs will not be saved.
    """
    qibo.set_backend(args.pop("backend"))
    qibo.set_precision(args.pop("precision"))
    if device is not None:
        qibo.set_device(device)

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

    # Create log dict
    logs.append({
        "nqubits": nqubits, "circuit_type": type,
        "backend": qibo.get_backend(), "precision": qibo.get_precision(),
        "device": qibo.get_device(), "accelerators": accelerators,
        "nshots": nshots, "fuse": fuse, "compile": compile
        })

    params = {k: v for k, v in params.items() if v is not None}
    kwargs = {"nqubits": nqubits, "circuit_type": type}
    if params: kwargs["params"] = params
    if nlayers is not None: kwargs["nlayers"] = nlayers
    if gate_type is not None: kwargs["gate_type"] = gate_type
    if accelerators is not None:
        kwargs["accelerators"] = accelerators
        kwargs["device"] = device
    logs[-1].update(kwargs)

    start_time = time.time()
    circuit = circuits.CircuitFactory(**kwargs)
    if nshots is not None:
        # add measurement gates
        circuit.add(qibo.gates.M(*range(nqubits)))
    if fuse:
        circuit = circuit.fuse()
    logs[-1]["creation_time"] = time.time() - start_time

    if compile:
        start_time = time.time()
        circuit.compile()
        # Try executing here so that compile time is not included
        # in the simulation time
        result = circuit(nshots=nshots)
        logs[-1]["compile_time"] = time.time() - start_time

    start_time = time.time()
    result = circuit(nshots=nshots)
    logs[-1]["simulation_time"] = time.time() - start_time
    logs[-1]["dtype"] = str(result.dtype)

    if nshots is not None:
        start_time = time.time()
        freqs = result.frequencies()
        logs[-1]["measurement_time"] = time.time() - start_time

    print()
    for k, v in logs[-1].items():
        print("{}: {}".format(k, v))

    if filename is not None:
        with open(filename, "w") as file:
            json.dump(logs, file)


if __name__ == "__main__":
    args["accelerators"] = parse_accelerators(args.pop("accelerators"))
    args["params"] = {k: args.pop(k) for k in _PARAM_NAMES}
    main(**args)
