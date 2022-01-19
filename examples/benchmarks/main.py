"""
Generic benchmark script that runs circuits defined in `benchmark_models.py`.

The type of the circuit is selected using the ``--type`` flag.
"""
import argparse
import os
import time
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # disable Tensorflow warnings


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=20, type=int)
parser.add_argument("--circuit", default="qft", type=str)
parser.add_argument("--backend", default="qibojit", type=str)
parser.add_argument("--precision", default="double", type=str)
parser.add_argument("--nreps", default=1, type=int)
parser.add_argument("--nshots", default=None, type=int)
parser.add_argument("--transfer", action="store_true")
parser.add_argument("--fuse", action="store_true")

parser.add_argument("--device", default=None, type=str)
parser.add_argument("--accelerators", default=None, type=str)
parser.add_argument("--memory", default=None, type=int)
parser.add_argument("--threading", default=None, type=str)
parser.add_argument("--compile", action="store_true")

parser.add_argument("--nlayers", default=None, type=int)
parser.add_argument("--gate-type", default=None, type=str)

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


def select_numba_threading(threading):
    from numba import config
    print(f"\nSwitching threading to {threading}.\n")
    config.THREADING_LAYER = threading


threading = args.pop("threading")
if args.get("backend") == "qibojit" and threading is not None:
    select_numba_threading(threading)

memory = args.pop("memory")
if args.get("backend") in {"qibotf", "tensorflow"}:
    limit_gpu_memory(memory)

import qibo
import circuits
from utils import BenchmarkLogger, parse_accelerators


def get_active_branch_name():
    """Returns the name of the active git branch."""
    from pathlib import Path
    qibo_dir = Path(qibo.__file__).parent.parent.parent
    head_dir = qibo_dir / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()
    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def main(nqubits, circuit_name, backend="custom", precision="double",
         nreps=1, nshots=None, transfer=False, fuse=False,
         device=None, accelerators=None, threadsafe=False,
         compile=False, get_branch=True, nlayers=None, gate_type=None,
         params={}, filename=None):
    """Runs circuit simulation benchmarks for different circuits.

    See benchmark documentation for a description of arguments.
    """
    qibo.set_backend(backend)
    qibo.set_precision(precision)
    if device is not None:
        qibo.set_device(device)

    logs = BenchmarkLogger(filename)
    # Create log dict
    logs.append({
        "nqubits": nqubits, "circuit_name": circuit_name, "threading": "",
        "backend": qibo.get_backend(), "precision": qibo.get_precision(),
        "device": qibo.get_device(), "accelerators": accelerators,
        "nshots": nshots, "transfer": transfer,
        "fuse": fuse, "compile": compile,
        })
    if get_branch:
        logs[-1]["branch"] = get_active_branch_name()

    params = {k: v for k, v in params.items() if v is not None}
    kwargs = {"nqubits": nqubits, "circuit_name": circuit_name}
    if params: kwargs["params"] = params
    if nlayers is not None: kwargs["nlayers"] = nlayers
    if gate_type is not None: kwargs["gate_type"] = gate_type
    if accelerators is not None:
        kwargs["accelerators"] = accelerators
    logs[-1].update(kwargs)

    start_time = time.time()
    circuit = circuits.CircuitFactory(**kwargs)
    if nshots is not None:
        # add measurement gates
        circuit.add(qibo.gates.M(*range(nqubits)))
    if fuse:
        circuit = circuit.fuse()
    logs[-1]["creation_time"] = time.time() - start_time

    start_time = time.time()
    if compile:
        circuit.compile()
        # Try executing here so that compile time is not included
        # in the simulation time
        result = circuit(nshots=nshots)
        del(result)
    logs[-1]["compile_time"] = time.time() - start_time

    start_time = time.time()
    result = circuit(nshots=nshots)
    logs[-1]["dry_run_time"] = time.time() - start_time
    start_time = time.time()
    if transfer:
        result = result.numpy()
    logs[-1]["dry_run_transfer_time"] = time.time() - start_time
    del(result)


    simulation_times, transfer_times = [], []
    for _ in range(nreps):
        start_time = time.time()
        result = circuit(nshots=nshots)
        simulation_times.append(time.time() - start_time)
        start_time = time.time()
        if transfer:
            result = result.numpy()
        transfer_times.append(time.time() - start_time)
        logs[-1]["dtype"] = str(result.dtype)
        if nshots is None:
            del(result)

    logs[-1]["simulation_times"] = simulation_times
    logs[-1]["transfer_times"] = transfer_times
    logs[-1]["simulation_times_mean"] = np.mean(simulation_times)
    logs[-1]["simulation_times_std"] = np.std(simulation_times)
    logs[-1]["transfer_times_mean"] = np.mean(transfer_times)
    logs[-1]["transfer_times_std"] = np.std(transfer_times)

    start_time = time.time()
    if nshots is not None:
        freqs = result.frequencies()
    logs[-1]["measurement_time"] = time.time() - start_time

    if logs[-1]["backend"] == "qibojit" and qibo.K.platform.name == "numba":
        from numba import threading_layer
        logs[-1]["threading"] = threading_layer()

    print()
    print(logs)
    print()
    logs.dump()


if __name__ == "__main__":
    args["circuit_name"] = args.pop("circuit")
    args["accelerators"] = parse_accelerators(args.pop("accelerators"))
    args["params"] = {k: args.pop(k) for k in _PARAM_NAMES}
    main(**args)
