import os
import pytest
import sys
import importlib
import signal
from contextlib import contextmanager
base_dir = os.path.join(os.getcwd(), "examples")
sys.path.append(base_dir)


@pytest.fixture(autouse=True)
def max_time_setter(request):
    request.function.__globals__['max_time'] = request.config.getoption(
        "--examples-timeout")


@contextmanager
def timeout(time):
    """Auxiliary timeout method. Register an alarm for a given
    input time. This function provides a silent timeout error.

    Args:
        time (int): timeout seconds.
    """
    # register signal
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        pass
    finally:
        # unregister signal
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


def run_script(args, script_name="main.py"):
    """Executes external Python script with given arguments.

    Args:
        args (dict): Dictionary with arguments required by the script's main
            function.
        script_name (str): Name of the script file.
        max_time (float): Time-out time in seconds.
    """
    import qibo
    qibo.set_backend("custom")
    code = open(script_name, "r").read()
    end = code.find("\nif __name__ ==")
    code = code[:end] + "\n\nmain(**args)"
    with timeout(max_time):
        exec(code, {"args": args})


@pytest.mark.parametrize("N", [3])
@pytest.mark.parametrize("p", [0, 0.001])
def test_3_tangle(N, p, shots=100, post_selection=True, no_plot=True):
    args = locals()
    path = os.path.join(base_dir, "3_tangle")
    sys.path.append(path)
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("maxsteps", [5])
@pytest.mark.parametrize("T_max", [2])
def test_aavqe(nqubits, layers, maxsteps, T_max):
    args = locals()
    os.chdir(os.path.join(base_dir, "aavqe"))
    run_script(args)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("compress", [1])
@pytest.mark.parametrize("lambdas", [[0.9, 0.95, 1.0, 1.05, 1.10]])
def test_autoencoder(nqubits, layers, compress, lambdas):
    args = locals()
    os.chdir(os.path.join(base_dir, "autoencoder"))
    run_script(args)


@pytest.mark.parametrize("nqubits", [4, 8])
def test_grover3sat(nqubits):
    args = {"file_name": f"n{nqubits}.txt"}
    path = os.path.join(base_dir, "grover3sat")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize(("h_value", "collisions", "b"),
                         [(163, 2, 7)])
def test_hash_grover(h_value, collisions, b):
    # remove ``functions`` module from 3SAT because the same name is used
    # for a different module in the Hash
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "hash-grover")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize(("nqubits", "subsize"), [(3, 1), (4, 2)])
@pytest.mark.parametrize("nlayers", [1, 2])
@pytest.mark.parametrize("nshots", [1000])
@pytest.mark.parametrize("RY", [False, True])
def test_qsvd(nqubits, subsize, nlayers, nshots, RY, method="Powell"):
    args = locals()
    path = os.path.join(base_dir, "qsvd")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("dataset", ["circle", "square"])
@pytest.mark.parametrize("layers", [2, 3])
def test_reuploading_classifier(dataset, layers):
    args = locals()
    path = os.path.join(base_dir, "reuploading_classifier")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("data", [(2, 0.4, 0.05, 0.1, 1.9)])
@pytest.mark.parametrize("bins", [8, 16])
def test_unary(data, bins, M=10, shots=1000):
    args = locals()
    if "functions" in sys.modules:
        del sys.modules["functions"]
    path = os.path.join(base_dir, "unary")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("nqubits_list", [[3, 4]])
@pytest.mark.parametrize("type", ["qft", "variational"])
def test_benchmarks(nqubits_list, type):
    args = locals()
    path = os.path.join(base_dir, "benchmarks")
    sys.path[-1] = path
    os.chdir(path)
    code = open("main.py", "r").read()
    start = code.find("def main")
    end = code.find("\nif __name__ ==")
    header = ("import argparse\nimport os\nimport time"
              "\nfrom typing import Dict, List, Optional"
              "\nimport tensorflow as tf"
              "\nimport qibo\nimport circuits\nimport utils\n\n")
    import qibo
    qibo.set_backend("custom")
    code = header + code[start: end] + "\n\nmain(**args)"
    with timeout(max_time):
        exec(code, {"args": args})


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("nlayers", [1, 2])
@pytest.mark.parametrize("varlayer", [False, True])
def test_vqe_benchmarks(nqubits, nlayers, varlayer, method="Powell"):
    args = locals()
    path = os.path.join(base_dir, "benchmarks")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="vqe.py")


@pytest.mark.parametrize("nclasses", [3])
@pytest.mark.parametrize("nqubits", [4])
@pytest.mark.parametrize("nlayers", [4, 5])
@pytest.mark.parametrize("nshots", [int(1e5)])
@pytest.mark.parametrize("training", [False])
@pytest.mark.parametrize("RxRzRx", [False])
def test_variational_classifier(nclasses, nqubits, nlayers,
                                nshots, training, RxRzRx, method='Powell'):
    args = locals()
    path = os.path.join(base_dir, "variational_classifier")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)
