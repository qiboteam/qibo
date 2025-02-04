import os
import signal
import sys
from contextlib import contextmanager

import pytest

base_dir = os.path.join(os.getcwd(), "examples")
sys.path.append(base_dir)


@pytest.fixture(autouse=True)
def max_time_setter(request):
    request.function.__globals__["max_time"] = request.config.getoption(
        "--examples-timeout"
    )


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
    code = open(script_name).read()
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
@pytest.mark.parametrize("maxiter", [1])
def test_autoencoder(nqubits, layers, compress, lambdas, maxiter):
    args = locals()
    os.chdir(os.path.join(base_dir, "autoencoder"))
    run_script(args)


@pytest.mark.parametrize(("h_value", "collisions", "b"), [(163, 2, 7)])
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
def test_qsvd(nqubits, subsize, nlayers, nshots, RY, method="Powell", maxiter=1):
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
@pytest.mark.parametrize("bins", [8, 10])
def test_unary(data, bins, M=10, shots=1000):
    args = locals()
    if "functions" in sys.modules:
        del sys.modules["functions"]
    path = os.path.join(base_dir, "unary")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("nqubits", [3, 6])
@pytest.mark.parametrize("circuit_name", ["qft", "variational"])
def test_benchmarks(nqubits, circuit_name):
    path = os.path.join(base_dir, "benchmarks")
    sys.path[-1] = path
    os.chdir(path)
    code = open("main.py").read()
    start = code.find("def main")
    end = code.find("\nif __name__ ==")
    header = (
        "import argparse\nimport os\nimport time\nimport numpy as np"
        "\nimport qibo\nimport circuits\nfrom utils import "
        "BenchmarkLogger, parse_accelerators\n\n"
    )
    args = {
        "nqubits": nqubits,
        "circuit_name": circuit_name,
        "backend": "qibojit",
        "precision": "double",
        "device": None,
        "accelerators": None,
        "get_branch": False,
        "nshots": None,
        "fuse": False,
        "compile": False,
        "nlayers": None,
        "gate_type": None,
        "params": {},
        "filename": None,
    }
    code = header + code[start:end] + "\n\nmain(**args)"
    with timeout(max_time):
        exec(code, {"args": args})


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("nlayers", [1, 2])
@pytest.mark.parametrize("fuse", [False, True])
def test_vqe_benchmarks(nqubits, nlayers, fuse, method="Powell"):
    args = locals()
    args["backend"] = "qibojit"
    path = os.path.join(base_dir, "benchmarks")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="vqe.py")


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("nangles", [2])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("solver", ["exp", "rk4"])
def test_qaoa_benchmarks(nqubits, nangles, dense, solver, method="Powell"):
    args = locals()
    path = os.path.join(base_dir, "benchmarks")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="qaoa.py")


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("dt", [0.1, 0.01])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("solver", ["exp", "rk4"])
@pytest.mark.parametrize("backend", ["qibojit"])
def test_evolution_benchmarks(nqubits, dt, dense, solver, backend):
    args = locals()
    path = os.path.join(base_dir, "benchmarks")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="evolution.py")


@pytest.mark.parametrize("nclasses", [3])
@pytest.mark.parametrize("nqubits", [4])
@pytest.mark.parametrize("nlayers", [4, 5])
@pytest.mark.parametrize("nshots", [int(1e5)])
@pytest.mark.parametrize("training", [False])
@pytest.mark.parametrize("RxRzRx", [False])
def test_variational_classifier(
    nclasses, nqubits, nlayers, nshots, training, RxRzRx, method="Powell"
):
    args = locals()
    path = os.path.join(base_dir, "variational_classifier")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("nqubits", [4, 8])
@pytest.mark.parametrize("instance", [1])
def test_grover3sat(nqubits, instance):
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "grover3sat")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("nqubits,hfield,T,dt", [(4, 1, 10, 1e-1)])
@pytest.mark.parametrize("solver", ["exp", "rk4"])
def test_adiabatic_linear(nqubits, hfield, T, dt, solver, save=False):
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "adiabatic")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="linear.py")


@pytest.mark.parametrize("nqubits,hfield,dt, params", [(4, 1, 1e-2, [1.0])])
@pytest.mark.parametrize("solver", ["exp"])
@pytest.mark.parametrize("method", ["Powell"])
def test_adiabatic_optimize(
    nqubits, hfield, params, dt, solver, method, maxiter=None, save=False
):
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "adiabatic")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="optimize.py")


@pytest.mark.parametrize("nqubits,hfield,T", [(4, 1, 1)])
def test_adiabatic_trotter(nqubits, hfield, T, save=False):
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "adiabatic")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="trotter_error.py")


@pytest.mark.parametrize("nqubits,instance,T,dt", [(4, 1, 10, 1e-1)])
@pytest.mark.parametrize("solver", ["exp", "rk4"])
@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("params", [[0.5, 0.5]])
@pytest.mark.parametrize("method,maxiter", [("BFGS", 1)])
def test_adiabatic3sat(
    nqubits, instance, T, dt, solver, dense, params, method, maxiter, plot=False
):
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "adiabatic3sat")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("layers", [3, 2])
@pytest.mark.parametrize("autoencoder", [0, 1])
@pytest.mark.parametrize("example", [0, 1])
@pytest.mark.parametrize("maxiter", [1])
def test_ef_qae(layers, autoencoder, example, maxiter):
    args = locals()
    os.chdir(os.path.join(base_dir, "EF_QAE"))
    run_script(args)


@pytest.mark.parametrize("N", [15, 21])
@pytest.mark.parametrize("times", [2, 10])
@pytest.mark.parametrize("A", [None])
@pytest.mark.parametrize("semiclassical", [True, False])
@pytest.mark.parametrize("enhance", [True, False])
def test_shor(N, times, A, semiclassical, enhance):
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "shor")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("delta_t", [0.5, 0.1])
@pytest.mark.parametrize("max_layers", [10, 100])
def test_falqon(nqubits, delta_t, max_layers):
    if "functions" in sys.modules:
        del sys.modules["functions"]
    args = locals()
    path = os.path.join(base_dir, "falqon")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args)


@pytest.mark.parametrize("nqubits", [5, 6, 7])
def test_grover_example1(nqubits):
    args = locals()
    path = os.path.join(base_dir, "grover")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="example1.py")


@pytest.mark.parametrize("nqubits", [5, 8, 10])
@pytest.mark.parametrize("num_1", [1, 2])
@pytest.mark.parametrize("iterative", [False, True])
def test_grover_example2(nqubits, num_1, iterative):
    args = locals()
    path = os.path.join(base_dir, "grover")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="example2.py")


@pytest.mark.parametrize("nqubits", [5, 8, 10])
@pytest.mark.parametrize("num_1", [1, 2])
def test_grover_example3(nqubits, num_1):
    args = locals()
    path = os.path.join(base_dir, "grover")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="example3.py")


@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("batch_size", [20])
@pytest.mark.parametrize("nepochs", [7])
@pytest.mark.parametrize("train_size", [100])
@pytest.mark.parametrize("filename", ["parameters/test_params"])
@pytest.mark.parametrize("lr_boundaries", [[1, 2, 3, 4, 5, 6]])
def test_anomalydetection_train(
    n_layers, batch_size, nepochs, train_size, filename, lr_boundaries
):
    args = locals()
    path = os.path.join(base_dir, "anomaly_detection")
    sys.path[-1] = path
    os.chdir(path)
    run_script(args, script_name="train.py")
