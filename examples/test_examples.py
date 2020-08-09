import os
import pytest
import sys
import importlib
import signal
from contextlib import contextmanager
base_dir = os.path.join(os.getcwd(), "examples")
sys.path.append(base_dir)
TIMEOUT = 3


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


def get_code(script_name="main.py"):
    code = open(script_name, "r").read()
    end = code.find('\nif __name__ ==')
    return code[:end] + '\n\nmain(**args)'


@pytest.mark.parametrize("N", [3])
@pytest.mark.parametrize("p", [0, 0.001])
@pytest.mark.parametrize("shots", [100])
@pytest.mark.parametrize("post_selection", [True])
def test_3_tangle(N, p, shots, post_selection):
    path = os.path.join(base_dir, "3_tangle")
    sys.path.append(path)
    os.chdir(path)
    main = importlib.import_module("3_tangle.main")
    with timeout(TIMEOUT):
        main.main(N, p, shots, post_selection, True)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("maxsteps", [5])
@pytest.mark.parametrize("T_max", [2])
def test_aavqe(nqubits, layers, maxsteps, T_max):
    os.chdir(os.path.join(base_dir, "aavqe"))
    from aavqe import main
    with timeout(TIMEOUT):
        main.main(nqubits, layers, maxsteps, T_max)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("compress", [1])
@pytest.mark.parametrize("lambdas", [[0.9, 0.95, 1.0, 1.05, 1.10]])
def test_autoencoder(nqubits, layers, compress, lambdas):
    os.chdir(os.path.join(base_dir, "autoencoder"))
    from autoencoder import main
    with timeout(TIMEOUT):
        main.main(nqubits, layers, compress, lambdas)


@pytest.mark.parametrize("nqubits", [4, 8])
def test_grover3sat(nqubits):
    path = os.path.join(base_dir, "grover3sat")
    sys.path[-1] = path
    os.chdir(path)
    from grover3sat import main
    with timeout(TIMEOUT):
        main.main(f"n{nqubits}.txt")


@pytest.mark.parametrize(("h_value", "collisions", "b"),
                         [(163, 2, 7)])
def test_hash_grover(h_value, collisions, b):
    # remove ``functions`` module from 3SAT because the same name is used
    # for a different module in the Hash
    del sys.modules["functions"]
    path = os.path.join(base_dir, "hash-grover")
    sys.path[-1] = path
    os.chdir(path)
    main = importlib.import_module("hash-grover.main")
    with timeout(TIMEOUT):
        main.main(h_value, collisions, b)


@pytest.mark.skip
@pytest.mark.parametrize(("nqubits", "subsize"), [(3, 1), (4, 2)])
@pytest.mark.parametrize("nlayers", [1, 2])
@pytest.mark.parametrize("nshots", [1000])
@pytest.mark.parametrize("RY", [False, True])
def test_qsvd(nqubits, subsize, nlayers, nshots, RY, method="Powell"):
    path = os.path.join(base_dir, "qsvd")
    os.chdir(path)
    main = importlib.import_module("qsvd.main")
    with timeout(TIMEOUT):
        main.main(nqubits, subsize, nlayers, nshots, RY, method)


@pytest.mark.skip
@pytest.mark.parametrize("dataset", ["tricrown", "circle", "square"])
@pytest.mark.parametrize("layers", [2, 3])
def test_reuploading_classifier(dataset, layers):
    path = os.path.join(base_dir, "reuploading_classifier")
    sys.path[-1] = path
    os.chdir(path)
    from reuploading_classifier import main
    with timeout(TIMEOUT):
        main.main(dataset, layers)


@pytest.mark.parametrize("data", [(2, 0.4, 0.05, 0.1, 1.9)])
@pytest.mark.parametrize("bins", [8, 16])
def test_unary(data, bins, M=10, shots=1000):
    del sys.modules["functions"]
    path = os.path.join(base_dir, "unary")
    sys.path[-1] = path
    os.chdir(path)
    from unary import main
    with timeout(TIMEOUT):
        main.main(data, bins, M, shots)


@pytest.mark.skip
@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("circuit_type", ["qft", "variational"])
def test_benchmarks(nqubits, circuit_type):
    path = os.path.join(base_dir, "benchmarks")
    sys.path[-1] = path
    os.chdir(path)
    from benchmarks import main
    with timeout(TIMEOUT):
        main.main(nqubits, circuit_type)
