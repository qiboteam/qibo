import os
import pytest
import sys
import importlib
base_dir = "/".join(os.getcwd().split("/")[:-3] + ["examples"])
sys.path.append(base_dir)
TIMEOUT = 3


@pytest.mark.xfail
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("N", [3])
@pytest.mark.parametrize("p", [0, 0.001])
@pytest.mark.parametrize("shots", [100])
@pytest.mark.parametrize("post_selection", [True])
def test_3_tangle(N, p, shots, post_selection):
    path = os.path.join(base_dir, "3_tangle")
    sys.path.append(path)
    os.chdir(path)
    main = importlib.import_module("3_tangle.main")
    main.main(N, p, shots, post_selection)


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("maxsteps", [5])
@pytest.mark.parametrize("T_max", [2])
def test_aavqe(nqubits, layers, maxsteps, T_max):
    os.chdir(os.path.join(base_dir, "aavqe"))
    from aavqe import main
    main.main(nqubits, layers, maxsteps, T_max)


@pytest.mark.skip("Fails because the global variable 'count' is used.")
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("compress", [1])
@pytest.mark.parametrize("lambdas", [[0.9, 0.95, 1.0, 1.05, 1.10]])
def test_autoencoder(nqubits, layers, compress, lambdas):
    os.chdir(os.path.join(base_dir, "autoencoder"))
    from autoencoder import main
    main.main(nqubits, layers, compress, lambdas)


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("nqubits", [4, 8])
def test_grover3sat(nqubits):
    path = os.path.join(base_dir, "grover3sat")
    sys.path[-1] = path
    os.chdir(path)
    from grover3sat import main
    main.main(f"n{nqubits}.txt")


@pytest.mark.xfail
@pytest.mark.timeout(TIMEOUT)
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
    main.main(h_value, collisions, b)
