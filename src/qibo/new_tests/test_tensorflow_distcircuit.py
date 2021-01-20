import inspect
import pytest
from qibo.config import raise_error
from qibo.new_tests import test_core_circuit, test_core_circuit_parametrized


ACCELERATORS = [{"/GPU:0": 2}, {"/GPU:0": 1, "/GPU:1": 1},
                {"/GPU:0": 2, "/GPU:1":1, "/GPU:2": 1}]


# TODO: Add tests for :class:`qibo.tensorflow.distcircuit.DistributedCircuit`


# Rerun circuit tests from other test files for distributed circuits
def get_names(arg_names=["backend", "accelerators"],
              modules=[test_core_circuit, test_core_circuit_parametrized]):
    for module in modules:
        for name in dir(module):
            # memory check because is tested seperately to include the
            # `pytest.mark.linux` flag
            if name[:4] == "test" and name != "test_memory_error":
                func = getattr(module, name)
                fargs = inspect.getfullargspec(func).args
                if fargs == arg_names:
                    yield func


@pytest.mark.linux
@pytest.mark.parametrize("accelerators", ACCELERATORS)
def test_memory_error(accelerators):
    test_core_circuit.test_memory_error("custom", accelerators)


@pytest.mark.parametrize("testfunc", get_names(["backend", "accelerators"]))
@pytest.mark.parametrize("accelerators", ACCELERATORS)
def test_distributed(testfunc, accelerators):
    testfunc("custom", accelerators)


@pytest.mark.parametrize("testfunc", get_names(["backend", "nqubits", "accelerators"]))
@pytest.mark.parametrize("nqubits", [4, 5, 6])
@pytest.mark.parametrize("accelerators", ACCELERATORS)
def test_distributed_with_nqubits(testfunc, nqubits, accelerators):
    testfunc("custom", nqubits, accelerators)


@pytest.mark.parametrize("testfunc", get_names(["backend", "trainable", "accelerators"]))
@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("accelerators", ACCELERATORS)
def test_distributed_with_trainable(testfunc, trainable, accelerators):
    testfunc("custom", trainable, accelerators)


@pytest.mark.parametrize("testfunc", get_names(["backend", "nqubits", "trainable", "accelerators"]))
@pytest.mark.parametrize("nqubits", [5, 6])
@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("accelerators", ACCELERATORS)
def test_distributed_with_nqubits_and_trainable(testfunc, nqubits, trainable, accelerators):
    testfunc("custom", nqubits, trainable, accelerators)
