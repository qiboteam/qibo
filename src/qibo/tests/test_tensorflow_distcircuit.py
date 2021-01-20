import inspect
import pytest
from qibo.tests import test_core_circuit


ACCELERATORS = [{"/GPU:0": 2}, {"/GPU:0": 1, "/GPU:1": 1},
                {"/GPU:0": 2, "/GPU:1":1, "/GPU:2": 1}]
TESTNAMES = [name for name in dir(test_core_circuit) if name[:4] == "test"]
# remove memory check because we will do it seperately to include the
# `pytest.mark.linux` flag
TESTNAMES.remove("test_memory_error")

# TODO: Add tests for :class:`qibo.tensorflow.distcircuit.DistributedCircuit`

# Re-run `test_core_circuit.py` tests for distributed circuits
@pytest.mark.parametrize("name", TESTNAMES)
@pytest.mark.parametrize("accelerators", ACCELERATORS)
def test_distributed(name, accelerators):
    func = getattr(test_core_circuit, name)
    funcargs = inspect.getfullargspec(func).args
    if funcargs == ["backend", "accelerators"]:
        func("custom", accelerators)
    else:
        pytest.skip("Skipping {} because it is not relevant for distributed "
                    "circuits.".format(name))


@pytest.mark.linux
@pytest.mark.parametrize("accelerators", ACCELERATORS)
def test_memory_error(accelerators):
    test_core_circuit.test_memory_error("custom", accelerators)
