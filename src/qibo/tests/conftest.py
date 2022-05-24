"""
conftest.py

Pytest fixtures.
"""
import sys
import pytest
from qibo.engines import construct_backend


ACTIVE_TESTS = {
    "qibo.tests.test_gates_abstract",
    "qibo.tests.test_gates_channels",
    "qibo.tests.test_gates_gates",
    "qibo.tests.test_gates_density_matrix",
    "qibo.tests.test_gates_special",
    "qibo.tests.test_models_circuit_execution",
    "qibo.tests.test_models_circuit_features",
    "qibo.tests.test_models_circuit_fuse",
    "qibo.tests.test_models_circuit_parametrized",
    "qibo.tests.test_models_circuit_qasm",
    "qibo.tests.test_models_circuit",
    "qibo.tests.test_models_qft",
    "qibo.tests.test_simulators"
}


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(
        mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip("Cannot run test on platform {}.".format(plat))


def pytest_configure(config):
    config.addinivalue_line("markers", "linux: mark test to run only on linux")


@pytest.fixture
def backend(backend_name):
    yield construct_backend(backend_name, show_error=True)
    

def pytest_generate_tests(metafunc):
    module_name = metafunc.module.__name__
    if module_name not in ACTIVE_TESTS:
        pytest.skip()

    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", ["numpy"])

    if "accelerators" in metafunc.fixturenames:
        metafunc.parametrize("accelerators", [None])