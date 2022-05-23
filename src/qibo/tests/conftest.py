"""
conftest.py

Pytest fixtures.
"""
import sys
import pytest
from qibo.engines import construct_backend


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
    active_tests = {
        "qibo.tests.test_abstract_circuit",
        "qibo.tests.test_abstract_circuit_qasm",
        "qibo.tests.test_gates_abstract",
        "qibo.tests.test_gates_gates",
        "qibo.tests.test_gates_special",
        "qibo.tests.test_circuit_fuse",
        "qibo.tests.test_simulators"
    }
    module_name = metafunc.module.__name__
    if module_name not in active_tests:
        pytest.skip()

    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", ["numpy"])