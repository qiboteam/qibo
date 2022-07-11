"""
conftest.py

Pytest fixtures.
"""
import sys
import pytest
from qibo.backends import construct_backend


# backends to be tested
BACKENDS = ["numpy", "tensorflow", "qibojit-numba", "qibojit-cupy", "qibojit-cuquantum"]
# multigpu configurations to be tested (only with qibojit-cupy)
ACCELERATORS = [{"/GPU:0": 1, "/GPU:1": 1}, {"/GPU:0": 2, "/GPU:1": 2},
                {"/GPU:0": 1, "/GPU:1": 1, "/GPU:2": 1, "/GPU:3": 1}]


def get_backend(backend_name):
    if "-" in backend_name:
        name, platform = backend_name.split("-")
    else:
        name, platform = backend_name, None
    return construct_backend(name, platform=platform)

# ignore backends that are not available in the current testing environment
AVAILABLE_BACKENDS = []
MULTIGPU_BACKENDS = []
for backend_name in BACKENDS:
    try:
        _backend = get_backend(backend_name)
        AVAILABLE_BACKENDS.append(backend_name)
        if _backend.supports_multigpu:  # pragma: no cover
            MULTIGPU_BACKENDS.append(backend_name)
    except (ModuleNotFoundError, ImportError):
        pass


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
    yield get_backend(backend_name)


def pytest_generate_tests(metafunc):
    module_name = metafunc.module.__name__

    if module_name == "qibo.tests.test_models_qgan" and "tensorflow" not in AVAILABLE_BACKENDS:  # pragma: no cover
        pytest.skip("Skipping QGAN tests because tensorflow is not available.")

    if module_name == "qibo.tests.test_models_distcircuit_execution":
        config = [(bk, acc) for acc in ACCELERATORS for bk in MULTIGPU_BACKENDS]
        metafunc.parametrize("backend_name,accelerators", config)

    else:
        if "backend_name" in metafunc.fixturenames:
            if "accelerators" in metafunc.fixturenames:
                config = [(backend, None) for backend in AVAILABLE_BACKENDS]
                config.extend((bk, acc) for acc in ACCELERATORS for bk in MULTIGPU_BACKENDS)
                metafunc.parametrize("backend_name,accelerators", config)
            else:
                metafunc.parametrize("backend_name", AVAILABLE_BACKENDS)

        elif "accelerators" in metafunc.fixturenames:
            metafunc.parametrize("accelerators", ACCELERATORS)
