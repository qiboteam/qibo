"""conftest.py.

Pytest fixtures.
"""

import sys

import pytest

from qibo.backends import _Global, construct_backend

# backends to be tested
BACKENDS = [
    "numpy",
    "pytorch",
    "qibojit-numba",
    "qibojit-cupy",
    "qibojit-cuquantum",
    "qiboml-tensorflow",
]
# multigpu configurations to be tested (only with qibojit-cupy)
ACCELERATORS = [
    {"/GPU:0": 1, "/GPU:1": 1},
    {"/GPU:0": 2, "/GPU:1": 2},
    {"/GPU:0": 1, "/GPU:1": 1, "/GPU:2": 1, "/GPU:3": 1},
]


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
    except ImportError:
        pass

try:
    get_backend("qulacs")
    QULACS_INSTALLED = True
except ImportError:
    QULACS_INSTALLED = False


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip(f"Cannot run test on platform {plat}.")
    elif not QULACS_INSTALLED and item.fspath.purebasename == "test_backends_qulacs":
        # case not covered by workflows
        pytest.skip(f"Cannot test `qulacs` on platform {plat}.")


def pytest_configure(config):
    config.addinivalue_line("markers", "linux: mark test to run only on linux")


@pytest.fixture
def backend(backend_name):
    yield get_backend(backend_name)


@pytest.fixture(autouse=True)
def clear():
    yield
    _Global._backend = None
    _Global._transpiler = None


def pytest_generate_tests(metafunc):
    module_name = metafunc.module.__name__

    if module_name == "tests.test_models_distcircuit_execution":
        config = [(bk, acc) for acc in ACCELERATORS for bk in MULTIGPU_BACKENDS]
        metafunc.parametrize("backend_name,accelerators", config)

    else:
        if "backend_name" in metafunc.fixturenames:
            if "accelerators" in metafunc.fixturenames:
                config = [(backend, None) for backend in AVAILABLE_BACKENDS]
                config.extend(
                    (bk, acc) for acc in ACCELERATORS for bk in MULTIGPU_BACKENDS
                )
                metafunc.parametrize("backend_name,accelerators", config)
            else:
                metafunc.parametrize("backend_name", AVAILABLE_BACKENDS)

        elif "accelerators" in metafunc.fixturenames:
            metafunc.parametrize("accelerators", ACCELERATORS)
