"""
conftest.py

Pytest fixtures.
"""
import sys
import pytest

try:
    import tensorflow as tf
    _BACKENDS = "custom,defaulteinsum,matmuleinsum,"\
                "numpy_defaulteinsum,numpy_matmuleinsum"
    _ENGINES = "numpy,tensorflow"
    _ACCELERATORS = "2/GPU:0,1/GPU:0+1/GPU:1,2/GPU:0+1/GPU:1+1/GPU:2"
except ModuleNotFoundError: # pragma: no cover
    # workflows install Tensorflow automatically so this case is not covered
    _ENGINES = "numpy"
    _BACKENDS = "numpy_defaulteinsum,numpy_matmuleinsum"
    _ACCELERATORS = None


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(
        mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip("Cannot run test on platform {}.".format(plat))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "linux: mark test to run only on linux"
    )


def pytest_addoption(parser):
    parser.addoption("--engines", type=str, default=_ENGINES)
    parser.addoption("--backends", type=str, default=_BACKENDS)
    parser.addoption("--accelerators", type=str, default=_ACCELERATORS)
    parser.addoption("--target-backend", type=str, default="numpy")


def pytest_generate_tests(metafunc):
    engines = metafunc.config.option.engines.split(",")
    backends = metafunc.config.option.backends.split(",")
    accelerators = metafunc.config.option.accelerators
    if "tensorflow" not in engines: # pragma: no cover
        # CI uses Tensorflow engine for test execution
        accelerators = None
        for x in ["custom", "defaulteinsum", "matmuleinsum"]:
            if x in backends:
                backends.remove(x)

    # for `test_backends_matrices.py`
    if "engine" in metafunc.fixturenames:
        metafunc.parametrize("engine", engines)

    # for `test_backends_agreement.py`
    if "tested_backend" in metafunc.fixturenames:
        target = metafunc.config.option.target_backend
        metafunc.parametrize("tested_backend", [x for x in engines if x != target])
        metafunc.parametrize("target_backend", [target])

    # for `test_core_*.py`
    if "backend" in metafunc.fixturenames:
        if "accelerators" in metafunc.fixturenames:
            if accelerators is None: # pragma: no cover
                # `accelerators` is never run in CI test execution
                metafunc.parametrize("backend", backends)
                metafunc.parametrize("accelerators", [None])
            else:
                config = [(b, None) for b in backends]
                config.extend([("custom", {dev[1:]: int(dev[0]) for dev in x.split("+")})
                               for x in accelerators.split(",")])
                metafunc.parametrize("backend,accelerators", config)
        else:
            metafunc.parametrize("backend", backends)
