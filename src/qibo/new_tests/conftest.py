"""
conftest.py

Pytest fixtures.
"""
import sys
import pytest

_ENGINES = ["numpy", "tensorflow"]
try:
    import tensorflow as tf
    _BACKENDS = "custom,defaulteinsum,matmuleinsum,"\
                "numpy_defaulteinsum,numpy_matmuleinsum"
except ModuleNotFoundError: # pragma: no cover
    _BACKENDS = "numpy_defaulteinsum,numpy_matmuleinsum"


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(
        mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip("cannot run on platform {}".format(plat))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "linux: mark test to run only on linux"
    )


def pytest_addoption(parser):
    parser.addoption("--backends", type=str, default=_BACKENDS)
    parser.addoption("--target-backend", type=str, default="numpy")


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        backends = metafunc.config.option.backends.split(",")
        metafunc.parametrize("backend", backends)
    elif "tested_backend" in metafunc.fixturenames:
        target = metafunc.config.option.target_backend
        engines = [x for x in _ENGINES if x != target]
        metafunc.parametrize("tested_backend", engines)
        metafunc.parametrize("target_backend", [target])
