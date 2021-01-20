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
except ModuleNotFoundError: # pragma: no cover
    _ENGINES = "numpy"
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
    parser.addoption("--engines", type=str, default=_ENGINES)
    parser.addoption("--backends", type=str, default=_BACKENDS)
    parser.addoption("--target-backend", type=str, default="numpy")


def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        backends = metafunc.config.option.backends.split(",")
        metafunc.parametrize("backend", backends)
    if "engine"  in metafunc.fixturenames:
        engines = metafunc.config.option.engines.split(",")
        metafunc.parametrize("engine", engines)
    if "tested_backend" in metafunc.fixturenames:
        engines = metafunc.config.option.engines.split(",")
        target = metafunc.config.option.target_backend
        engines = [x for x in engines if x != target]
        metafunc.parametrize("tested_backend", engines)
        metafunc.parametrize("target_backend", [target])
