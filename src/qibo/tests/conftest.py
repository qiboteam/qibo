"""
conftest.py

Pytest fixtures.
"""
import os
import sys
import pytest


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


def pytest_generate_tests(metafunc):
    from qibo import K
    if "accelerators" in metafunc.fixturenames:
        if "custom" in K.available_backends:
            accelerators = [None, {"/GPU:0": 1, "/GPU:1": 1}]
        else: # pragma: no cover
            accelerators = [None]
        metafunc.parametrize("accelerators", accelerators)

    if "backend" in metafunc.fixturenames:
        backends = ["custom", "defaulteinsum", "matmuleinsum"]
        if "custom" not in K.available_backends: # pragma: no cover
            backends.remove("custom")
        metafunc.parametrize("backend", backends)

    # skip distributed tests if "custom" backend is not available
    module_name = "qibo.tests.test_distributed"
    if metafunc.module.__name__ == module_name:
        if "custom" not in K.available_backends: # pragma: no cover
            pytest.skip("Distributed circuits require custom operators.")

    # skip parallel tests on Windows
    if os.name == "nt": # pragma: no cover
        module_name = "qibo.tests.test_parallel"
        if metafunc.module.__name__ == module_name:
            pytest.skip("Multiprocessing is not available on Windows.")
