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
    import qibo.tensorflow.custom_operators as op
    if not op._custom_operators_loaded: # pragma: no cover
        _BACKENDS = "defaulteinsum,matmuleinsum,"\
                    "numpy_defaulteinsum,numpy_matmuleinsum"
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
    parser.addoption("--engines", type=str, default=_ENGINES,
                     help="Backend libaries (eg. numpy, tensorflow, etc.) to test.")
    parser.addoption("--backends", type=str, default=_BACKENDS,
                     help="Calculation schemes (eg. custom, defaulteinsum, etc.) to test.")
    parser.addoption("--accelerators", type=str, default=_ACCELERATORS,
                     help="Accelerator configurations for testing the distributed circuit.")
    # see `_ACCELERATORS` for the string format of the `--accelerators` flag
    parser.addoption("--target-backend", type=str, default="numpy",
                     help="Base backend that other backends are tested against.")
    # `test_backends_agreement.py` tests that backend methods agree between
    # different backends by testing each backend in `--engines` with the
    # `--target-backend`


def pytest_generate_tests(metafunc):
    """Generates all tests defined under `src/qibo/tests_new`.

    Test functions may have one or more of the following arguments:
        engine: Backend library (eg. numpy, tensorflow, etc.),
        backend: Calculation backend (eg. custom, defaulteinsum,
            numpy_defaulteinsum, etc.),
        accelerators: Dictionary with the accelerator configuration for
            distributed circuits, for example: {'/GPU:0': 1, '/GPU:1': 1},
        tested_backend: The first backend when testing agreement between
            backend methods (in `test_backends_agreement.py`)
        target_backend: The second backend when testing agreement between
            backend methods (in `test_backends_agreement.py`)

    This function parametrizes the above arguments using the values given by
    the user when calling `pytest`.
    """
    engines = metafunc.config.option.engines.split(",")
    backends = metafunc.config.option.backends.split(",")
    accelerators = metafunc.config.option.accelerators
    if "tensorflow" not in engines: # pragma: no cover
        # CI uses Tensorflow engine for test execution
        accelerators = None
        for x in ["custom", "defaulteinsum", "matmuleinsum"]:
            if x in backends:
                backends.remove(x)

    if "custom" not in backends: # pragma: no cover
        # skip tests that require custom operators
        tests_to_skip = {
            "qibo.tests_new.test_tensorflow_custom_operators",
            "qibo.tests_new.test_core_states_distributed"
        }
        # for `test_tensorflow_custom_operators.py`
        if metafunc.module.__name__ in tests_to_skip:
            pytest.skip("Custom operator tests require Tensorflow engine.")

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
                # `accelerators` is never `None` in CI test execution
                metafunc.parametrize("backend", backends)
                metafunc.parametrize("accelerators", [None])
            else:
                config = [(b, None) for b in backends]
                if "custom" in backends:
                    for x in accelerators.split(","):
                        devdict = {dev[1:]: int(dev[0]) for dev in x.split("+")}
                        config.append(("custom", devdict))
                metafunc.parametrize("backend,accelerators", config)
        else:
            metafunc.parametrize("backend", backends)
