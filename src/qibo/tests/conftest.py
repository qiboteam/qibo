"""
conftest.py

Pytest fixtures.
"""
import sys
import pytest
import qibo
from qibo import K

_available_backends = set(b.get('name') for b in K.profile.get('backends')
                          if (not b.get('is_hardware', False) and
                          K.check_availability(b.get('name'))))
_available_backends.add("numpy")
_ACCELERATORS = None
for bkd in _available_backends:
    if K.construct_backend(bkd).supports_multigpu:
        _ACCELERATORS = "2/GPU:0,1/GPU:0+1/GPU:1,2/GPU:0+1/GPU:1+1/GPU:2"
_BACKENDS = ",".join(_available_backends)


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


def pytest_addoption(parser):
    parser.addoption("--backends", type=str, default=_BACKENDS,
                     help="Calculation schemes (eg. qibojit, qibotf, tensorflow, numpy etc.) to test.")
    parser.addoption("--accelerators", type=str, default=_ACCELERATORS,
                     help="Accelerator configurations for testing the distributed circuit.")
    # see `_ACCELERATORS` for the string format of the `--accelerators` flag
    parser.addoption("--target-backend", type=str, default="numpy",
                     help="Base backend that other backends are tested against.")
    # `test_backends_agreement.py` tests that backend methods agree between
    # different backends by testing each backend in `--backends` with the
    # `--target-backend`


@pytest.fixture
def backend(backend_platform):
    if "-" in backend_platform:
        backend_name, platform_name = backend_platform.split("-")
    else:
        backend_name, platform_name = backend_platform, None

    original_backend = qibo.get_backend()
    original_platform = K.get_platform()
    qibo.set_backend(backend_name, platform=platform_name)
    yield
    qibo.set_backend(original_backend, platform=original_platform)


def pytest_generate_tests(metafunc):
    """Generates all tests defined under `src/qibo/tests`.

    Test functions may have one or more of the following arguments:
        engine: Backend library (eg. numpy, tensorflow, etc.),
        backend: Calculation backend (eg. qibojit, qibotf, tensorflow, numpy),
        accelerators: Dictionary with the accelerator configuration for
            distributed circuits, for example: {'/GPU:0': 1, '/GPU:1': 1},
        tested_backend: The first backend when testing agreement between
            backend methods (in `test_backends_agreement.py`)
        target_backend: The second backend when testing agreement between
            backend methods (in `test_backends_agreement.py`)

    This function parametrizes the above arguments using the values given by
    the user when calling `pytest`.
    """
    backends = metafunc.config.option.backends.split(",")
    # construct backend instances to find what platforms each backend supports
    # and if it supports multi-GPU
    backend_platforms = []
    distributed_backends = [] # list
    for name in backends:
        instance = K.construct_backend(name)
        platforms = [f"{name}-{p}" if p is not None else name
                     for p in instance.available_platforms]
        backend_platforms.extend(platforms)
        if instance.supports_multigpu:
            distributed_backends.extend(platforms)

    # parse accelerator stings to dicts
    accelerators = metafunc.config.option.accelerators.split(",")
    if accelerators is not None:
        accelerators = [{dev[1:]: int(dev[0]) for dev in x.split("+")} for x in accelerators]

    distributed_tests = {
        "qibo.tests.test_core_states_distributed",
        "qibo.tests.test_core_distutils",
        "qibo.tests.test_core_distcircuit",
        "qibo.tests.test_core_distcircuit_execution"
    }
    module_name = metafunc.module.__name__
    # skip distributed tests if qibojit or qibotf are not installed
    if module_name in distributed_tests and not distributed_backends: # pragma: no cover
        pytest.skip("Skipping distributed tests because are not supported by "
                    "the available backends.")
    # skip distributed tests on mac
    if sys.platform == "darwin":  # pragma: no cover
        accelerators = None
        if module_name in distributed_tests:
            pytest.skip("macos does not support distributed circuits.")

    # for `test_backends_agreement.py`
    if "tested_backend" in metafunc.fixturenames:
        target = metafunc.config.option.target_backend
        metafunc.parametrize("tested_backend", [x for x in backends if x != target])
        metafunc.parametrize("target_backend", [target])

    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", backends)

    if "backend_platform" in metafunc.fixturenames:
        if metafunc.module.__name__ in distributed_tests:
            metafunc.parametrize("backend_platform", distributed_backends)
            if "accelerators" in metafunc.fixturenames:
                metafunc.parametrize("accelerators", accelerators)

        elif "accelerators" in metafunc.fixturenames:
            if accelerators is None: # pragma: no cover
                # `accelerators` is never `None` in CI test execution
                metafunc.parametrize("backend_platform", backend_platforms)
                metafunc.parametrize("accelerators", [None])
            else:
                config = [(b, None) for b in backend_platforms]
                config.extend((b, a) for b in distributed_backends for a in accelerators)
                metafunc.parametrize("backend_platform,accelerators", config)

        else:
            metafunc.parametrize("backend_platform", backend_platforms)
