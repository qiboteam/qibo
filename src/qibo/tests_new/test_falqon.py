"""
Testing Variational Quantum Circuits.
"""
import numpy as np
import pathlib
import pytest
import qibo
from qibo import gates, models, hamiltonians
from qibo.tests_new.test_core_gates import random_state
from scipy.linalg import expm

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(array, filename, rtol=1e-5):
    """Check array matches data inside filename.

    Args:
        array: numpy array/
        filename: fixture filename

    If filename does not exists, this function
    creates the missing file otherwise it loads
    from file and compare.
    """
    def load(filename):
        return np.loadtxt(filename)

    filename = REGRESSION_FOLDER/filename
    try:
        array_fixture = load(filename)
    except: # pragma: no cover
        # case not tested in GitHub workflows because files exist
        np.savetxt(filename, array)
        array_fixture = load(filename)
    np.testing.assert_allclose(array, array_fixture, rtol=rtol)


test_names = "delta_t,max_layers,tolerance,filename"
test_values = [
    (0.1, 5, None, "falqon1.out"),
    (0.01, 2, None, "falqon2.out"),
    (0.01, 2, 1e-5, "falqon3.out")
    ]
@pytest.mark.parametrize(test_names, test_values)
def test_falqon_optimization(backend, delta_t, max_layers, tolerance, filename):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    h = hamiltonians.XXZ(3)
    falqon = models.FALQON(h)
    best, params, extra = falqon.minimize(delta_t, max_layers, tol=tolerance)
    if filename is not None:
        assert_regression_fixture(params, filename)
    qibo.set_backend(original_backend)


def test_falqon_optimization_callback(backend):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)

    class TestCallback:
        from qibo import K

        def __call__(self, x):
            return self.K.sum(x)

    callback = TestCallback()
    h = hamiltonians.XXZ(3)
    falqon = models.FALQON(h)
    best, params, extra = falqon.minimize(0.1, 5, callback=callback)
    assert len(extra["callbacks"]) == 5
    qibo.set_backend(original_backend)
