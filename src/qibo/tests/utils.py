"""Various utilities for tests."""
import pathlib
import numpy as np


REGRESSION_FOLDER = pathlib.Path(__file__).with_name('regressions')

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


def random_numpy_complex(shape, dtype=np.complex128):
  return (np.random.random(shape) + 1j * np.random.random(shape)).astype(dtype)


def random_tensorflow_complex(shape, dtype="float64"):
    import tensorflow as tf
    if isinstance(dtype, str):
        dtype = getattr(tf, dtype)
    _re = tf.random.uniform(shape, dtype=dtype)
    _im = tf.random.uniform(shape, dtype=dtype)
    return tf.complex(_re, _im)


def random_numpy_state(nqubits, dtype=np.complex128):
    """Generates a random normalized state vector as numpy array.

    Args:
        nqubits (int): Number of qubits in the state.
        dtype: Numpy type of the state array.

    Returns:
        Numpy array for state vector of shape (2 ** nqubits,).
    """
    x = random_numpy_complex(2 ** nqubits, dtype)
    return (x / np.sqrt((np.abs(x) ** 2).sum())).astype(dtype)
