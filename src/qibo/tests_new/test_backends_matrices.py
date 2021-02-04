import pytest
import numpy as np
from qibo.backends import matrices
from qibo.config import raise_error

TARGET_MATRICES = {
    "I": np.array([[1, 0], [0, 1]]),
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
    "CNOT": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                      [0, 0, 0, 1], [0, 0, 1, 0]]),
    "CZ": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                    [0, 0, 1, 0], [0, 0, 0, -1]]),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                      [0, 1, 0, 0], [0, 0, 0, 1]]),
    "TOFFOLI": np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 1, 0]])
}


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_matrices(engine, dtype):
    if engine == "numpy":
        mobj = matrices.NumpyMatrices(getattr(np, dtype))
    elif engine == "tensorflow":
        import tensorflow as tf
        mobj = matrices.TensorflowMatrices(getattr(tf, dtype))
    else: # pragma: no cover
        # this case exists only for test consistency checking and
        # should never execute
        raise_error(ValueError, "Unknown engine {}.".format(engine))
    for matrixname, target in TARGET_MATRICES.items():
        np.testing.assert_allclose(getattr(mobj, matrixname), target)


def test_modifying_matrices_error():
    from qibo import matrices
    with pytest.raises(AttributeError):
        matrices.I = np.zeros((2, 2))
