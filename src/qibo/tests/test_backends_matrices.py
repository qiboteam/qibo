import pytest
import numpy as np
import tensorflow as tf
from qibo.backends import matrices

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


@pytest.mark.parametrize("name,dtype", [("NumpyMatrices", np.complex64),
                                        ("NumpyMatrices", np.complex128),
                                        ("TensorflowMatrices", tf.complex64),
                                        ("TensorflowMatrices", tf.complex128)])
def test_matrices(name, dtype):
    mobj = getattr(matrices, name)(dtype)
    for matrixname, target in TARGET_MATRICES.items():
        np.testing.assert_allclose(getattr(mobj, matrixname), target)
