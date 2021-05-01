import pytest
import qibo
import numpy as np


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_matrices(backend, dtype):
    from qibo.backends.matrices import Matrices
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    mobj = Matrices(qibo.K)
    target_matrices = {
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
    for matrixname, target in target_matrices.items():
        np.testing.assert_allclose(getattr(mobj, matrixname), target)
    qibo.set_backend(original_backend)


def test_modifying_matrices_error():
    from qibo import matrices
    with pytest.raises(AttributeError):
        matrices.I = np.zeros((2, 2))
