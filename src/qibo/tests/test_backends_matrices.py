import pytest
import numpy as np
#from qibo import K


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_matrices(backend, dtype):
    from qibo.backends.matrices import Matrices
    mobj = Matrices(K)
    target_matrices = {
        "I": np.array([[1, 0], [0, 1]]),
        "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
        "S": np.array([[1, 0], [0, 1j]]),
        "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]]),
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
        matrix = getattr(mobj, matrixname)
        K.assert_allclose(matrix, target)


def test_modifying_matrices_error():
    from qibo import matrices
    with pytest.raises(AttributeError):
        matrices.I = np.zeros((2, 2))
