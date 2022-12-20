import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_vectorization(nqubits):
    with pytest.raises(TypeError):
        vectorization(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(TypeError):
        vectorization(np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0]]]))
    with pytest.raises(TypeError):
        vectorization(np.array([]))

    if nqubits == 1:
        matrix_test = [0, 2, 1, 3]
    elif nqubits == 2:
        matrix_test = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    else:
        matrix_test = [
            0,
            8,
            1,
            9,
            16,
            24,
            17,
            25,
            2,
            10,
            3,
            11,
            18,
            26,
            19,
            27,
            32,
            40,
            33,
            41,
            48,
            56,
            49,
            57,
            34,
            42,
            35,
            43,
            50,
            58,
            51,
            59,
            4,
            12,
            5,
            13,
            20,
            28,
            21,
            29,
            6,
            14,
            7,
            15,
            22,
            30,
            23,
            31,
            36,
            44,
            37,
            45,
            52,
            60,
            53,
            61,
            38,
            46,
            39,
            47,
            54,
            62,
            55,
            63,
        ]
    matrix_test = np.array(matrix_test)

    d = 2**nqubits
    matrix = np.arange(d**2).reshape((d, d))
    matrix = vectorization(matrix)

    assert np.linalg.norm(matrix - matrix_test) < PRECISION_TOL


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_pauli_basis(nqubits, normalize, vectorize):
    with pytest.raises(ValueError):
        pauli_basis(-1)
    with pytest.raises(TypeError):
        pauli_basis("1")
    with pytest.raises(TypeError):
        pauli_basis(1, "True")
    with pytest.raises(TypeError):
        pauli_basis(1, False, "True")

    single_basis = [matrices.I, matrices.X, matrices.Y, matrices.Z]

    if vectorize:
        single_basis = [matrix.reshape((1, -1), order="F") for matrix in single_basis]

    if nqubits == 1:
        basis_test = single_basis
    else:
        basis_test = list(product(single_basis, repeat=nqubits))
        if vectorize:
            basis_test = [reduce(np.outer, matrix).ravel() for matrix in basis_test]
        else:
            basis_test = [reduce(np.kron, matrix) for matrix in basis_test]

    if normalize:
        basis_test /= np.sqrt(2**nqubits)

    basis = pauli_basis(nqubits, normalize, vectorize)

    for pauli, pauli_test in zip(basis, basis_test):
        assert np.linalg.norm(pauli - pauli_test) < PRECISION_TOL

    comp_basis_to_pauli(nqubits, normalize)
    pauli_to_comp_basis(nqubits, normalize)
