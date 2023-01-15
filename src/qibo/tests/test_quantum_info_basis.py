import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_vectorization(nqubits, order):
    with pytest.raises(TypeError):
        vectorization(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(TypeError):
        vectorization(
            np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0]]], dtype="object")
        )
    with pytest.raises(TypeError):
        vectorization(np.array([]))
    with pytest.raises(TypeError):
        vectorization(random_statevector(4), order=1)
    with pytest.raises(ValueError):
        vectorization(random_statevector(4), order="1")

    d = 2**nqubits

    if nqubits == 1:
        if order == "system" or order == "column":
            matrix_test = [0, 2, 1, 3]
        else:
            matrix_test = [0, 1, 2, 3]
    elif nqubits == 2:
        if order == "row":
            matrix_test = np.arange(d**2)
        elif order == "column":
            matrix_test = np.arange(d**2)
            matrix_test = np.reshape(matrix_test, (d, d))
            matrix_test = np.reshape(matrix_test, (1, -1), order="F")[0]
        else:
            matrix_test = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    else:
        if order == "row":
            matrix_test = np.arange(d**2)
        elif order == "column":
            matrix_test = np.arange(d**2)
            matrix_test = np.reshape(matrix_test, (d, d))
            matrix_test = np.reshape(matrix_test, (1, -1), order="F")[0]
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
    matrix = vectorization(matrix, order)

    assert np.linalg.norm(matrix - matrix_test) < PRECISION_TOL


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("nqubits", [2, 3, 4, 5])
def test_unvectorization(nqubits, order):
    with pytest.raises(TypeError):
        unvectorization(random_density_matrix(2**nqubits))
    with pytest.raises(TypeError):
        unvectorization(random_statevector(4**nqubits), order=1)
    with pytest.raises(ValueError):
        unvectorization(random_statevector(4**2), order="1")

    d = 2**nqubits
    matrix_test = random_density_matrix(d)

    matrix = vectorization(matrix_test, order)
    matrix = unvectorization(matrix, order)

    assert np.linalg.norm(matrix_test - matrix) < PRECISION_TOL


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_pauli_basis(nqubits, normalize, vectorize, order):
    with pytest.raises(ValueError):
        pauli_basis(-1)
    with pytest.raises(TypeError):
        pauli_basis("1")
    with pytest.raises(TypeError):
        pauli_basis(1, "True")
    with pytest.raises(TypeError):
        pauli_basis(1, False, "True")
    with pytest.raises(ValueError):
        pauli_basis(1, False, True)

    basis_test = [matrices.I, matrices.X, matrices.Y, matrices.Z]
    if nqubits >= 2:
        basis_test = list(product(basis_test, repeat=nqubits))
        basis_test = [reduce(np.kron, matrices) for matrices in basis_test]

    if vectorize:
        basis_test = [vectorization(matrix, order=order) for matrix in basis_test]

    basis_test = np.array(basis_test)

    if normalize:
        basis_test /= np.sqrt(2**nqubits)

    basis = pauli_basis(nqubits, normalize, vectorize, order)

    for pauli, pauli_test in zip(basis, basis_test):
        assert np.linalg.norm(pauli - pauli_test) < PRECISION_TOL

    comp_basis_to_pauli(nqubits, normalize, order)
    pauli_to_comp_basis(nqubits, normalize, order)
