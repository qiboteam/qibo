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


test_a0 = np.sqrt(0.4) * matrices.X
test_a1 = np.sqrt(0.6) * matrices.Z
test_kraus = [((0,), test_a0), ((0,), test_a1)]
test_superop = np.array(
    [
        [0.6 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.4 + 0.0j],
        [0.0 + 0.0j, -0.6 + 0.0j, 0.4 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.4 + 0.0j, -0.6 + 0.0j, 0.0 + 0.0j],
        [0.4 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.6 + 0.0j],
    ]
)
test_choi = np.reshape(test_superop, [2] * 4).swapaxes(0, 3).reshape([4, 4])


@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_choi(order):
    choi = liouville_to_choi(test_superop, order)

    assert np.linalg.norm(choi - test_choi) < PRECISION_TOL, True


@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_liouville(order):
    liouville = choi_to_liouville(test_choi, order=order)

    assert np.linalg.norm(liouville - test_superop) < PRECISION_TOL, True


@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_kraus(order):
    with pytest.raises(TypeError):
        choi_to_kraus(test_choi, "1e-8")
    with pytest.raises(ValueError):
        choi_to_kraus(test_choi, -1.0 * 1e-8)
    with pytest.raises(ValueError):
        # a random unitary is extremely unlikely to be Hermitian
        # so this test should be fine
        d = 2 ** 3
        choi_to_kraus(random_unitary(d))

    kraus_ops, coefficients = choi_to_kraus(test_choi, order=order)

    a0 = coefficients[0] * kraus_ops[0]
    a1 = coefficients[1] * kraus_ops[1]

    state = random_density_matrix(2)

    evolution_a0 = a0 @ state @ a0.T.conj()
    evolution_a1 = a1 @ state @ a1.T.conj()

    test_evolution_a0 = test_a0 @ state @ test_a0.T.conj()
    test_evolution_a1 = test_a1 @ state @ test_a1.T.conj()

    assert np.linalg.norm(evolution_a0 - test_evolution_a0) < PRECISION_TOL, True
    assert np.linalg.norm(evolution_a1 - test_evolution_a1) < PRECISION_TOL, True


@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_choi(order):
    choi = kraus_to_choi(test_kraus, order=order)

    assert np.linalg.norm(choi - test_choi) < PRECISION_TOL, True


@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_liouville(order):
    liouville = kraus_to_liouville(test_kraus, order=order)

    assert np.linalg.norm(liouville - test_superop) < PRECISION_TOL, True


@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_kraus(order):
    kraus_ops, coefficients = liouville_to_kraus(test_superop, order=order)

    a0 = coefficients[0] * kraus_ops[0]
    a1 = coefficients[1] * kraus_ops[1]

    state = random_density_matrix(2)

    evolution_a0 = a0 @ state @ a0.T.conj()
    evolution_a1 = a1 @ state @ a1.T.conj()

    test_evolution_a0 = test_a0 @ state @ test_a0.T.conj()
    test_evolution_a1 = test_a1 @ state @ test_a1.T.conj()

    assert np.linalg.norm(evolution_a0 - test_evolution_a0) < PRECISION_TOL, True
    assert np.linalg.norm(evolution_a1 - test_evolution_a1) < PRECISION_TOL, True


@pytest.mark.parametrize("order", ["row", "column"])
def test_reshuffling(order):
    from qibo.quantum_info.superoperator_transformations import _reshuffling

    with pytest.raises(TypeError):
        _reshuffling(test_superop, True)
    with pytest.raises(ValueError):
        _reshuffling(test_superop, "sustem")
    with pytest.raises(NotImplementedError):
        _reshuffling(test_superop, "system")

    reshuffled = _reshuffling(test_superop, order)
    reshuffled = _reshuffling(reshuffled, order)

    assert np.linalg.norm(reshuffled - test_superop) < PRECISION_TOL, True

    reshuffled = _reshuffling(test_choi, order)
    reshuffled = _reshuffling(reshuffled, order)

    assert np.linalg.norm(reshuffled - test_choi) < PRECISION_TOL, True
