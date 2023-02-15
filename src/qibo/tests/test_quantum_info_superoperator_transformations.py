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
test_non_CP = np.array(
    [
        [0.20031418, 0.37198771, 0.05642046, 0.37127765],
        [0.15812476, 0.21466209, 0.41971479, 0.20749836],
        [0.29408764, 0.01474688, 0.40320494, 0.28796054],
        [0.17587069, 0.42052825, 0.16171658, 0.24188449],
    ]
)
test_kraus_left = np.array(
    [
        [[-0.50326669, -0.50293148], [-0.49268667, -0.50104133]],
        [[-0.54148474, 0.32931326], [0.64742923, -0.42329947]],
        [[0.49035364, -0.62330138], [0.495577, -0.35419223]],
        [[-0.46159531, -0.50010808], [0.30413594, 0.66657558]],
    ]
)
test_kraus_right = np.array(
    [
        [[-0.41111026, -0.51035787], [-0.51635098, -0.55127567]],
        [[0.13825514, -0.69451163], [0.69697805, -0.11296331]],
        [[0.43827885, -0.49057101], [-0.48197173, 0.57875296]],
        [[0.78726458, 0.12856331], [-0.12371948, -0.59023677]],
    ]
)
test_coefficients = np.array([1.002719, 0.65635444, 0.43548, 0.21124177])


@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_choi(order):
    choi = liouville_to_choi(test_superop, order)

    assert np.linalg.norm(choi - test_choi) < PRECISION_TOL, True


@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_liouville(order):
    liouville = choi_to_liouville(test_choi, order=order)

    assert np.linalg.norm(liouville - test_superop) < PRECISION_TOL, True


@pytest.mark.parametrize("validate_CP", [False, True])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_kraus(order, validate_CP):
    with pytest.raises(TypeError):
        choi_to_kraus(test_choi, "1e-8")
    with pytest.raises(ValueError):
        choi_to_kraus(test_choi, -1.0 * 1e-8)
    with pytest.raises(TypeError):
        choi_to_kraus(test_choi, validate_CP="True")

    kraus_ops, coefficients = choi_to_kraus(
        test_choi, order=order, validate_CP=validate_CP
    )

    a0 = kraus_ops[0]
    a1 = kraus_ops[1]

    state = random_density_matrix(2)

    evolution_a0 = a0 @ state @ a0.T.conj()
    evolution_a1 = a1 @ state @ a1.T.conj()

    test_evolution_a0 = test_a0 @ state @ test_a0.T.conj()
    test_evolution_a1 = test_a1 @ state @ test_a1.T.conj()

    assert np.linalg.norm(evolution_a0 - test_evolution_a0) < PRECISION_TOL, True
    assert np.linalg.norm(evolution_a1 - test_evolution_a1) < PRECISION_TOL, True

    if validate_CP and order == "row":
        (kraus_left, kraus_right), _ = choi_to_kraus(
            test_non_CP, order=order, validate_CP=validate_CP
        )

        for test_left, left, test_right, right, test_coeff in zip(
            test_kraus_left,
            kraus_left,
            test_kraus_right,
            kraus_right,
            test_coefficients,
        ):
            state = random_density_matrix(2)
            evolution = left @ state @ right.T.conj()
            test_evolution = test_coeff**2 * test_left @ state @ test_right.T.conj()

            assert np.linalg.norm(evolution - test_evolution) < PRECISION_TOL, True


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
