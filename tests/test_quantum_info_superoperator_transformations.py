import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_vectorization(backend, nqubits, order):
    with pytest.raises(TypeError):
        vectorization(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]), backend=backend)
    with pytest.raises(TypeError):
        vectorization(
            np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0]]], dtype="object"),
            backend=backend,
        )
    with pytest.raises(TypeError):
        vectorization(np.array([]), backend=backend)
    with pytest.raises(TypeError):
        vectorization(random_statevector(4, backend=backend), order=1, backend=backend)
    with pytest.raises(ValueError):
        vectorization(
            random_statevector(4, backend=backend), order="1", backend=backend
        )

    dim = 2**nqubits

    if nqubits == 1:
        if order == "system" or order == "column":
            matrix_test = [0, 2, 1, 3]
        else:
            matrix_test = [0, 1, 2, 3]
    elif nqubits == 2:
        if order == "row":
            matrix_test = np.arange(dim**2)
        elif order == "column":
            matrix_test = np.arange(dim**2)
            matrix_test = np.reshape(matrix_test, (dim, dim))
            matrix_test = np.reshape(matrix_test, (1, -1), order="F")[0]
        else:
            matrix_test = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    else:
        if order == "row":
            matrix_test = np.arange(dim**2)
        elif order == "column":
            matrix_test = np.arange(dim**2)
            matrix_test = np.reshape(matrix_test, (dim, dim))
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
    matrix_test = backend.cast(matrix_test)

    dim = 2**nqubits
    matrix = np.arange(dim**2).reshape((dim, dim))
    matrix = vectorization(matrix, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(matrix - matrix_test) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("nqubits", [2, 3, 4, 5])
def test_unvectorization(backend, nqubits, order):
    with pytest.raises(TypeError):
        unvectorization(
            random_density_matrix(2**nqubits, backend=backend), backend=backend
        )
    with pytest.raises(TypeError):
        unvectorization(
            random_statevector(4**nqubits, backend=backend), order=1, backend=backend
        )
    with pytest.raises(ValueError):
        unvectorization(
            random_statevector(4**2, backend=backend), order="1", backend=backend
        )

    dim = 2**nqubits
    matrix_test = random_density_matrix(dim, backend=backend)

    matrix = vectorization(matrix_test, order, backend)
    matrix = unvectorization(matrix, order, backend)

    backend.assert_allclose(
        backend.calculate_norm(matrix_test - matrix) < PRECISION_TOL, True
    )


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
test_pauli = np.diag([2.0, -0.4, -2.0, 0.4])
test_chi = np.diag([0, 1.6, 0, 2.4])


@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_choi(backend, order):
    choi = liouville_to_choi(test_superop, order, backend)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)

    backend.assert_allclose(
        backend.calculate_norm(choi - test_choi) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_liouville(backend, order, test_superop):
    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    liouville = choi_to_liouville(test_choi, order=order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(liouville - test_superop) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_kraus_right", [test_kraus_right])
@pytest.mark.parametrize("test_kraus_left", [test_kraus_left])
@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("validate_cp", [False, True])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_kraus(
    backend, order, validate_cp, test_a0, test_a1, test_kraus_left, test_kraus_right
):
    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])

    with pytest.raises(TypeError):
        choi_to_kraus(test_choi, str(PRECISION_TOL), backend=backend)
    with pytest.raises(ValueError):
        choi_to_kraus(test_choi, -1.0 * PRECISION_TOL, backend=backend)
    with pytest.raises(TypeError):
        choi_to_kraus(test_choi, validate_cp="True", backend=backend)

    kraus_ops, _ = choi_to_kraus(
        test_choi, order=order, validate_cp=validate_cp, backend=backend
    )

    a0 = kraus_ops[0]
    a1 = kraus_ops[1]
    a0 = backend.cast(a0, dtype=a0.dtype)
    a1 = backend.cast(a1, dtype=a1.dtype)

    test_a0 = backend.cast(test_a0, dtype=test_a0.dtype)
    test_a1 = backend.cast(test_a1, dtype=test_a1.dtype)

    test_kraus_left = backend.cast(test_kraus_left, dtype=test_kraus_left.dtype)
    test_kraus_right = backend.cast(test_kraus_right, dtype=test_kraus_right.dtype)

    state = random_density_matrix(2, backend=backend)

    evolution_a0 = a0 @ state @ np.transpose(np.conj(a0))
    evolution_a1 = a1 @ state @ np.transpose(np.conj(a1))

    test_evolution_a0 = test_a0 @ state @ np.transpose(np.conj(test_a0))
    test_evolution_a1 = test_a1 @ state @ np.transpose(np.conj(test_a1))

    backend.assert_allclose(
        backend.calculate_norm(evolution_a0 - test_evolution_a0) < 2 * PRECISION_TOL,
        True,
    )
    backend.assert_allclose(
        backend.calculate_norm(evolution_a1 - test_evolution_a1) < 2 * PRECISION_TOL,
        True,
    )

    if validate_cp and order == "row":
        (kraus_left, kraus_right), _ = choi_to_kraus(
            test_non_CP, order=order, validate_cp=validate_cp, backend=backend
        )

        for test_left, left, test_right, right, test_coeff in zip(
            test_kraus_left,
            kraus_left,
            test_kraus_right,
            kraus_right,
            test_coefficients,
        ):
            state = random_density_matrix(2, backend=backend)
            evolution = left @ state @ np.transpose(np.conj(right))
            test_evolution = (
                test_coeff**2 * test_left @ state @ np.transpose(np.conj(test_right))
            )

            backend.assert_allclose(
                backend.calculate_norm(evolution - test_evolution) < 2 * PRECISION_TOL,
                True,
            )


@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_choi(backend, order):
    choi = kraus_to_choi(test_kraus, order=order, backend=backend)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    test_choi = backend.cast(test_choi)

    backend.assert_allclose(
        backend.calculate_norm(choi - test_choi) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_liouville(backend, order, test_superop):
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    liouville = kraus_to_liouville(test_kraus, order=order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(liouville - test_superop) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_kraus(backend, order, test_a0, test_a1):
    kraus_ops, _ = liouville_to_kraus(test_superop, order=order, backend=backend)

    a0 = kraus_ops[0]
    a1 = kraus_ops[1]
    a0 = backend.cast(a0, dtype=a0.dtype)
    a1 = backend.cast(a1, dtype=a1.dtype)

    test_a0 = backend.cast(test_a0, dtype=test_a0.dtype)
    test_a1 = backend.cast(test_a1, dtype=test_a1.dtype)

    state = random_density_matrix(2, backend=backend)

    evolution_a0 = a0 @ state @ np.transpose(np.conj(a0))
    evolution_a1 = a1 @ state @ np.transpose(np.conj(a1))

    test_evolution_a0 = test_a0 @ state @ np.transpose(np.conj(test_a0))
    test_evolution_a1 = test_a1 @ state @ np.transpose(np.conj(test_a1))

    backend.assert_allclose(
        backend.calculate_norm(evolution_a0 - test_evolution_a0) < PRECISION_TOL, True
    )
    backend.assert_allclose(
        backend.calculate_norm(evolution_a1 - test_evolution_a1) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_liouville(backend, normalize, order, test_pauli, test_superop):
    with pytest.raises(ValueError):
        pauli_to_liouville(test_pauli[:-1, :], normalize, order, backend=backend)

    dim = int(np.sqrt(test_superop.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_pauli = backend.cast(test_pauli)

    super_op = pauli_to_liouville(test_pauli / aux, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_superop - super_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("normalize", [False, True])
def test_liouville_to_pauli(backend, normalize, order, test_pauli, test_superop):
    with pytest.raises(ValueError):
        liouville_to_pauli(test_superop[:-1, :], normalize, order, backend=backend)

    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_pauli = backend.cast(test_pauli)

    pauli_op = liouville_to_pauli(
        test_superop, normalize=normalize, order=order, backend=backend
    )

    backend.assert_allclose(
        backend.calculate_norm(test_pauli / aux - pauli_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_choi(backend, normalize, order, test_pauli, test_superop):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    choi_super_op = pauli_to_choi(test_pauli / aux, normalize, order, backend=backend)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop, [2] * 4), *axes).reshape([4, 4])

    backend.assert_allclose(
        backend.calculate_norm(test_choi - choi_super_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_choi_to_pauli(backend, normalize, order, test_pauli, test_superop):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop * aux, [2] * 4), *axes).reshape(
        [4, 4]
    )

    pauli_op = choi_to_pauli(test_choi, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_pauli - pauli_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_kraus(backend, normalize, order, test_pauli, test_a0, test_a1):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    kraus_ops, _ = pauli_to_kraus(
        test_pauli / aux, normalize, order=order, backend=backend
    )

    a0 = kraus_ops[0]
    a1 = kraus_ops[1]
    a0 = backend.cast(a0, dtype=a0.dtype)
    a1 = backend.cast(a1, dtype=a1.dtype)

    test_a0 = backend.cast(test_a0, dtype=test_a0.dtype)
    test_a1 = backend.cast(test_a1, dtype=test_a1.dtype)

    state = random_density_matrix(2, backend=backend)

    evolution_a0 = a0 @ state @ np.transpose(np.conj(a0))
    evolution_a1 = a1 @ state @ np.transpose(np.conj(a1))

    test_evolution_a0 = test_a0 @ state @ np.transpose(np.conj(test_a0))
    test_evolution_a1 = test_a1 @ state @ np.transpose(np.conj(test_a1))

    backend.assert_allclose(
        backend.calculate_norm(evolution_a0 - test_evolution_a0) < 2 * PRECISION_TOL,
        True,
    )
    backend.assert_allclose(
        backend.calculate_norm(evolution_a1 - test_evolution_a1) < 2 * PRECISION_TOL,
        True,
    )


@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("test_kraus", [test_kraus])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_kraus_to_pauli(backend, normalize, order, test_kraus, test_pauli):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    pauli_op = kraus_to_pauli(test_kraus, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_pauli / aux - pauli_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_choi_to_chi(backend, normalize, order, test_superop, test_chi):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1 if normalize == False else dim

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop * aux, [2] * 4), *axes).reshape(
        [4, 4]
    )

    chi_matrix = choi_to_chi(
        test_choi, normalize=normalize, order=order, backend=backend
    )
    backend.assert_allclose(
        backend.calculate_norm(test_chi - chi_matrix) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("test_kraus", [test_kraus])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_kraus_to_chi(backend, normalize, order, test_kraus, test_chi):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    chi_matrix = kraus_to_chi(test_kraus, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_chi / aux - chi_matrix) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_liouville_to_chi(backend, normalize, order, test_superop, test_chi):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    chi_matrix = liouville_to_chi(test_superop, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_chi / aux - chi_matrix) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_chi(backend, normalize, order, test_pauli, test_chi):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    chi_matrix = pauli_to_chi(test_pauli / aux, normalize, order, backend=backend)

    aux = 1.0 if normalize == False else dim
    backend.assert_allclose(
        backend.calculate_norm(test_chi / aux - chi_matrix) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_choi(backend, normalize, order, test_chi, test_superop):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop, [2] * 4), *axes).reshape([4, 4])

    choi_super_op = chi_to_choi(test_chi / aux, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_choi - choi_super_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_liouville(backend, normalize, order, test_chi, test_superop):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    super_op = chi_to_liouville(test_chi / aux, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_superop - super_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_pauli", [test_pauli])
@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_pauli(backend, normalize, order, test_chi, test_pauli):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    pauli_op = chi_to_pauli(test_chi / aux, normalize, order, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(test_pauli - pauli_op) < PRECISION_TOL, True
    )


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("test_chi", [test_chi])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_kraus(backend, normalize, order, test_chi, test_a0, test_a1):
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    kraus_ops, _ = chi_to_kraus(test_chi / aux, normalize, order=order, backend=backend)

    a0 = kraus_ops[0]
    a1 = kraus_ops[1]
    a0 = backend.cast(a0, dtype=a0.dtype)
    a1 = backend.cast(a1, dtype=a1.dtype)

    test_a0 = backend.cast(test_a0, dtype=test_a0.dtype)
    test_a1 = backend.cast(test_a1, dtype=test_a1.dtype)

    state = random_density_matrix(2, backend=backend)

    evolution_a0 = a0 @ state @ np.transpose(np.conj(a0))
    evolution_a1 = a1 @ state @ np.transpose(np.conj(a1))

    test_evolution_a0 = test_a0 @ state @ np.transpose(np.conj(test_a0))
    test_evolution_a1 = test_a1 @ state @ np.transpose(np.conj(test_a1))

    backend.assert_allclose(
        backend.calculate_norm(evolution_a0 - test_evolution_a0) < 2 * PRECISION_TOL,
        True,
    )
    backend.assert_allclose(
        backend.calculate_norm(evolution_a1 - test_evolution_a1) < 2 * PRECISION_TOL,
        True,
    )


@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_unitaries(backend, order):
    test_a0 = np.sqrt(0.4) * matrices.X
    test_a1 = np.sqrt(0.6) * matrices.Y
    # test_a0 = backend.cast(test_a0, dtype=test_a0.dtype)
    # test_a1 = backend.cast(test_a1, dtype=test_a1.dtype)
    test_kraus = [((0,), test_a0), ((0,), test_a1)]

    with pytest.raises(TypeError):
        kraus_to_unitaries(test_kraus, order, str(PRECISION_TOL), backend=backend)
    with pytest.raises(ValueError):
        kraus_to_unitaries(test_kraus, order, -1.0 * PRECISION_TOL, backend=backend)

    target = kraus_to_liouville(test_kraus, order=order, backend=backend)

    unitaries, probabilities = kraus_to_unitaries(
        test_kraus, order=order, backend=backend
    )
    unitaries = np.array(
        [np.sqrt(prob) * unitary for prob, unitary in zip(probabilities, unitaries)]
    )
    unitaries = list(zip([(0,)] * len(unitaries), unitaries))

    operator = kraus_to_liouville(unitaries, backend=backend)

    backend.assert_allclose(
        backend.calculate_norm(target - operator) < 2 * PRECISION_TOL, True
    )

    # warning coverage
    test_a0 = np.sqrt(0.4) * matrices.X
    test_a1 = np.sqrt(0.6) * matrices.Z
    test_kraus = [((0,), test_a0), ((0,), test_a1)]
    kraus_to_unitaries(test_kraus, order=order, backend=backend)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
def test_reshuffling(backend, order, test_superop):
    from qibo.quantum_info.superoperator_transformations import _reshuffling

    with pytest.raises(TypeError):
        _reshuffling(test_superop, True, backend=backend)
    with pytest.raises(ValueError):
        _reshuffling(test_superop, "sustem", backend=backend)
    with pytest.raises(NotImplementedError):
        _reshuffling(test_superop, "system", backend=backend)
    with pytest.raises(ValueError):
        _reshuffling(test_superop[:-1, :-1], order, backend=backend)

    reshuffled = _reshuffling(test_superop, order, backend=backend)
    reshuffled = _reshuffling(reshuffled, order, backend=backend)

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    backend.assert_allclose(
        np.linalg.norm(reshuffled - test_superop) < PRECISION_TOL, True
    )

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(
        np.swapaxes(np.reshape(test_superop, [2] * 4), *axes), [4, 4]
    )

    reshuffled = _reshuffling(test_choi, order, backend=backend)
    reshuffled = _reshuffling(reshuffled, order, backend=backend)

    backend.assert_allclose(
        np.linalg.norm(reshuffled - test_choi) < PRECISION_TOL, True
    )
