import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info.random_ensembles import random_density_matrix, random_statevector
from qibo.quantum_info.superoperator_transformations import (
    chi_to_choi,
    chi_to_kraus,
    chi_to_liouville,
    chi_to_pauli,
    choi_to_chi,
    choi_to_kraus,
    choi_to_liouville,
    choi_to_pauli,
    kraus_to_chi,
    kraus_to_choi,
    kraus_to_liouville,
    kraus_to_pauli,
    kraus_to_unitaries,
    liouville_to_chi,
    liouville_to_choi,
    liouville_to_kraus,
    liouville_to_pauli,
    pauli_to_chi,
    pauli_to_choi,
    pauli_to_kraus,
    pauli_to_liouville,
    unvectorization,
    vectorization,
)


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

    backend.assert_allclose(matrix, matrix_test, atol=PRECISION_TOL)


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

    backend.assert_allclose(matrix_test, matrix, atol=PRECISION_TOL)


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

test_stinespring = np.array(
    [
        [0.0 + 0.0j, 0.0 + 0.0j, 0.63245553 + 0.0j, 0.0 + 0.0j],
        [0.77459667 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.63245553 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 0.0 + 0.0j, -0.77459667 + 0.0j, 0.0 + 0.0j],
    ]
)


def pauli_superop(pauli_order):
    elements = {"I": 2.0, "X": -0.4, "Y": -2.0, "Z": 0.4}
    return np.diag([elements[p] for p in pauli_order])


def chi_superop(pauli_order):
    elements = {"I": 0, "X": 1.6, "Y": 0, "Z": 2.4}
    return np.diag([elements[p] for p in pauli_order])


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_liouville(backend, order, test_superop):
    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    liouville = choi_to_liouville(test_choi, order=order, backend=backend)

    backend.assert_allclose(liouville, test_superop, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_choi_to_pauli(backend, normalize, order, pauli_order, test_superop):
    test_pauli = pauli_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop * aux, [2] * 4), *axes).reshape(
        [4, 4]
    )

    pauli_op = choi_to_pauli(
        test_choi, normalize, order, pauli_order=pauli_order, backend=backend
    )

    backend.assert_allclose(test_pauli, pauli_op, atol=PRECISION_TOL)


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

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=2 * PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=2 * PRECISION_TOL)

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

            backend.assert_allclose(evolution, test_evolution, atol=2 * PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_choi_to_chi(backend, normalize, order, pauli_order, test_superop):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = 1 if normalize == False else dim

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop * aux, [2] * 4), *axes).reshape(
        [4, 4]
    )

    chi_matrix = choi_to_chi(
        test_choi,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    backend.assert_allclose(test_chi, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_stinespring", [test_stinespring])
@pytest.mark.parametrize("nqubits", [1])
@pytest.mark.parametrize("validate_cp", [False, True])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_stinespring(backend, order, validate_cp, nqubits, test_stinespring):
    if validate_cp is True:
        with pytest.raises(NotImplementedError):
            test = choi_to_stinespring(
                test_non_CP,
                order=order,
                validate_cp=validate_cp,
                nqubits=nqubits,
                backend=backend,
            )

    test_stinespring = backend.cast(test_stinespring, dtype=test_stinespring.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)
    stinespring = choi_to_stinespring(
        test_choi,
        order=order,
        validate_cp=validate_cp,
        nqubits=nqubits,
        backend=backend,
    )

    backend.assert_allclose(stinespring, test_stinespring, atol=PRECISION_TOL)


@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_choi(backend, order):
    choi = kraus_to_choi(test_kraus, order=order, backend=backend)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)

    backend.assert_allclose(choi, test_choi, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_liouville(backend, order, test_superop):
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    liouville = kraus_to_liouville(test_kraus, order=order, backend=backend)

    backend.assert_allclose(liouville, test_superop, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_kraus", [test_kraus])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_kraus_to_pauli(backend, normalize, order, pauli_order, test_kraus):
    test_pauli = pauli_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    pauli_op = kraus_to_pauli(
        test_kraus, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_pauli / aux, pauli_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_kraus", [test_kraus])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_kraus_to_chi(backend, normalize, order, pauli_order, test_kraus):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    chi_matrix = kraus_to_chi(
        test_kraus, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_chi / aux, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("nqubits", [1])
def test_kraus_to_stinespring(backend, nqubits):
    with pytest.raises(ValueError):
        initial_state_env = random_statevector(4, backend=backend)
        test = kraus_to_stinespring(
            test_kraus,
            nqubits=nqubits,
            initial_state_env=initial_state_env,
            backend=backend,
        )
    with pytest.raises(ValueError):
        initial_state_env = random_density_matrix(2, pure=True, backend=backend)
        test = kraus_to_stinespring(
            test_kraus,
            nqubits=nqubits,
            initial_state_env=initial_state_env,
            backend=backend,
        )

    stinespring = kraus_to_stinespring(
        test_kraus,
        nqubits=nqubits,
        backend=backend,
    )

    backend.assert_allclose(stinespring, test_stinespring, atol=PRECISION_TOL)


@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_choi(backend, order):
    choi = liouville_to_choi(test_superop, order, backend)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)

    backend.assert_allclose(choi, test_choi, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("normalize", [False, True])
def test_liouville_to_pauli(backend, normalize, order, pauli_order, test_superop):
    test_pauli = pauli_superop(pauli_order)

    with pytest.raises(ValueError):
        liouville_to_pauli(test_superop[:-1, :], normalize, order, backend=backend)

    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_pauli = backend.cast(test_pauli)

    pauli_op = liouville_to_pauli(
        test_superop,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    backend.assert_allclose(test_pauli / aux, pauli_op, atol=PRECISION_TOL)


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

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_liouville_to_chi(backend, normalize, order, pauli_order, test_superop):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = 1.0 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    chi_matrix = liouville_to_chi(
        test_superop, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_chi / aux, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_liouville(backend, normalize, order, pauli_order, test_superop):
    test_pauli = pauli_superop(pauli_order)

    with pytest.raises(ValueError):
        pauli_to_liouville(test_pauli[:-1, :], normalize, order, backend=backend)

    dim = int(np.sqrt(test_superop.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_pauli = backend.cast(test_pauli)

    super_op = pauli_to_liouville(
        test_pauli / aux, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_superop, super_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_choi(backend, normalize, order, pauli_order, test_superop):
    test_pauli = pauli_superop(pauli_order)

    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    choi_super_op = pauli_to_choi(
        test_pauli / aux, normalize, order, pauli_order, backend=backend
    )

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop, [2] * 4), *axes).reshape([4, 4])

    backend.assert_allclose(test_choi, choi_super_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_kraus(backend, normalize, order, pauli_order, test_a0, test_a1):
    test_pauli = pauli_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    kraus_ops, _ = pauli_to_kraus(
        test_pauli / aux,
        normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
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

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=2 * PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=2 * PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_chi(backend, normalize, order, pauli_order):
    test_pauli = pauli_superop(pauli_order)
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_pauli = backend.cast(test_pauli / aux, dtype=test_pauli.dtype)

    chi_matrix = pauli_to_chi(
        test_pauli, normalize, order, pauli_order, backend=backend
    )

    aux = 1.0 if normalize == False else dim
    backend.assert_allclose(test_chi / aux, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_choi(backend, normalize, order, pauli_order, test_superop):
    test_chi = chi_superop(pauli_order=pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.swapaxes(np.reshape(test_superop, [2] * 4), *axes).reshape([4, 4])

    choi_super_op = chi_to_choi(
        test_chi / aux, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_choi, choi_super_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_liouville(backend, normalize, order, pauli_order, test_superop):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    super_op = chi_to_liouville(
        test_chi / aux, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_superop, super_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_pauli(backend, normalize, order, pauli_order):
    test_pauli = pauli_superop(pauli_order)
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    pauli_op = chi_to_pauli(
        test_chi / aux, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_pauli, pauli_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_kraus(backend, normalize, order, pauli_order, test_a0, test_a1):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = dim**2 if normalize == False else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    kraus_ops, _ = chi_to_kraus(
        test_chi / aux, normalize, order=order, pauli_order=pauli_order, backend=backend
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

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=2 * PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=2 * PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("nqubits", [None, 1])
@pytest.mark.parametrize("dim_env", [2])
@pytest.mark.parametrize("stinespring", [test_stinespring])
def test_stinespring_to_choi(
    backend, stinespring, dim_env, nqubits, order, test_superop
):
    stinespring = backend.cast(stinespring, dtype=stinespring.dtype)
    choi_super_op = stinespring_to_choi(
        stinespring, dim_env=dim_env, nqubits=nqubits, order=order, backend=backend
    )

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(
        np.swapaxes(np.reshape(test_superop, [2] * 4), *axes), [4, 4]
    )
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)

    backend.assert_allclose(choi_super_op, test_choi, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("nqubits", [None, 1])
@pytest.mark.parametrize("dim_env", [2])
@pytest.mark.parametrize("stinespring", [test_stinespring])
def test_stinespring_to_liouville(
    backend, stinespring, dim_env, nqubits, order, test_superop
):
    stinespring = backend.cast(stinespring, dtype=stinespring.dtype)

    super_op = stinespring_to_liouville(
        stinespring, dim_env, nqubits=nqubits, order=order, backend=backend
    )

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    backend.assert_allclose(super_op, test_superop, atol=PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [None, 1])
@pytest.mark.parametrize("dim_env", [2])
@pytest.mark.parametrize("stinespring", [test_stinespring])
def test_stinespring_to_pauli(
    backend, stinespring, dim_env, nqubits, normalize, order, pauli_order
):
    test_pauli = pauli_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if normalize == False else dim

    stinespring = backend.cast(stinespring, dtype=stinespring.dtype)
    test_pauli = backend.cast(test_pauli / aux, dtype=test_pauli.dtype)

    super_op_pauli = stinespring_to_pauli(
        stinespring,
        dim_env,
        nqubits=nqubits,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    backend.assert_allclose(super_op_pauli, test_pauli, atol=PRECISION_TOL)


@pytest.mark.parametrize("nqubits", [None, 1])
@pytest.mark.parametrize("dim_env", [2])
@pytest.mark.parametrize("stinespring", [test_stinespring])
def test_stinespring_to_kraus(backend, stinespring, dim_env, nqubits):
    with pytest.raises(TypeError):
        test = stinespring_to_kraus(stinespring, dim_env=2.0, nqubits=nqubits)
    with pytest.raises(ValueError):
        test = stinespring_to_kraus(stinespring, dim_env=-1, nqubits=nqubits)
    with pytest.raises(ValueError):
        state = random_density_matrix(2, pure=True, backend=backend)
        test = stinespring_to_kraus(
            stinespring,
            dim_env=dim_env,
            initial_state_env=state,
            nqubits=nqubits,
            backend=backend,
        )
    with pytest.raises(TypeError):
        test = stinespring_to_kraus(
            stinespring,
            dim_env=dim_env,
            nqubits=1.0,
            backend=backend,
        )
    with pytest.raises(ValueError):
        test = stinespring_to_kraus(
            stinespring,
            dim_env=dim_env,
            nqubits=-1,
            backend=backend,
        )
    with pytest.raises(ValueError):
        test = stinespring_to_kraus(stinespring, dim_env=3, nqubits=2, backend=backend)

    stinespring = backend.cast(stinespring, dtype=stinespring.dtype)
    test = stinespring_to_kraus(stinespring, dim_env, nqubits=nqubits, backend=backend)

    for kraus, test_kraus in zip(test, [test_a0, test_a1]):
        backend.assert_allclose(kraus, test_kraus, atol=PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [None, 1])
@pytest.mark.parametrize("dim_env", [2])
@pytest.mark.parametrize("stinespring", [test_stinespring])
def test_stinespring_to_chi(
    backend, stinespring, dim_env, nqubits, normalize, order, pauli_order
):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = 1.0 if normalize == False else dim

    stinespring = backend.cast(stinespring, dtype=stinespring.dtype)
    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    chi_matrix = stinespring_to_chi(
        stinespring,
        dim_env,
        nqubits=nqubits,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    backend.assert_allclose(test_chi / aux, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("order", ["row", "column"])
def test_kraus_to_unitaries(backend, order):
    test_a0 = np.sqrt(0.4) * matrices.X
    test_a1 = np.sqrt(0.6) * matrices.Y
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

    backend.assert_allclose(target, operator, atol=2 * PRECISION_TOL)

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

    backend.assert_allclose(reshuffled, test_choi, atol=PRECISION_TOL)
