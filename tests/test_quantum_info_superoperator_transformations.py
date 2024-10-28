import numpy as np
import pytest  # type: ignore

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info.linalg_operations import partial_trace
from qibo.quantum_info.random_ensembles import random_density_matrix, random_statevector
from qibo.quantum_info.superoperator_transformations import (
    chi_to_choi,
    chi_to_kraus,
    chi_to_liouville,
    chi_to_pauli,
    chi_to_stinespring,
    choi_to_chi,
    choi_to_kraus,
    choi_to_liouville,
    choi_to_pauli,
    choi_to_stinespring,
    kraus_to_chi,
    kraus_to_choi,
    kraus_to_liouville,
    kraus_to_pauli,
    kraus_to_stinespring,
    kraus_to_unitaries,
    liouville_to_chi,
    liouville_to_choi,
    liouville_to_kraus,
    liouville_to_pauli,
    liouville_to_stinespring,
    pauli_to_chi,
    pauli_to_choi,
    pauli_to_kraus,
    pauli_to_liouville,
    pauli_to_stinespring,
    stinespring_to_chi,
    stinespring_to_choi,
    stinespring_to_kraus,
    stinespring_to_liouville,
    stinespring_to_pauli,
    to_chi,
    to_choi,
    to_liouville,
    to_pauli_liouville,
    to_stinespring,
    unvectorization,
    vectorization,
)


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
@pytest.mark.parametrize("statevector", [True, False])
def test_vectorization(backend, nqubits, order, statevector):
    with pytest.raises(TypeError):
        x = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        x = backend.cast(x)
        vectorization(x, backend=backend)
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
        if statevector:
            if order == "system":
                matrix_test = [0, 0, 0, 1, 0, 0, 2, 3, 0, 2, 0, 3, 4, 6, 6, 9]
            else:
                matrix_test = [0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9]
        else:
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
    if statevector and nqubits == 1:
        matrix = np.arange(dim**2)
    else:
        matrix = np.arange(dim**2).reshape((dim, dim))
    matrix = vectorization(backend.cast(matrix), order, backend=backend)
    backend.assert_allclose(matrix, matrix_test, atol=PRECISION_TOL)


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
@pytest.mark.parametrize("statevector", [True, False])
def test_batched_vectorization(backend, nqubits, order, statevector):
    if statevector:
        state = backend.cast(
            [random_statevector(2**nqubits, 42, backend=backend) for _ in range(3)]
        ).reshape(3, 1, -1)
    else:
        state = backend.cast(
            [
                random_density_matrix(2**nqubits, seed=42, backend=backend)
                for _ in range(3)
            ]
        )

    batched_vec = vectorization(state, order=order, backend=backend)
    for i, element in enumerate(state):
        if statevector:
            element = element.ravel()
        backend.assert_allclose(
            batched_vec[i], vectorization(element, order=order, backend=backend)
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


@pytest.mark.parametrize("order", ["row", "column"])
def test_to_choi(backend, order):
    test_a0_ = backend.cast(test_a0)
    test_a1_ = backend.cast(test_a1)
    choi_a0 = to_choi(test_a0_, order=order, backend=backend)
    choi_a1 = to_choi(test_a1_, order=order, backend=backend)
    choi = choi_a0 + choi_a1

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4)
    test_choi = np.swapaxes(test_choi, *axes)
    test_choi = np.reshape(test_choi, (4, 4))

    backend.assert_allclose(choi, test_choi, atol=PRECISION_TOL)


@pytest.mark.parametrize("order", ["row", "column"])
def test_to_liouville(backend, order):
    test_a0_ = backend.cast(test_a0)
    test_a1_ = backend.cast(test_a1)
    liouville_a0 = to_liouville(test_a0_, order=order, backend=backend)
    liouville_a1 = to_liouville(test_a1_, order=order, backend=backend)
    liouville = liouville_a0 + liouville_a1

    backend.assert_allclose(liouville, test_superop, atol=PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_to_pauli_liouville(backend, normalize, order, pauli_order):
    test_a0_ = backend.cast(test_a0)
    test_a1_ = backend.cast(test_a1)
    pauli_a0 = to_pauli_liouville(
        test_a0_,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    pauli_a1 = to_pauli_liouville(
        test_a1_,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    pauli = pauli_a0 + pauli_a1

    test_pauli = pauli_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = 1.0 if not normalize else dim
    test_pauli = backend.cast(test_pauli / aux, dtype=test_pauli.dtype)

    backend.assert_allclose(pauli, test_pauli, atol=PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_to_chi(backend, normalize, order, pauli_order):
    test_a0_ = backend.cast(test_a0)
    test_a1_ = backend.cast(test_a1)
    chi_a0 = to_chi(
        test_a0_,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    chi_a1 = to_chi(
        test_a1_,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )
    chi = chi_a0 + chi_a1

    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = 1.0 if not normalize else dim
    test_chi = backend.cast(test_chi / aux, dtype=test_chi.dtype)

    backend.assert_allclose(chi, test_chi, atol=PRECISION_TOL)


@pytest.mark.parametrize("partition", [None, (0,)])
@pytest.mark.parametrize("test_a0", [test_a0])
def test_to_stinespring(backend, test_a0, partition):
    test_a0_ = backend.cast(test_a0)
    state = random_density_matrix(2, seed=8, backend=backend)

    target = test_a0_ @ state @ backend.np.conj(test_a0_.T)

    environment = (1, 2)

    global_state = backend.identity_density_matrix(len(environment), normalize=True)
    global_state = backend.np.kron(state, global_state)

    stinespring = to_stinespring(
        test_a0_, partition=partition, nqubits=len(environment) + 1, backend=backend
    )
    stinespring = stinespring @ global_state @ backend.np.conj(stinespring.T)
    stinespring = partial_trace(stinespring, traced_qubits=environment, backend=backend)

    backend.assert_allclose(stinespring, target, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_liouville(backend, order, test_superop):
    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.cast(
        np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    )

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
    aux = 1 if not normalize else dim

    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.np.swapaxes(
        backend.np.reshape(test_superop * aux, [2] * 4), *axes
    ).reshape([4, 4])

    pauli_op = choi_to_pauli(
        test_choi, normalize, order, pauli_order=pauli_order, backend=backend
    )

    backend.assert_allclose(test_pauli, pauli_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_non_CP", [test_non_CP])
@pytest.mark.parametrize("test_kraus_right", [test_kraus_right])
@pytest.mark.parametrize("test_kraus_left", [test_kraus_left])
@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("validate_cp", [False, True])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_kraus(
    backend,
    order,
    validate_cp,
    test_a0,
    test_a1,
    test_kraus_left,
    test_kraus_right,
    test_non_CP,
):
    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.cast(
        np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    )

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

    test_kraus_left = backend.cast(test_kraus_left, dtype=backend.dtype)
    test_kraus_right = backend.cast(test_kraus_right, dtype=backend.dtype)

    state = random_density_matrix(2, backend=backend)

    evolution_a0 = a0 @ state @ backend.np.conj(a0).T
    evolution_a1 = a1 @ state @ backend.np.conj(a1).T

    test_evolution_a0 = test_a0 @ state @ backend.np.conj(test_a0).T
    test_evolution_a1 = test_a1 @ state @ backend.np.conj(test_a1).T

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=2 * PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=2 * PRECISION_TOL)

    if validate_cp and order == "row":
        test_non_CP = backend.cast(test_non_CP, dtype=test_non_CP.dtype)
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
            evolution = left @ state @ backend.np.conj(right).T
            test_evolution = (
                test_coeff**2 * test_left @ state @ backend.np.conj(test_right).T
            )

            backend.assert_allclose(evolution, test_evolution, atol=2 * PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_choi_to_chi(backend, normalize, order, pauli_order, test_superop):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = 1 if not normalize else dim

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.np.swapaxes(
        backend.np.reshape(test_superop * aux, [2] * 4), *axes
    ).reshape([4, 4])

    chi_matrix = choi_to_chi(
        test_choi,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        backend=backend,
    )

    backend.assert_allclose(test_chi, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("test_stinespring", [test_stinespring])
@pytest.mark.parametrize("nqubits", [None, 1])
@pytest.mark.parametrize("validate_cp", [False, True])
@pytest.mark.parametrize("order", ["row", "column"])
def test_choi_to_stinespring(
    backend, order, validate_cp, nqubits, test_stinespring, test_a0, test_a1
):
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
    test_a0 = backend.cast(test_a0, dtype=test_a0.dtype)
    test_a1 = backend.cast(test_a1, dtype=test_a1.dtype)

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
    if nqubits is None:
        nqubits = 1

    v_0 = np.array([1, 0], dtype=complex)
    v_0 = backend.cast(v_0, dtype=v_0.dtype)

    state = random_density_matrix(2**nqubits, backend=backend)

    # action of Kraus channel on state
    state_final_test = test_a0 @ state @ test_a0 + test_a1 @ state @ test_a1

    # action of stinespring channel on state + environment
    stinespring = (
        stinespring
        @ backend.np.kron(state, backend.np.outer(v_0, v_0))
        @ backend.np.conj(stinespring).T
    )

    # partial trace of the environment
    stinespring = backend.np.reshape(stinespring, (2**nqubits, 2, 2**nqubits, 2))
    stinespring = backend.np.swapaxes(stinespring, 1, 2)
    state_final = np.zeros((2**nqubits, 2**nqubits), dtype=complex)
    state_final = backend.cast(state_final, dtype=state_final.dtype)
    for alpha in range(2):
        vector_alpha = np.zeros(2, dtype=complex)
        vector_alpha[alpha] = 1.0
        vector_alpha = backend.cast(vector_alpha, dtype=vector_alpha.dtype)
        state_final = (
            state_final + backend.np.conj(vector_alpha) @ stinespring @ vector_alpha
        )

    backend.assert_allclose(state_final, state_final_test, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_choi(backend, order, test_superop):
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    choi = liouville_to_choi(test_superop, order=order, backend=backend)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.np.reshape(test_superop, [2] * 4)
    test_choi = backend.np.swapaxes(test_choi, *axes)
    test_choi = backend.np.reshape(test_choi, (4, 4))

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
    aux = 1.0 if not normalize else dim

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

    evolution_a0 = a0 @ state @ backend.np.conj(a0).T
    evolution_a1 = a1 @ state @ backend.np.conj(a1).T

    test_evolution_a0 = test_a0 @ state @ backend.np.conj(test_a0).T
    test_evolution_a1 = test_a1 @ state @ backend.np.conj(test_a1).T

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_liouville_to_chi(backend, normalize, order, pauli_order, test_superop):
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = 1.0 if not normalize else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)
    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)

    chi_matrix = liouville_to_chi(
        test_superop, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_chi / aux, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("test_stinespring", [test_stinespring])
@pytest.mark.parametrize("nqubits", [1])
@pytest.mark.parametrize("validate_cp", [False, True])
@pytest.mark.parametrize("order", ["row", "column"])
def test_liouville_to_stinespring(
    backend,
    order,
    validate_cp,
    nqubits,
    test_stinespring,
    test_superop,
    test_a0,
    test_a1,
):
    test_stinespring = backend.cast(test_stinespring, dtype=test_stinespring.dtype)
    test_a0 = backend.cast(test_a0, dtype=test_a0.dtype)
    test_a1 = backend.cast(test_a1, dtype=test_a1.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(*axes).reshape([4, 4])
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)
    stinespring = liouville_to_stinespring(
        test_superop,
        order=order,
        validate_cp=validate_cp,
        nqubits=nqubits,
        backend=backend,
    )

    v_0 = np.array([1, 0], dtype=complex)
    v_0 = backend.cast(v_0, dtype=v_0.dtype)

    state = random_density_matrix(2**nqubits, backend=backend)

    # action of Kraus channel on state
    state_final_test = test_a0 @ state @ test_a0 + test_a1 @ state @ test_a1

    # action of stinespring channel on state + environment
    stinespring = (
        stinespring
        @ backend.np.kron(state, backend.np.outer(v_0, v_0))
        @ backend.np.conj(stinespring).T
    )

    # partial trace of the environment
    stinespring = backend.np.reshape(stinespring, (2**nqubits, 2, 2**nqubits, 2))
    stinespring = backend.np.swapaxes(stinespring, 1, 2)
    state_final = np.zeros((2**nqubits, 2**nqubits), dtype=complex)
    state_final = backend.cast(state_final, dtype=state_final.dtype)
    for alpha in range(2):
        vector_alpha = np.zeros(2, dtype=complex)
        vector_alpha[alpha] = 1.0
        vector_alpha = backend.cast(vector_alpha, dtype=vector_alpha.dtype)
        state_final = (
            state_final + backend.np.conj(vector_alpha) @ stinespring @ vector_alpha
        )

    backend.assert_allclose(state_final, state_final_test, atol=PRECISION_TOL)


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
    aux = 1.0 if not normalize else dim

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
    aux = 1.0 if not normalize else dim

    test_chi = backend.cast(test_chi, dtype=test_chi.dtype)

    chi_matrix = kraus_to_chi(
        test_kraus, normalize, order, pauli_order, backend=backend
    )

    backend.assert_allclose(test_chi / aux, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("nqubits", [None, 1])
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


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_liouville(backend, normalize, order, pauli_order, test_superop):
    test_pauli = pauli_superop(pauli_order)

    with pytest.raises(ValueError):
        pauli_to_liouville(test_pauli[:-1, :], normalize, order, backend=backend)

    dim = int(np.sqrt(test_superop.shape[0]))
    aux = dim**2 if not normalize else dim

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
    aux = dim**2 if not normalize else dim

    test_pauli = backend.cast(test_pauli, dtype=backend.dtype)
    test_superop = backend.cast(test_superop, dtype=backend.dtype)
    choi_super_op = pauli_to_choi(
        test_pauli / aux, normalize, order, pauli_order, backend=backend
    )

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.np.swapaxes(backend.np.reshape(test_superop, [2] * 4), *axes)
    test_choi = backend.np.reshape(test_choi, (4, 4))

    backend.assert_allclose(test_choi, choi_super_op, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_kraus(backend, normalize, order, pauli_order, test_a0, test_a1):
    test_pauli = pauli_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if not normalize else dim

    test_pauli = backend.cast(test_pauli, dtype=backend.dtype)

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

    evolution_a0 = a0 @ state @ backend.np.conj(a0).T
    evolution_a1 = a1 @ state @ backend.np.conj(a1).T

    test_evolution_a0 = test_a0 @ state @ backend.np.conj(test_a0).T
    test_evolution_a1 = test_a1 @ state @ backend.np.conj(test_a1).T

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=2 * PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=2 * PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_pauli_to_chi(backend, normalize, order, pauli_order):
    test_pauli = pauli_superop(pauli_order)
    test_chi = chi_superop(pauli_order)
    dim = int(np.sqrt(test_pauli.shape[0]))
    aux = dim**2 if not normalize else dim

    test_chi = backend.cast(test_chi, dtype=backend.dtype)
    test_pauli = backend.cast(test_pauli / aux, dtype=backend.dtype)

    chi_matrix = pauli_to_chi(
        test_pauli, normalize, order, pauli_order, backend=backend
    )

    aux = 1.0 if not normalize else dim
    backend.assert_allclose(test_chi / aux, chi_matrix, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1])
@pytest.mark.parametrize("validate_cp", [False, True])
@pytest.mark.parametrize("order", ["row", "column"])
def test_pauli_to_stinespring(
    backend,
    order,
    validate_cp,
    nqubits,
    normalize,
    pauli_order,
    test_a0,
    test_a1,
):
    test_pauli = pauli_superop(pauli_order)
    test_pauli = backend.cast(test_pauli, dtype=backend.dtype)

    dim = 2**nqubits
    aux = dim**2 if normalize is False else dim
    test_a0 = backend.cast(test_a0, dtype=backend.dtype)
    test_a1 = backend.cast(test_a1, dtype=backend.dtype)

    stinespring = pauli_to_stinespring(
        test_pauli,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        validate_cp=validate_cp,
        nqubits=nqubits,
        backend=backend,
    )

    v_0 = np.array([1, 0], dtype=complex)
    v_0 = backend.cast(v_0, dtype=v_0.dtype)

    state = random_density_matrix(2**nqubits, backend=backend)

    # action of Kraus channel on state
    state_final_test = test_a0 @ state @ test_a0 + test_a1 @ state @ test_a1

    # action of stinespring channel on state + environment
    stinespring = (
        stinespring
        @ backend.np.kron(state, backend.np.outer(v_0, v_0))
        @ backend.np.conj(stinespring).T
    )

    # partial trace of the environment
    stinespring = backend.np.reshape(stinespring, (2**nqubits, 2, 2**nqubits, 2))
    stinespring = backend.np.swapaxes(stinespring, 1, 2)
    state_final = np.zeros((2**nqubits, 2**nqubits), dtype=complex)
    state_final = backend.cast(state_final, dtype=state_final.dtype)
    for alpha in range(2):
        vector_alpha = np.zeros(2, dtype=complex)
        vector_alpha[alpha] = 1.0
        vector_alpha = backend.cast(vector_alpha, dtype=vector_alpha.dtype)
        state_final = (
            state_final + backend.np.conj(vector_alpha) @ stinespring @ vector_alpha
        )

    backend.assert_allclose(state_final, aux * state_final_test, atol=PRECISION_TOL)


@pytest.mark.parametrize("test_superop", [test_superop])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_choi(backend, normalize, order, pauli_order, test_superop):
    test_chi = chi_superop(pauli_order=pauli_order)
    dim = int(np.sqrt(test_chi.shape[0]))
    aux = dim**2 if not normalize else dim

    test_chi = backend.cast(test_chi, dtype=backend.dtype)
    test_superop = backend.cast(test_superop, dtype=backend.dtype)

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.np.swapaxes(
        backend.np.reshape(test_superop, [2] * 4), *axes
    ).reshape([4, 4])

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
    aux = dim**2 if not normalize else dim

    test_chi = backend.cast(test_chi, dtype=backend.dtype)
    test_superop = backend.cast(test_superop, dtype=backend.dtype)

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
    aux = 1.0 if normalize else dim**2

    test_chi = backend.cast(test_chi, dtype=backend.dtype)
    test_pauli = backend.cast(test_pauli, dtype=backend.dtype)

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
    aux = dim**2 if not normalize else dim

    test_chi = backend.cast(test_chi, dtype=backend.dtype)

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

    evolution_a0 = a0 @ state @ backend.np.conj(a0).T
    evolution_a1 = a1 @ state @ backend.np.conj(a1).T

    test_evolution_a0 = test_a0 @ state @ backend.np.conj(test_a0).T
    test_evolution_a1 = test_a1 @ state @ backend.np.conj(test_a1).T

    backend.assert_allclose(evolution_a0, test_evolution_a0, atol=2 * PRECISION_TOL)
    backend.assert_allclose(evolution_a1, test_evolution_a1, atol=2 * PRECISION_TOL)


@pytest.mark.parametrize("test_a1", [test_a1])
@pytest.mark.parametrize("test_a0", [test_a0])
@pytest.mark.parametrize("nqubits", [1])
@pytest.mark.parametrize("validate_cp", [False, True])
@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("normalize", [False, True])
def test_chi_to_stinespring(
    backend, normalize, order, pauli_order, validate_cp, nqubits, test_a0, test_a1
):
    test_chi = chi_superop(pauli_order)
    test_chi = backend.cast(test_chi, dtype=backend.dtype)

    dim = int(np.sqrt(test_chi.shape[0]))
    aux = dim**2 if not normalize else dim
    test_a0 = backend.cast(test_a0, dtype=backend.dtype)
    test_a1 = backend.cast(test_a1, dtype=backend.dtype)

    stinespring = chi_to_stinespring(
        test_chi,
        normalize=normalize,
        order=order,
        pauli_order=pauli_order,
        validate_cp=validate_cp,
        nqubits=nqubits,
        backend=backend,
    )

    v_0 = np.array([1, 0], dtype=complex)
    v_0 = backend.cast(v_0, dtype=v_0.dtype)

    state = random_density_matrix(2**nqubits, backend=backend)

    # action of Kraus channel on state
    state_final_test = test_a0 @ state @ test_a0 + test_a1 @ state @ test_a1

    # action of stinespring channel on state + environment
    stinespring = (
        stinespring
        @ backend.np.kron(state, backend.np.outer(v_0, v_0))
        @ backend.np.conj(stinespring).T
    )

    # partial trace of the environment
    stinespring = backend.np.reshape(stinespring, (2**nqubits, 2, 2**nqubits, 2))
    stinespring = backend.np.swapaxes(stinespring, 1, 2)
    state_final = np.zeros((2**nqubits, 2**nqubits), dtype=complex)
    state_final = backend.cast(state_final, dtype=state_final.dtype)
    for alpha in range(2):
        vector_alpha = np.zeros(2, dtype=complex)
        vector_alpha[alpha] = 1.0
        vector_alpha = backend.cast(vector_alpha, dtype=vector_alpha.dtype)
        state_final = (
            state_final + backend.np.conj(vector_alpha) @ stinespring @ vector_alpha
        )

    backend.assert_allclose(state_final, aux * state_final_test, atol=PRECISION_TOL)


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
    aux = 1.0 if not normalize else dim

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
    aux = 1.0 if not normalize else dim

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
        np.linalg.norm(backend.to_numpy(reshuffled) - backend.to_numpy(test_superop))
        < PRECISION_TOL,
        True,
    )

    axes = [1, 2] if order == "row" else [0, 3]
    test_choi = backend.np.reshape(
        backend.np.swapaxes(backend.np.reshape(test_superop, [2] * 4), *axes), [4, 4]
    )

    reshuffled = _reshuffling(test_choi, order, backend=backend)
    reshuffled = _reshuffling(reshuffled, order, backend=backend)

    backend.assert_allclose(reshuffled, test_choi, atol=PRECISION_TOL)
