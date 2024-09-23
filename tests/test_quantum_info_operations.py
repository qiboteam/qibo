import numpy as np
import pytest

from qibo import Circuit, gates, matrices
from qibo.quantum_info.linalg_operations import (
    anticommutator,
    commutator,
    matrix_power,
    partial_trace,
    partial_transpose,
)
from qibo.quantum_info.metrics import purity
from qibo.quantum_info.random_ensembles import random_density_matrix, random_statevector


def test_commutator(backend):
    matrix_1 = np.random.rand(2, 2, 2)
    matrix_1 = backend.cast(matrix_1, dtype=matrix_1.dtype)

    matrix_2 = np.random.rand(2, 2)
    matrix_2 = backend.cast(matrix_2, dtype=matrix_2.dtype)

    matrix_3 = np.random.rand(3, 3)
    matrix_3 = backend.cast(matrix_3, dtype=matrix_3.dtype)

    with pytest.raises(TypeError):
        test = commutator(matrix_1, matrix_2)
    with pytest.raises(TypeError):
        test = commutator(matrix_2, matrix_1)
    with pytest.raises(TypeError):
        test = commutator(matrix_2, matrix_3)

    I, X, Y, Z = matrices.I, matrices.X, matrices.Y, matrices.Z
    I = backend.cast(I, dtype=I.dtype)
    X = backend.cast(X, dtype=X.dtype)
    Y = backend.cast(Y, dtype=Y.dtype)
    Z = backend.cast(Z, dtype=Z.dtype)

    comm = commutator(X, I)
    backend.assert_allclose(comm, 0.0)

    comm = commutator(X, X)
    backend.assert_allclose(comm, 0.0)

    comm = commutator(X, Y)
    backend.assert_allclose(comm, 2j * Z)

    comm = commutator(X, Z)
    backend.assert_allclose(comm, -2j * Y)


def test_anticommutator(backend):
    matrix_1 = np.random.rand(2, 2, 2)
    matrix_1 = backend.cast(matrix_1, dtype=matrix_1.dtype)

    matrix_2 = np.random.rand(2, 2)
    matrix_2 = backend.cast(matrix_2, dtype=matrix_2.dtype)

    matrix_3 = np.random.rand(3, 3)
    matrix_3 = backend.cast(matrix_3, dtype=matrix_3.dtype)

    with pytest.raises(TypeError):
        test = anticommutator(matrix_1, matrix_2)
    with pytest.raises(TypeError):
        test = anticommutator(matrix_2, matrix_1)
    with pytest.raises(TypeError):
        test = anticommutator(matrix_2, matrix_3)

    I, X, Y, Z = matrices.I, matrices.X, matrices.Y, matrices.Z
    I = backend.cast(I, dtype=I.dtype)
    X = backend.cast(X, dtype=X.dtype)
    Y = backend.cast(Y, dtype=Y.dtype)
    Z = backend.cast(Z, dtype=Z.dtype)

    anticomm = anticommutator(X, I)
    backend.assert_allclose(anticomm, 2 * X)

    anticomm = anticommutator(X, X)
    backend.assert_allclose(anticomm, 2 * I)

    anticomm = anticommutator(X, Y)
    backend.assert_allclose(anticomm, 0.0)

    anticomm = anticommutator(X, Z)
    backend.assert_allclose(anticomm, 0.0)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_partial_trace(backend, density_matrix):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2).astype(complex)
        state += 1j * np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        test = partial_trace(state, 1, backend=backend)
    with pytest.raises(ValueError):
        state = (
            random_density_matrix(5, backend=backend)
            if density_matrix
            else random_statevector(5, backend=backend)
        )
        test = partial_trace(state, 1, backend=backend)

    nqubits = 4

    circuit = Circuit(nqubits, density_matrix=density_matrix)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, qubit + 1) for qubit in range(1, nqubits - 1))
    state = backend.execute_circuit(circuit).state()

    traced = partial_trace(state, (1, 2, 3), backend=backend)

    Id = backend.identity_density_matrix(1, normalize=True)

    backend.assert_allclose(traced, Id)


def _werner_state(p, backend):
    zero, one = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    psi = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)
    psi = np.outer(psi, np.conj(psi.T))
    psi = backend.cast(psi, dtype=psi.dtype)

    state = p * psi + (1 - p) * backend.identity_density_matrix(2, normalize=True)

    # partial transpose of two-qubit werner state is known analytically
    transposed = (1 / 4) * np.array(
        [
            [1 - p, 0, 0, -2 * p],
            [0, p + 1, 0, 0],
            [0, 0, p + 1, 0],
            [-2 * p, 0, 0, 1 - p],
        ],
        dtype=complex,
    )
    transposed = backend.cast(transposed, dtype=transposed.dtype)

    return state, transposed


@pytest.mark.parametrize("statevector", [False, True])
@pytest.mark.parametrize("p", [1 / 5, 1 / 3, 1.0])
def test_partial_transpose(backend, p, statevector):
    with pytest.raises(ValueError):
        state = random_density_matrix(3, backend=backend)
        test = partial_transpose(state, [0], backend)

    zero, one = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    psi = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)

    if statevector:
        # testing statevector
        target = np.zeros((4, 4), dtype=complex)
        target[0, 3] = -1 / 2
        target[1, 1] = 1 / 2
        target[2, 2] = 1 / 2
        target[3, 0] = -1 / 2
        target = backend.cast(target, dtype=target.dtype)

        psi = backend.cast(psi, dtype=psi.dtype)

        transposed = partial_transpose(psi, [0], backend=backend)
        backend.assert_allclose(transposed, target)
    else:
        psi = np.outer(psi, np.conj(psi.T))
        psi = backend.cast(psi, dtype=psi.dtype)

        state = p * psi + (1 - p) * backend.identity_density_matrix(2, normalize=True)

        # partial transpose of two-qubit werner state is known analytically
        target = (1 / 4) * np.array(
            [
                [1 - p, 0, 0, -2 * p],
                [0, p + 1, 0, 0],
                [0, 0, p + 1, 0],
                [-2 * p, 0, 0, 1 - p],
            ],
            dtype=complex,
        )
        target = backend.cast(target, dtype=target.dtype)

        transposed = partial_transpose(state, [1], backend)
        backend.assert_allclose(transposed, target)


@pytest.mark.parametrize("power", [2, 2.0, "2"])
def test_matrix_power(backend, power):
    nqubits = 2
    dims = 2**nqubits

    state = random_density_matrix(dims, backend=backend)

    if isinstance(power, str):
        with pytest.raises(TypeError):
            test = matrix_power(state, power, backend)
    else:
        power = matrix_power(state, power, backend)

        backend.assert_allclose(
            float(backend.np.real(backend.np.trace(power))),
            purity(state, backend=backend),
        )
