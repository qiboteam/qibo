import numpy as np
import pytest

from qibo import Circuit, gates, matrices
from qibo.quantum_info.linalg_operations import (
    anticommutator,
    commutator,
    matrix_power,
    partial_trace,
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
