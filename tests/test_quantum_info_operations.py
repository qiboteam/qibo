import numpy as np
import pytest

from qibo import matrices
from qibo.quantum_info.operations import anticommutator, commutator


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
        test = commutator(matrix_2, matrix_3)

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
