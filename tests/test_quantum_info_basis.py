"""Tests for quantum_info.basis module."""

from functools import reduce
from itertools import product

import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import (
    comp_basis_to_pauli,
    pauli_basis,
    pauli_to_comp_basis,
    vectorization,
)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_pauli_basis(
    backend, nqubits, normalize, vectorize, sparse, order, pauli_order
):

    with pytest.raises(TypeError):
        pauli_basis(nqubits=1, normalize=False, pauli_order=1, backend=backend)
    with pytest.raises(ValueError):
        pauli_basis(nqubits=1, normalize=False, pauli_order="IXY", backend=backend)
    with pytest.raises(ValueError):
        pauli_basis(
            nqubits=1, normalize=False, vectorize=True, order=None, backend=backend
        )
    if pauli_order == "IXYZ":
        basis_test = [matrices.I, matrices.X, matrices.Y, matrices.Z]
    else:
        basis_test = [matrices.I, matrices.Z, matrices.X, matrices.Y]
    if nqubits >= 2:
        basis_test = list(product(basis_test, repeat=nqubits))
        basis_test = [reduce(np.kron, matrices) for matrices in basis_test]

    if vectorize:
        basis_test = [
            vectorization(backend.cast(matrix), order=order, backend=backend)
            for matrix in basis_test
        ]

    basis_test = backend.cast(basis_test)

    if normalize:
        basis_test = basis_test / np.sqrt(2**nqubits)

    if vectorize and sparse:
        elements, indexes = [], []
        for row in basis_test:
            row_indexes = list(np.flatnonzero(backend.to_numpy(row)))
            indexes.append(row_indexes)
            elements.append(row[row_indexes])
        indexes = backend.cast(indexes)

    if not vectorize and sparse:
        with pytest.raises(NotImplementedError):
            pauli_basis(nqubits=1, vectorize=False, sparse=True, order="row")
    else:
        basis = pauli_basis(
            nqubits, normalize, vectorize, sparse, order, pauli_order, backend
        )

        if vectorize and sparse:
            for elem_test, ind_test, elem, ind in zip(
                elements, indexes, basis[0], basis[1]
            ):
                backend.assert_allclose(elem_test, elem, atol=PRECISION_TOL)
                backend.assert_allclose(ind_test, ind, atol=PRECISION_TOL)
        else:
            for pauli, pauli_test in zip(basis, basis_test):
                backend.assert_allclose(pauli, pauli_test, atol=PRECISION_TOL)

        comp_basis_to_pauli(nqubits, normalize, sparse, order, pauli_order, backend)
        pauli_to_comp_basis(nqubits, normalize, sparse, order, pauli_order, backend)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "ZYXI"])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_basis_change_matrices(backend, nqubits, order, pauli_order):
    dim = 2**nqubits
    kwargs = {"order": order, "pauli_order": pauli_order, "backend": backend}

    comp_to_pauli = comp_basis_to_pauli(nqubits, normalize=True, **kwargs)
    pauli_to_comp = pauli_to_comp_basis(nqubits, normalize=True, **kwargs)

    identity = backend.cast(np.eye(dim**2), dtype=comp_to_pauli.dtype)
    backend.assert_allclose(comp_to_pauli @ pauli_to_comp, identity, atol=PRECISION_TOL)
    backend.assert_allclose(pauli_to_comp @ comp_to_pauli, identity, atol=PRECISION_TOL)
    backend.assert_allclose(
        pauli_to_comp, backend.conj(comp_to_pauli).T, atol=PRECISION_TOL
    )

    # the Pauli-Liouville representation of a Hermitian operator is real
    state = backend.cast(np.diag(np.arange(1, dim + 1) / np.sum(np.arange(1, dim + 1))))
    state = backend.cast(state, dtype=comp_to_pauli.dtype)
    state_pauli = comp_to_pauli @ vectorization(state, order=order, backend=backend)
    backend.assert_allclose(backend.to_numpy(state_pauli).imag, 0.0, atol=PRECISION_TOL)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "ZYXI"])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("function", [comp_basis_to_pauli, pauli_to_comp_basis])
def test_basis_change_matrices_sparse(backend, function, normalize, order, pauli_order):
    nqubits = 2
    kwargs = {"order": order, "pauli_order": pauli_order, "backend": backend}

    dense = backend.to_numpy(function(nqubits, normalize, **kwargs))
    elements, indexes = function(nqubits, normalize, sparse=True, **kwargs)
    elements, indexes = backend.to_numpy(elements), backend.to_numpy(indexes)

    assert elements.shape == indexes.shape == (4**nqubits, 2**nqubits)

    from_sparse = np.zeros(dense.shape, dtype=dense.dtype)
    np.put_along_axis(from_sparse, indexes.astype(int), elements, axis=1)
    np.testing.assert_allclose(from_sparse, dense, atol=PRECISION_TOL)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("function", [comp_basis_to_pauli, pauli_to_comp_basis])
def test_basis_invalid_arguments(backend, function, sparse):
    with pytest.raises(ValueError):
        function(1, sparse=sparse, order="invalid", backend=backend)
    with pytest.raises(ValueError):
        function(1, sparse=sparse, pauli_order="IXYZZ", backend=backend)
    with pytest.raises(ValueError):
        function(1, sparse=sparse, pauli_order="XXYZ", backend=backend)

    with pytest.raises(ValueError):
        pauli_basis(1, vectorize=True, order="invalid", backend=backend)
    with pytest.raises(ValueError):
        pauli_basis(1, pauli_order="IIXYZ", backend=backend)
