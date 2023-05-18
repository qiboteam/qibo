import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_pauli_basis(
    backend, nqubits, normalize, vectorize, sparse, order, pauli_order
):
    with pytest.raises(ValueError):
        pauli_basis(nqubits=-1, backend=backend)
    with pytest.raises(TypeError):
        pauli_basis(nqubits="1", backend=backend)
    with pytest.raises(TypeError):
        pauli_basis(nqubits=1, normalize="True", backend=backend)
    with pytest.raises(TypeError):
        pauli_basis(nqubits=1, normalize=False, vectorize="True", backend=backend)
    with pytest.raises(TypeError):
        pauli_basis(nqubits=1, normalize=False, sparse="True", backend=backend)
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
        basis_test = [vectorization(matrix, order=order) for matrix in basis_test]

    basis_test = np.array(basis_test)

    if normalize:
        basis_test /= np.sqrt(2**nqubits)

    basis_test = backend.cast(basis_test, dtype=basis_test.dtype)

    if vectorize and sparse:
        elements, indexes = [], []
        for row in basis_test:
            row_indexes = list(np.flatnonzero(row))
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
                backend.assert_allclose(
                    np.linalg.norm(elem_test - elem) < PRECISION_TOL, True
                )
                backend.assert_allclose(
                    np.linalg.norm(ind_test - ind) < PRECISION_TOL, True
                )
        else:
            for pauli, pauli_test in zip(basis, basis_test):
                backend.assert_allclose(
                    np.linalg.norm(pauli - pauli_test) < PRECISION_TOL, True
                )

        comp_basis_to_pauli(nqubits, normalize, sparse, order, pauli_order, backend)
        pauli_to_comp_basis(nqubits, normalize, sparse, order, pauli_order, backend)
