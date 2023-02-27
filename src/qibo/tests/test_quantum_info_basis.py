import numpy as np
import pytest

from qibo import matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import *


@pytest.mark.parametrize("order", ["row", "column", "system"])
@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_pauli_basis(nqubits, normalize, vectorize, order):
    with pytest.raises(ValueError):
        pauli_basis(-1)
    with pytest.raises(TypeError):
        pauli_basis("1")
    with pytest.raises(TypeError):
        pauli_basis(1, "True")
    with pytest.raises(TypeError):
        pauli_basis(1, False, "True")
    with pytest.raises(ValueError):
        pauli_basis(1, False, True)

    basis_test = [matrices.I, matrices.X, matrices.Y, matrices.Z]
    if nqubits >= 2:
        basis_test = list(product(basis_test, repeat=nqubits))
        basis_test = [reduce(np.kron, matrices) for matrices in basis_test]

    if vectorize:
        basis_test = [vectorization(matrix, order=order) for matrix in basis_test]

    basis_test = np.array(basis_test)

    if normalize:
        basis_test /= np.sqrt(2**nqubits)

    basis = pauli_basis(nqubits, normalize, vectorize, order)

    for pauli, pauli_test in zip(basis, basis_test):
        assert np.linalg.norm(pauli - pauli_test) < PRECISION_TOL

    comp_basis_to_pauli(nqubits, normalize, order)
    pauli_to_comp_basis(nqubits, normalize, order)
