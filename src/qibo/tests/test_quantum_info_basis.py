import numpy as np
import pytest

from qibo.config import PRECISION_TOL
from qibo.gates.gates import I, X, Y, Z
from qibo.quantum_info import *


@pytest.mark.parametrize("vectorize", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_pauli_basis(backend, nqubits, normalize, vectorize):
    with pytest.raises(ValueError):
        pauli_basis(-1)
    with pytest.raises(TypeError):
        pauli_basis("1")
    with pytest.raises(TypeError):
        pauli_basis(1, "True")
    with pytest.raises(TypeError):
        pauli_basis(1, False, "True")

    single_basis = [
        I(0).asmatrix(backend),
        X(0).asmatrix(backend),
        Y(0).asmatrix(backend),
        Z(0).asmatrix(backend),
    ]

    if vectorize:
        single_basis = [matrix.reshape((1, -1), order="F") for matrix in single_basis]
        print(single_basis)

    if nqubits == 1:
        basis_test = single_basis
    else:
        basis_test = list(product(single_basis, repeat=nqubits))
        if vectorize:
            basis_test = [reduce(np.outer, matrix).ravel() for matrix in basis_test]
        else:
            basis_test = [reduce(np.kron, matrix) for matrix in basis_test]

    if normalize:
        basis_test /= np.sqrt(2**nqubits)

    basis = pauli_basis(nqubits, normalize, vectorize)

    for pauli, pauli_test in zip(basis, basis_test):
        backend.assert_allclose(
            np.linalg.norm(pauli - pauli_test) < PRECISION_TOL, True
        )

    comp_basis_to_pauli(nqubits, normalize)
    pauli_to_comp_basis(nqubits, normalize)
