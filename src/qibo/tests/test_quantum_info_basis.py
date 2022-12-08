import numpy as np
import pytest

from qibo.config import PRECISION_TOL
from qibo.gates.gates import I, X, Y, Z
from qibo.quantum_info import *


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_pauli_basis(backend, nqubits, normalize):
    with pytest.raises(ValueError):
        pauli_basis(-1)
    with pytest.raises(TypeError):
        pauli_basis("1")
    with pytest.raises(TypeError):
        pauli_basis(1, "True")

    single_basis = [
        I(0).asmatrix(backend),
        X(0).asmatrix(backend),
        Y(0).asmatrix(backend),
        Z(0).asmatrix(backend),
    ]

    basis = pauli_basis(nqubits, normalize)
    if nqubits == 1:
        basis_test = single_basis
    else:
        basis_test = list(product(single_basis, repeat=nqubits))
        basis_test = [reduce(np.kron, matrix) for matrix in basis_test]

    print(basis_test)

    if normalize:
        basis_test /= np.sqrt(2**nqubits)

    for pauli, pauli_test in zip(basis, basis_test):
        backend.assert_allclose(
            np.linalg.norm(pauli - pauli_test) < PRECISION_TOL, True
        )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_comp_basis_to_pauli(backend, nqubits, normalize):
    test = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
        ]
    )

    if normalize:
        test /= np.sqrt(2**nqubits)

    test = backend.cast(test, dtype=test.dtype)
    if nqubits == 1:
        matrix = comp_basis_to_pauli(nqubits, normalize, backend)
        backend.assert_allclose(np.linalg.norm(matrix - test) < PRECISION_TOL, True)

    matrix = comp_basis_to_pauli(nqubits, normalize)
