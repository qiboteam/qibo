"""Test dense matrix of Hamiltonians constructed using symbols."""
import pytest
import numpy as np
import sympy
from qibo import hamiltonians, matrices
from qibo.symbols import I, X, Y, Z, Symbol
from qibo.tests.utils import random_hermitian


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("hamtype", ["normal", "symbolic"])
@pytest.mark.parametrize("calcterms", [False, True])
def test_tfim_hamiltonian_from_symbols(backend, nqubits, hamtype, calcterms):
    """Check creating TFIM Hamiltonian using sympy."""
    if hamtype == "symbolic":
        h = 0.5
        symham = sum(Z(i) * Z(i + 1) for i in range(nqubits - 1))
        symham += Z(0) * Z(nqubits - 1)
        symham += h * sum(X(i) for i in range(nqubits))
        ham = hamiltonians.SymbolicHamiltonian(-symham, backend=backend)
    else:
        h = 0.5
        z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(nqubits))))
        x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(nqubits))))

        symham = sum(z_symbols[i] * z_symbols[i + 1] for i in range(nqubits - 1))
        symham += z_symbols[0] * z_symbols[-1]
        symham += h * sum(x_symbols)
        symmap = {z: (i, matrices.Z) for i, z in enumerate(z_symbols)}
        symmap.update({x: (i, matrices.X) for i, x in enumerate(x_symbols)})
        ham = hamiltonians.Hamiltonian.from_symbolic(-symham, symmap, backend=backend)

    if calcterms:
        _ = ham.terms
    final_matrix = ham.matrix
    target_matrix = hamiltonians.TFIM(nqubits, h=h, backend=backend).matrix
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("hamtype", ["normal", "symbolic"])
@pytest.mark.parametrize("calcterms", [False, True])
def test_from_symbolic_with_power(backend, hamtype, calcterms):
    """Check ``from_symbolic`` when the expression contains powers."""
    if hamtype == "symbolic":
        matrix = random_hermitian(1)
        symham =  (Symbol(0, matrix) ** 2 - Symbol(1, matrix) ** 2 +
                   3 * Symbol(1, matrix) - 2 * Symbol(0, matrix) * Symbol(2, matrix) + 1)
        ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)
    else:
        z = sympy.symbols(" ".join((f"Z{i}" for i in range(3))))
        symham =  z[0] ** 2 - z[1] ** 2 + 3 * z[1] - 2 * z[0] * z[2] + 1
        matrix = random_hermitian(1)
        symmap = {x: (i, matrix) for i, x in enumerate(z)}
        ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap, backend=backend)

    if calcterms:
        _ = ham.terms
    final_matrix = ham.matrix
    matrix2 = matrix.dot(matrix)
    eye = np.eye(2, dtype=matrix.dtype)
    target_matrix = np.kron(np.kron(matrix2, eye), eye)
    target_matrix -= np.kron(np.kron(eye, matrix2), eye)
    target_matrix += 3 * np.kron(np.kron(eye, matrix), eye)
    target_matrix -= 2 * np.kron(np.kron(matrix, eye), matrix)
    target_matrix += np.eye(8, dtype=matrix.dtype)
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("hamtype", ["normal", "symbolic"])
@pytest.mark.parametrize("calcterms", [False, True])
def test_from_symbolic_with_complex_numbers(backend, hamtype, calcterms):
    """Check ``from_symbolic`` when the expression contains imaginary unit."""
    if hamtype == "symbolic":
        symham = (1 + 2j) * X(0) * X(1) + 2 * Y(0) * Y(1) - 3j * X(0) * Y(1) + 1j * Y(0) * X(1)
        ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)
    else:
        x = sympy.symbols(" ".join((f"X{i}" for i in range(2))))
        y = sympy.symbols(" ".join((f"Y{i}" for i in range(2))))
        symham = (1 + 2j) * x[0] * x[1] + 2 * y[0] * y[1] - 3j * x[0] * y[1] + 1j * y[0] * x[1]
        symmap = {s: (i, matrices.X) for i, s in enumerate(x)}
        symmap.update({s: (i, matrices.Y) for i, s in enumerate(y)})
        ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap, backend=backend)

    if calcterms:
        _ = ham.terms
    final_matrix = ham.matrix
    target_matrix = (1 + 2j) * np.kron(matrices.X, matrices.X)
    target_matrix += 2 * np.kron(matrices.Y, matrices.Y)
    target_matrix -= 3j * np.kron(matrices.X, matrices.Y)
    target_matrix += 1j * np.kron(matrices.Y, matrices.X)
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("calcterms", [False, True])
def test_from_symbolic_application_hamiltonian(backend, calcterms):
    """Check ``from_symbolic`` for a specific four-qubit Hamiltonian."""
    z1, z2, z3, z4 = sympy.symbols("z1 z2 z3 z4")
    symmap = {z: (i, matrices.Z) for i, z in enumerate([z1, z2, z3, z4])}
    symham = (z1 * z2 - 0.5 * z1 * z3 + 2 * z2 * z3 + 0.35 * z2
              + 0.25 * z3 * z4 + 0.5 * z3 + z4 - z1)
    # Check that Trotter dense matrix agrees will full Hamiltonian matrix
    fham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap, backend=backend)
    symham = (Z(0) * Z(1) - 0.5 * Z(0) * Z(2) + 2 * Z(1) * Z(2) + 0.35 * Z(1)
              + 0.25 * Z(2) * Z(3) + 0.5 * Z(2) + Z(3) - Z(0))
    sham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)
    if calcterms:
        _ = sham.terms
    backend.assert_allclose(sham.matrix, fham.matrix)


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("hamtype", ["normal", "symbolic"])
@pytest.mark.parametrize("calcterms", [False, True])
def test_x_hamiltonian_from_symbols(backend, nqubits, hamtype, calcterms):
    """Check creating sum(X) Hamiltonian using sympy."""
    if hamtype == "symbolic":
        symham = -sum(X(i) for i in range(nqubits))
        ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)
    else:
        x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(nqubits))))
        symham =  -sum(x_symbols)
        symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}
        ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap, backend=backend)
    if calcterms:
        _ = ham.terms
    final_matrix = ham.matrix
    target_matrix = hamiltonians.X(nqubits, backend=backend).matrix
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("hamtype", ["normal", "symbolic"])
@pytest.mark.parametrize("calcterms", [False, True])
def test_three_qubit_term_hamiltonian_from_symbols(backend, hamtype, calcterms):
    """Check creating Hamiltonian with three-qubit interaction using sympy."""
    if hamtype == "symbolic":
        symham = X(0) * Y(1) * Z(2) + 0.5 * Y(0) * Z(1) * X(3) + Z(0) * X(2)
        symham += Y(2) + 1.5 * Z(1) - 2 - 3 * X(1) * Y(3)
        ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)
    else:
        x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(4))))
        y_symbols = sympy.symbols(" ".join((f"Y{i}" for i in range(4))))
        z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(4))))
        symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}
        symmap.update({x: (i, matrices.Y) for i, x in enumerate(y_symbols)})
        symmap.update({x: (i, matrices.Z) for i, x in enumerate(z_symbols)})

        symham = x_symbols[0] * y_symbols[1] * z_symbols[2]
        symham += 0.5 * y_symbols[0] * z_symbols[1] * x_symbols[3]
        symham += z_symbols[0] * x_symbols[2]
        symham += -3 * x_symbols[1] * y_symbols[3]
        symham += y_symbols[2]
        symham += 1.5 * z_symbols[1]
        symham -= 2
        ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap, backend=backend)

    if calcterms:
        _ = ham.terms
    final_matrix = ham.matrix
    target_matrix = np.kron(np.kron(matrices.X, matrices.Y),
                            np.kron(matrices.Z, matrices.I))
    target_matrix += 0.5 * np.kron(np.kron(matrices.Y, matrices.Z),
                                   np.kron(matrices.I, matrices.X))
    target_matrix += np.kron(np.kron(matrices.Z, matrices.I),
                             np.kron(matrices.X, matrices.I))
    target_matrix += -3 * np.kron(np.kron(matrices.I, matrices.X),
                             np.kron(matrices.I, matrices.Y))
    target_matrix += np.kron(np.kron(matrices.I, matrices.I),
                             np.kron(matrices.Y, matrices.I))
    target_matrix += 1.5 * np.kron(np.kron(matrices.I, matrices.Z),
                                   np.kron(matrices.I, matrices.I))
    target_matrix -= 2 * np.eye(2**4, dtype=target_matrix.dtype)
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("calcterms", [False, True])
def test_hamiltonian_with_identity_symbol(backend, calcterms):
    """Check creating Hamiltonian from expression which contains the identity symbol."""
    symham = X(0) * I(1) * Z(2) + 0.5 * Y(0) * Z(1) * I(3) + Z(0) * I(1) * X(2)
    ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)

    if calcterms:
        _ = ham.terms
    final_matrix = ham.matrix
    target_matrix = np.kron(np.kron(matrices.X, matrices.I),
                            np.kron(matrices.Z, matrices.I))
    target_matrix += 0.5 * np.kron(np.kron(matrices.Y, matrices.Z),
                                   np.kron(matrices.I, matrices.I))
    target_matrix += np.kron(np.kron(matrices.Z, matrices.I),
                             np.kron(matrices.X, matrices.I))
    backend.assert_allclose(final_matrix, target_matrix)
