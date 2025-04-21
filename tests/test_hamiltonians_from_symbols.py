"""Test dense matrix of Hamiltonians constructed using symbols."""

import pickle

import numpy as np
import pytest
import sympy

from qibo import get_backend, hamiltonians, matrices
from qibo.backends import NumpyBackend
from qibo.quantum_info import random_hermitian
from qibo.symbols import I, Symbol, X, Y, Z


@pytest.mark.parametrize("symbol", [I, X, Y, Z])
def test_symbols_pickling(symbol):
    symbol = symbol(int(np.random.randint(4)))
    dumped_symbol = pickle.dumps(symbol)
    new_symbol = pickle.loads(dumped_symbol)
    for attr in ("target_qubit", "name", "_gate"):
        assert getattr(symbol, attr) == getattr(new_symbol, attr)
    get_backend().assert_allclose(symbol.matrix, new_symbol.matrix)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_tfim_hamiltonian_from_symbols(backend, nqubits):
    """Check creating TFIM Hamiltonian using sympy."""
    h = 0.5
    symham = sum(
        Z(i, backend=backend) * Z(i + 1, backend=backend) for i in range(nqubits - 1)
    )
    symham += Z(0, backend=backend) * Z(nqubits - 1, backend=backend)
    symham += h * sum(X(i, backend=backend) for i in range(nqubits))
    ham = hamiltonians.SymbolicHamiltonian(-symham, backend=backend)
    final_matrix = ham.matrix
    target_matrix = hamiltonians.TFIM(nqubits, h=h, backend=backend).matrix
    backend.assert_allclose(final_matrix, target_matrix)


def test_from_symbolic_with_power(backend):
    """Check ``from_symbolic`` when the expression contains powers."""
    npbackend = NumpyBackend()
    matrix = random_hermitian(2, backend=npbackend)
    symham = (
        Symbol(0, matrix, backend=backend) ** 2
        - Symbol(1, matrix, backend=backend) ** 2
        + 3 * Symbol(1, matrix, backend=backend)
        - 2 * Symbol(0, matrix, backend=backend) * Symbol(2, matrix, backend=backend)
        + 1
    )
    ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)

    final_matrix = ham.matrix
    matrix2 = matrix.dot(matrix)
    eye = np.eye(2, dtype=matrix.dtype)
    target_matrix = np.kron(np.kron(matrix2, eye), eye)
    target_matrix -= np.kron(np.kron(eye, matrix2), eye)
    target_matrix += 3 * np.kron(np.kron(eye, matrix), eye)
    target_matrix -= 2 * np.kron(np.kron(matrix, eye), matrix)
    target_matrix += np.eye(8, dtype=matrix.dtype)
    backend.assert_allclose(final_matrix, target_matrix)


def test_from_symbolic_with_complex_numbers(backend):
    """Check ``from_symbolic`` when the expression contains imaginary unit."""
    symham = (
        (1 + 2j) * X(0, backend=backend) * X(1, backend=backend)
        + 2 * Y(0, backend=backend) * Y(1, backend=backend)
        - 3j * X(0, backend=backend) * Y(1, backend=backend)
        + 1j * Y(0, backend=backend) * X(1, backend=backend)
    )
    ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)

    final_matrix = ham.matrix
    target_matrix = (1 + 2j) * backend.np.kron(backend.matrices.X, backend.matrices.X)
    target_matrix += 2 * backend.np.kron(backend.matrices.Y, backend.matrices.Y)
    target_matrix -= 3j * backend.np.kron(backend.matrices.X, backend.matrices.Y)
    target_matrix += 1j * backend.np.kron(backend.matrices.Y, backend.matrices.X)
    backend.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_x_hamiltonian_from_symbols(backend, nqubits):
    """Check creating sum(X) Hamiltonian using sympy."""
    symham = -sum(X(i, backend=backend) for i in range(nqubits))
    ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)
    final_matrix = ham.matrix
    target_matrix = hamiltonians.X(nqubits, backend=backend).matrix
    backend.assert_allclose(final_matrix, target_matrix)


def test_three_qubit_term_hamiltonian_from_symbols(backend):
    """Check creating Hamiltonian with three-qubit interaction using sympy."""
    symham = (
        X(0, backend=backend) * Y(1, backend=backend) * Z(2, backend=backend)
        + 0.5 * Y(0, backend=backend) * Z(1, backend=backend) * X(3, backend=backend)
        + Z(0, backend=backend) * X(2, backend=backend)
    )
    symham += (
        Y(2, backend=backend)
        + 1.5 * Z(1, backend=backend)
        - 2
        - 3 * X(1, backend=backend) * Y(3, backend=backend)
    )
    ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)
    final_matrix = ham.matrix
    target_matrix = backend.np.kron(
        backend.np.kron(backend.matrices.X, backend.matrices.Y),
        backend.np.kron(backend.matrices.Z, backend.matrices.I()),
    )
    target_matrix += 0.5 * backend.np.kron(
        backend.np.kron(backend.matrices.Y, backend.matrices.Z),
        backend.np.kron(backend.matrices.I(), backend.matrices.X),
    )
    target_matrix += backend.np.kron(
        backend.np.kron(backend.matrices.Z, backend.matrices.I()),
        backend.np.kron(backend.matrices.X, backend.matrices.I()),
    )
    target_matrix += -3 * backend.np.kron(
        backend.np.kron(backend.matrices.I(), backend.matrices.X),
        backend.np.kron(backend.matrices.I(), backend.matrices.Y),
    )
    target_matrix += backend.np.kron(
        backend.np.kron(backend.matrices.I(), backend.matrices.I()),
        backend.np.kron(backend.matrices.Y, backend.matrices.I()),
    )
    target_matrix += 1.5 * backend.np.kron(
        backend.np.kron(backend.matrices.I(), backend.matrices.Z),
        backend.np.kron(backend.matrices.I(), backend.matrices.I()),
    )
    target_matrix -= 2 * backend.matrices.I(2**4)
    backend.assert_allclose(final_matrix, target_matrix)


def test_hamiltonian_with_identity_symbol(backend):
    """Check creating Hamiltonian from expression which contains the identity symbol."""
    symham = (
        X(0, backend=backend) * I(1, backend=backend) * Z(2, backend=backend)
        + 0.5 * Y(0, backend=backend) * Z(1, backend=backend) * I(3, backend=backend)
        + Z(0, backend=backend) * I(1, backend=backend) * X(2, backend=backend)
    )
    ham = hamiltonians.SymbolicHamiltonian(symham, backend=backend)

    final_matrix = ham.matrix
    target_matrix = backend.np.kron(
        backend.np.kron(backend.matrices.X, backend.matrices.I()),
        backend.np.kron(backend.matrices.Z, backend.matrices.I()),
    )
    target_matrix += 0.5 * np.kron(
        backend.np.kron(backend.matrices.Y, backend.matrices.Z),
        backend.np.kron(backend.matrices.I(), backend.matrices.I()),
    )
    target_matrix += backend.np.kron(
        backend.np.kron(backend.matrices.Z, backend.matrices.I()),
        backend.np.kron(backend.matrices.X, backend.matrices.I()),
    )
    backend.assert_allclose(final_matrix, target_matrix)
