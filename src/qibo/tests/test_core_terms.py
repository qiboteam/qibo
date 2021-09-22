"""Tests methods defined in `qibo/core/terms.py`."""
import pytest
import numpy as np
from qibo import K, models, gates
from qibo.core import terms
from qibo.tests.utils import random_state


def test_hamiltonian_term_initialization(backend):
    """Test initialization and matrix assignment of ``HamiltonianTerm``."""
    matrix = np.random.random((2, 2))
    term = terms.HamiltonianTerm(matrix, 0)
    assert term.target_qubits == (0,)
    K.assert_allclose(term.matrix, matrix)
    matrix = np.random.random((4, 4))
    term = terms.HamiltonianTerm(matrix, 2, 3)
    assert term.target_qubits == (2, 3)
    K.assert_allclose(term.matrix, matrix)


def test_hamiltonian_term_gates(backend):
    """Test gate application of ``HamiltonianTerm``."""
    matrix = np.random.random((4, 4))
    term = terms.HamiltonianTerm(matrix, 1, 2)
    gate = term.gate
    assert gate.target_qubits == (1, 2)
    K.assert_allclose(gate.matrix, matrix)

    initial_state = random_state(4)
    final_state = term(np.copy(initial_state))
    c = models.Circuit(4)
    c.add(gates.Unitary(matrix, 1, 2))
    target_state = c(np.copy(initial_state))
    K.assert_allclose(final_state, target_state)


def test_hamiltonian_term_exponentiation(backend):
    """Test exp gate application of ``HamiltonianTerm``."""
    from scipy.linalg import expm
    matrix = np.random.random((2, 2))
    term = terms.HamiltonianTerm(matrix, 1)
    exp_matrix = expm(-0.5j * matrix)
    K.assert_allclose(term.exp(0.5), exp_matrix)

    initial_state = random_state(2)
    final_state = term.expgate(0.5)(np.copy(initial_state))
    exp_gate = gates.Unitary(exp_matrix, 1)
    target_state = exp_gate(np.copy(initial_state))
    K.assert_allclose(final_state, target_state)


def test_hamiltonian_term_mul(backend):
    """Test scalar multiplication of ``HamiltonianTerm``."""
    matrix = np.random.random((4, 4))
    term = terms.HamiltonianTerm(matrix, 0, 2)
    mterm = 2 * term
    assert mterm.target_qubits == term.target_qubits
    K.assert_allclose(mterm.matrix, 2 * matrix)
    mterm = term * 5
    assert mterm.target_qubits == term.target_qubits
    K.assert_allclose(mterm.matrix, 5 * matrix)


def test_hamiltonian_term_merge(backend):
    """Test ``HamiltonianTerm.merge``."""
    matrix1 = np.random.random((2, 2))
    matrix2 = np.random.random((4, 4))
    term1 = terms.HamiltonianTerm(matrix1, 1)
    term2 = terms.HamiltonianTerm(matrix2, 0, 1)
    mterm = term2.merge(term1)
    target_matrix = np.kron(np.eye(2), matrix1) + matrix2
    assert mterm.target_qubits == (0, 1)
    K.assert_allclose(mterm.matrix, target_matrix)
    with pytest.raises(ValueError):
        term1.merge(term2)
