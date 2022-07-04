"""Tests methods defined in `qibo/core/terms.py`."""
import pytest
import numpy as np
from qibo import matrices, models, gates
from qibo.hamiltonians import terms
from qibo.tests.utils import random_state, random_density_matrix


def test_hamiltonian_term_initialization(backend):
    """Test initialization and matrix assignment of ``HamiltonianTerm``."""
    matrix = np.random.random((2, 2))
    term = terms.HamiltonianTerm(matrix, 0)
    assert term.target_qubits == (0,)
    backend.assert_allclose(term.matrix, matrix)
    matrix = np.random.random((4, 4))
    term = terms.HamiltonianTerm(matrix, 2, 3)
    assert term.target_qubits == (2, 3)
    backend.assert_allclose(term.matrix, matrix)


def test_hamiltonian_term_initialization_errors():
    """Test initializing ``HamiltonianTerm`` with wrong parameters."""
    # Wrong HamiltonianTerm matrix
    with pytest.raises(TypeError):
        t = terms.HamiltonianTerm("test", 0, 1)
    # Passing negative target qubits in HamiltonianTerm
    with pytest.raises(ValueError):
        t = terms.HamiltonianTerm("test", 0, -1)
    # Passing matrix shape incompatible with number of qubits
    with pytest.raises(ValueError):
        t = terms.HamiltonianTerm(np.random.random((4, 4)), 0, 1, 2)
    # Merging terms with invalid qubits
    t1 = terms.HamiltonianTerm(np.random.random((4, 4)), 0, 1)
    t2 = terms.HamiltonianTerm(np.random.random((4, 4)), 1, 2)
    with pytest.raises(ValueError):
        t = t1.merge(t2)


def test_hamiltonian_term_gates(backend):
    """Test gate application of ``HamiltonianTerm``."""
    nqubits = 4
    matrix = np.random.random((nqubits, nqubits))
    term = terms.HamiltonianTerm(matrix, 1, 2)
    gate = term.gate
    assert gate.target_qubits == (1, 2)
    backend.assert_allclose(gate.asmatrix(backend), matrix)

    initial_state = random_state(nqubits)
    final_state = term(backend, np.copy(initial_state), nqubits)
    c = models.Circuit(nqubits)
    c.add(gates.Unitary(matrix, 1, 2))
    target_state = backend.execute_circuit(c, np.copy(initial_state))
    backend.assert_allclose(final_state, target_state)


def test_hamiltonian_term_exponentiation(backend):
    """Test exp gate application of ``HamiltonianTerm``."""
    from scipy.linalg import expm
    matrix = np.random.random((2, 2))
    term = terms.HamiltonianTerm(matrix, 1)
    exp_matrix = expm(-0.5j * matrix)
    backend.assert_allclose(term.exp(0.5), exp_matrix)

    initial_state = random_state(2)
    final_state = term(backend, np.copy(initial_state), 2, term.expgate(0.5))
    exp_gate = gates.Unitary(exp_matrix, 1)
    target_state = backend.apply_gate(exp_gate, np.copy(initial_state), 2)
    backend.assert_allclose(final_state, target_state)


def test_hamiltonian_term_mul(backend):
    """Test scalar multiplication of ``HamiltonianTerm``."""
    matrix = np.random.random((4, 4))
    term = terms.HamiltonianTerm(matrix, 0, 2)
    mterm = 2 * term
    assert mterm.target_qubits == term.target_qubits
    backend.assert_allclose(mterm.matrix, 2 * matrix)
    mterm = term * 5
    assert mterm.target_qubits == term.target_qubits
    backend.assert_allclose(mterm.matrix, 5 * matrix)


def test_hamiltonian_term_merge(backend):
    """Test ``HamiltonianTerm.merge``."""
    matrix1 = np.random.random((2, 2))
    matrix2 = np.random.random((4, 4))
    term1 = terms.HamiltonianTerm(matrix1, 1)
    term2 = terms.HamiltonianTerm(matrix2, 0, 1)
    mterm = term2.merge(term1)
    target_matrix = np.kron(np.eye(2), matrix1) + matrix2
    assert mterm.target_qubits == (0, 1)
    backend.assert_allclose(mterm.matrix, target_matrix)
    with pytest.raises(ValueError):
        term1.merge(term2)


@pytest.mark.parametrize("use_symbols", [True, False])
def test_symbolic_term_creation(backend, use_symbols):
    """Test creating ``SymbolicTerm`` from sympy expression."""
    if use_symbols:
        from qibo.symbols import X, Y
        expression = X(0) * Y(1) * X(1)
        symbol_map = {}
    else:
        import sympy
        x0, x1, y1 = sympy.symbols("X0 X1 Y1", commutative=False)
        expression = x0 * y1 * x1
        symbol_map = {x0: (0, matrices.X), x1: (1, matrices.X),
                      y1: (1, matrices.Y)}
    term = terms.SymbolicTerm(2, expression, symbol_map)
    assert term.target_qubits == (0, 1)
    assert len(term.matrix_map) == 2
    backend.assert_allclose(term.matrix_map.get(0)[0], matrices.X)
    backend.assert_allclose(term.matrix_map.get(1)[0], matrices.Y)
    backend.assert_allclose(term.matrix_map.get(1)[1], matrices.X)


def test_symbolic_term_with_power_creation(backend):
    """Test creating ``SymbolicTerm`` from sympy expression that contains powers."""
    from qibo.symbols import X, Z
    expression = X(0) ** 4 * Z(1) ** 2 * X(2)
    term = terms.SymbolicTerm(2, expression)
    assert term.target_qubits == (0, 1, 2)
    assert len(term.matrix_map) == 3
    assert term.coefficient == 2
    backend.assert_allclose(term.matrix_map.get(0), 4 * [matrices.X])
    backend.assert_allclose(term.matrix_map.get(1), 2 * [matrices.Z])
    backend.assert_allclose(term.matrix_map.get(2), [matrices.X])


def test_symbolic_term_with_imag_creation(backend):
    """Test creating ``SymbolicTerm`` from sympy expression that contains imaginary coefficients."""
    from qibo.symbols import Y
    expression = 3j * Y(0)
    term = terms.SymbolicTerm(2, expression)
    assert term.target_qubits == (0,)
    assert term.coefficient == 6j


def test_symbolic_term_matrix(backend):
    """Test matrix calculation of ``SymbolicTerm``."""
    from qibo.symbols import X, Y, Z
    expression = X(0) * Y(1) * Z(2) * X(1)
    term = terms.SymbolicTerm(2, expression)
    assert term.target_qubits == (0, 1, 2)
    target_matrix = np.kron(matrices.X, matrices.Y @ matrices.X)
    target_matrix = 2 * np.kron(target_matrix, matrices.Z)
    backend.assert_allclose(term.matrix, target_matrix)


def test_symbolic_term_mul(backend):
    """Test multiplying scalar to ``SymbolicTerm``."""
    from qibo.symbols import X, Y, Z
    expression = Y(2) * Z(3) * X(2) * X(3)
    term = terms.SymbolicTerm(1, expression)
    assert term.target_qubits == (2, 3)
    target_matrix = np.kron(matrices.Y @ matrices.X, matrices.Z @ matrices.X)
    backend.assert_allclose(term.matrix, target_matrix)
    backend.assert_allclose((2 * term).matrix, 2 * target_matrix)
    backend.assert_allclose((term * 3).matrix, 3 * target_matrix)


@pytest.mark.parametrize("density_matrix", [False])
def test_symbolic_term_call(backend, density_matrix):
    """Test applying ``SymbolicTerm`` to state."""
    from qibo.symbols import X, Y, Z
    expression = Z(0) * X(1) * Y(2)
    term = terms.SymbolicTerm(2, expression)
    matrixlist = [np.kron(matrices.Z, np.eye(4)),
                  np.kron(np.kron(np.eye(2), matrices.X), np.eye(2)),
                  np.kron(np.eye(4), matrices.Y)]
    initial_state = random_density_matrix(3) if density_matrix else random_state(3)
    final_state = term(backend, np.copy(initial_state), 3, density_matrix=density_matrix)
    target_state = 2 * np.copy(initial_state)
    for matrix in matrixlist:
        target_state = matrix @ target_state
    backend.assert_allclose(final_state, target_state)


def test_symbolic_term_merge(backend):
    """Test merging ``SymbolicTerm`` to ``HamiltonianTerm``."""
    from qibo.symbols import X, Z
    matrix = np.random.random((4, 4))
    term1 = terms.HamiltonianTerm(matrix, 0, 1)
    term2 = terms.SymbolicTerm(1, X(0) * Z(1))
    term = term1.merge(term2)
    target_matrix = matrix + np.kron(matrices.X, matrices.Z)
    backend.assert_allclose(term.matrix, target_matrix)


def test_term_group_append():
    """Test ``GroupTerm.can_append`` method."""
    term1 = terms.HamiltonianTerm(np.random.random((8, 8)), 0, 1, 3)
    term2 = terms.HamiltonianTerm(np.random.random((2, 2)), 0)
    term3 = terms.HamiltonianTerm(np.random.random((2, 2)), 1)
    term4 = terms.HamiltonianTerm(np.random.random((2, 2)), 2)
    group = terms.TermGroup(term1)
    assert group.can_append(term2)
    assert group.can_append(term3)
    assert not group.can_append(term4)
    group.append(term2)
    group.append(term3)
    assert group.target_qubits == {0, 1, 3}


def test_term_group_to_term(backend):
    """Test ``GroupTerm.term`` property."""
    from qibo.symbols import X, Z
    matrix = np.random.random((8, 8))
    term1 = terms.HamiltonianTerm(matrix, 0, 1, 3)
    term2 = terms.SymbolicTerm(1, X(0) * Z(3))
    term3 = terms.SymbolicTerm(2, X(1))
    group = terms.TermGroup(term1)
    group.append(term2)
    group.append(term3)
    matrix2 = np.kron(np.kron(matrices.X, np.eye(2)), matrices.Z)
    matrix3 = np.kron(np.kron(np.eye(2), matrices.X), np.eye(2))
    target_matrix = matrix + matrix2 + 2 * matrix3
    backend.assert_allclose(group.term.matrix, target_matrix)
