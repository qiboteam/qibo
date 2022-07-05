"""Test methods of :class:`qibo.core.hamiltonians.SymbolicHamiltonian`."""
import pytest
import numpy as np
import sympy
from qibo import hamiltonians
from qibo.tests.utils import random_complex


def symbolic_tfim(nqubits, h=1.0):
    """Constructs symbolic Hamiltonian for TFIM."""
    from qibo.symbols import Z, X
    sham = -sum(Z(i) * Z(i + 1) for i in range(nqubits - 1))
    sham -= Z(0) * Z(nqubits - 1)
    sham -= h * sum(X(i) for i in range(nqubits))
    return sham


def test_symbolic_hamiltonian_errors(backend):
    # Wrong type of Symbol matrix
    from qibo.symbols import Symbol
    with pytest.raises(TypeError):
        s = Symbol(0, "test")
    # Wrong type of symbolic expression
    with pytest.raises(TypeError):
        ham = hamiltonians.SymbolicHamiltonian("test", backend=backend)
    # Passing form with symbol that is not in ``symbol_map``
    from qibo import matrices
    Z, X = sympy.Symbol("Z"), sympy.Symbol("X")
    symbol_map = {Z: (0, matrices.Z)}
    with pytest.raises(ValueError):
        ham = hamiltonians.SymbolicHamiltonian(Z * X, symbol_map, backend=backend)
    # Invalid operation in Hamiltonian expresion
    ham = hamiltonians.SymbolicHamiltonian(sympy.cos(Z), symbol_map, backend=backend)
    with pytest.raises(TypeError):
        dense = ham.dense


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("calcterms", [False, True])
def test_symbolictfim_hamiltonian_to_dense(backend, nqubits, calcterms):
    final_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1), backend=backend)
    target_ham = hamiltonians.TFIM(nqubits, h=1, backend=backend)
    if calcterms:
        _ = final_ham.terms
    backend.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("calcterms", [False, True])
def test_symbolicxxz_hamiltonian_to_dense(backend, nqubits, calcterms):
    from qibo.symbols import X, Y, Z
    sham = sum(X(i) * X(i + 1) for i in range(nqubits - 1))
    sham += sum(Y(i) * Y(i + 1) for i in range(nqubits - 1))
    sham += 0.5 * sum(Z(i) * Z(i + 1) for i in range(nqubits - 1))
    sham += X(0) * X(nqubits - 1) + Y(0) * Y(nqubits - 1) + 0.5 * Z(0) * Z(nqubits - 1)
    final_ham = hamiltonians.SymbolicHamiltonian(sham, backend=backend)
    target_ham = hamiltonians.XXZ(nqubits, backend=backend)
    if calcterms:
        _ = final_ham.terms
    backend.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_scalar_mul(backend, nqubits, calcterms, calcdense):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    target_ham = 2 * hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 * local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham * 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [4])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_scalar_add(backend, nqubits, calcterms, calcdense):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    target_ham = 2 + hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 + local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham + 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_scalar_sub(backend, nqubits, calcterms, calcdense):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    target_ham = 2 - hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 - local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) - 2
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham - 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_operator_add_and_sub(backend, nqubits, calcterms, calcdense):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    local_ham2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5), backend=backend)
    if calcterms:
        _ = local_ham1.terms
        _ = local_ham2.terms
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_ham = local_ham1 + local_ham2
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, backend=backend) +
                  hamiltonians.TFIM(nqubits, h=0.5, backend=backend))
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    local_ham2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5), backend=backend)
    if calcterms:
        _ = local_ham1.terms
        _ = local_ham2.terms
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_ham = local_ham1 - local_ham2
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, backend=backend) -
                  hamiltonians.TFIM(nqubits, h=0.5, backend=backend))
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [5])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_hamiltonianmatmul(backend, nqubits, calcterms, calcdense):
    local_ham1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    local_ham2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5), backend=backend)
    dense_ham1 = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    dense_ham2 = hamiltonians.TFIM(nqubits, h=0.5, backend=backend)
    if calcterms:
        _ = local_ham1.terms
        _ = local_ham2.terms
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_matmul = local_ham1 @ local_ham2
    target_matmul = dense_ham1 @ dense_ham2
    backend.assert_allclose(local_matmul.matrix, target_matmul.matrix)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("calcterms", [False, True])
def test_symbolic_hamiltonian_matmul(backend, nqubits, density_matrix, calcterms):
    if density_matrix:
        #from qibo.core.states import MatrixState
        shape = (2 ** nqubits, 2 ** nqubits)
        #state = MatrixState.from_tensor(random_complex(shape))
    else:
        #from qibo.core.states import VectorState
        shape = (2 ** nqubits,)
        #state = VectorState.from_tensor(random_complex(shape))
    state = random_complex(shape)
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    local_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    backend.assert_allclose(local_matmul, target_matmul)


@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_state_ev(backend, nqubits, normalize, calcterms, calcdense):
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend) + 2
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) + 2

    state = backend.cast(random_complex((2 ** nqubits,)))
    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(local_ev, target_ev)

    state = random_complex((2 ** nqubits,))
    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(local_ev, target_ev)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("calcterms", [False, True])
def test_symbolic_hamiltonian_abstract_symbol_ev(backend, density_matrix, calcterms):
    from qibo.symbols import X, Symbol
    matrix = np.random.random((2, 2))
    form = X(0) * Symbol(1, matrix) + Symbol(0, matrix) * X(1)
    local_ham = hamiltonians.SymbolicHamiltonian(form, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if density_matrix:
        state = backend.cast(random_complex((4, 4)))
    else:
        state = backend.cast(random_complex((4,)))
    local_ev = local_ham.expectation(state)
    target_ev = local_ham.dense.expectation(state)
    backend.assert_allclose(local_ev, target_ev)


def test_trotter_hamiltonian_operation_errors(backend):
    """Test errors in ``SymbolicHamiltonian`` addition and subtraction."""
    h1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(3, h=1.0), backend=backend)
    h2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(4, h=1.0), backend=backend)
    with pytest.raises(RuntimeError):
        h = h1 + h2
    with pytest.raises(RuntimeError):
        h = h1 - h2
    with pytest.raises(NotImplementedError):
        h = h1 + "test"
    with pytest.raises(NotImplementedError):
        h = "test" + h1
    with pytest.raises(NotImplementedError):
        h = h1 - "test"
    with pytest.raises(NotImplementedError):
        h = "test" - h1
    with pytest.raises(NotImplementedError):
        h = h1 * "test"
    with pytest.raises(NotImplementedError):
        h = h1 @ "test"
    with pytest.raises(NotImplementedError):
        h = h1 @ np.ones((2, 2, 2, 2))
    h2 = hamiltonians.XXZ(3, dense=False, backend=backend)
    with pytest.raises(NotImplementedError):
        h = h1 @ h2
