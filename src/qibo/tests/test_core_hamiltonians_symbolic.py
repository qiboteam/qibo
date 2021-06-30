"""Test methods of :class:`qibo.core.hamiltonians.SymbolicHamiltonian`."""
import pytest
import numpy as np
import sympy
import qibo
from qibo import hamiltonians, K
from qibo.tests.utils import random_complex


def symbolic_tfim(nqubits, h=1.0):
    """Constructs symbolic Hamiltonian for TFIM."""
    from qibo.symbols import Z, X
    sham = -sum(Z(i) * Z(i + 1) for i in range(nqubits - 1))
    sham -= Z(0) * Z(nqubits - 1)
    sham -= h * sum(X(i) for i in range(nqubits))
    return sham


def test_symbolic_hamiltonian_init():
    # Wrong type of symbolic expression
    with pytest.raises(TypeError):
        ham = hamiltonians.SymbolicHamiltonian("test")
    # Wrong type of Symbol matrix
    from qibo.symbols import Symbol
    with pytest.raises(TypeError):
        s = Symbol(0, "test")
    # Wrong HamiltonianTerm matrix
    from qibo.core.terms import HamiltonianTerm
    with pytest.raises(TypeError):
        t = HamiltonianTerm("test", 0, 1)


@pytest.mark.parametrize("nqubits", [3, 4])
#@pytest.mark.parametrize("model", ["TFIM", "XXZ", "Y", "MaxCut"])
def test_symbolic_hamiltonian_to_dense(backend, nqubits):
    # TODO: Extend this to other models when `hamiltonians.py` is updated
    final_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1))
    target_ham = hamiltonians.TFIM(nqubits, h=1)
    np.testing.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_scalar_mul(backend, calcdense, nqubits=3):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    target_ham = 2 * hamiltonians.TFIM(nqubits, h=1.0)
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 * local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham * 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_scalar_add(backend, calcdense, nqubits=4):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    target_ham = 2 + hamiltonians.TFIM(nqubits, h=1.0)
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 + local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham + 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_scalar_sub(backend, calcdense, nqubits=3):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    target_ham = 2 - hamiltonians.TFIM(nqubits, h=1.0)
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 - local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = hamiltonians.TFIM(nqubits, h=1.0) - 2
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham - 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_operator_add_and_sub(backend, calcdense, nqubits=3):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    local_ham2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5))
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_ham = local_ham1 + local_ham2
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0) +
                  hamiltonians.TFIM(nqubits, h=0.5))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    local_ham2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5))
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_ham = local_ham1 - local_ham2
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0) -
                  hamiltonians.TFIM(nqubits, h=0.5))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_hamiltonianmatmul(backend, calcdense, nqubits=5):
    local_ham1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    local_ham2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5))
    dense_ham1 = hamiltonians.TFIM(nqubits, h=1.0)
    dense_ham2 = hamiltonians.TFIM(nqubits, h=0.5)
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_matmul = local_ham1 @ local_ham2
    target_matmul = dense_ham1 @ dense_ham2
    np.testing.assert_allclose(local_matmul.matrix, target_matmul.matrix)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nqubits", [3, 4])
def test_symbolic_hamiltonian_matmul(backend, density_matrix, nqubits):
    if density_matrix:
        from qibo.core.states import MatrixState
        shape = (2 ** nqubits, 2 ** nqubits)
        state = MatrixState.from_tensor(random_complex(shape))
    else:
        from qibo.core.states import VectorState
        shape = (2 ** nqubits,)
        state = VectorState.from_tensor(random_complex(shape))
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0)
    local_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    np.testing.assert_allclose(local_matmul, target_matmul)


@pytest.mark.parametrize("calcdense", [False, True])
@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
def test_symbolic_hamiltonian_state_ev(backend, calcdense, nqubits, normalize):
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0)) + 2
    if calcdense:
        _ = local_ham.dense
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0) + 2

    state = K.cast(random_complex((2 ** nqubits,)))
    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(local_ev, target_ev)

    state = random_complex((2 ** nqubits,))
    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(local_ev, target_ev)


@pytest.mark.parametrize("density_matrix", [False, True])
def test_symbolic_hamiltonian_abstract_symbol_ev(backend, density_matrix):
    from qibo.symbols import X, Symbol
    matrix = np.random.random((2, 2))
    form = X(0) * Symbol(1, matrix) + Symbol(0, matrix) * X(1)
    local_ham = hamiltonians.SymbolicHamiltonian(form)
    if density_matrix:
        state = K.cast(random_complex((4, 4)))
    else:
        state = K.cast(random_complex((4,)))
    local_ev = local_ham.expectation(state)
    target_ev = local_ham.dense.expectation(state)
    np.testing.assert_allclose(local_ev, target_ev)


def test_trotter_hamiltonian_operation_errors():
    """Test errors in ``SymbolicHamiltonian`` addition and subtraction."""
    h1 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(3, h=1.0))
    h2 = hamiltonians.SymbolicHamiltonian(symbolic_tfim(4, h=1.0))
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
    h2 = hamiltonians.XXZ(3, dense=False)
    with pytest.raises(NotImplementedError):
        h = h1 @ h2
