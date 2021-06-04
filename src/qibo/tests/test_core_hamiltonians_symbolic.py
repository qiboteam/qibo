"""Test methods of :class:`qibo.core.hamiltonians.SymbolicHamiltonian`."""
import pytest
import numpy as np
import sympy
import qibo
from qibo import hamiltonians, K
from qibo.tests.utils import random_state, random_complex, random_hermitian


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


@pytest.mark.parametrize("nqubits", [3, 4])
#@pytest.mark.parametrize("model", ["TFIM", "XXZ", "Y", "MaxCut"])
def test_symbolic_hamiltonian_to_dense(nqubits):
    # TODO: Extend this to other models when `hamiltonians.py` is updated
    final_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1))
    target_ham = hamiltonians.TFIM(nqubits, h=1, numpy=True)
    np.testing.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_scalar_mul(calcdense, nqubits=3):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    target_ham = 2 * hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
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
def test_symbolic_hamiltonian_scalar_add(calcdense, nqubits=4):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    target_ham = 2 + hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
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
def test_symbolic_hamiltonian_scalar_sub(calcdense, nqubits=3):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    target_ham = 2 - hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 - local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = hamiltonians.TFIM(nqubits, h=1.0, numpy=True) - 2
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham - 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_symbolic_hamiltonian_operator_add_and_sub(nqubits=3):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham = (hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0)) +
                 hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5)))
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, numpy=True) +
                  hamiltonians.TFIM(nqubits, h=0.5, numpy=True))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham = (hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0)) -
                 hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=0.5)))
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, numpy=True) -
                  hamiltonians.TFIM(nqubits, h=0.5, numpy=True))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("calcdense", [False, True])
@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
def test_symbolic_hamiltonian_matmul(calcdense, nqubits, normalize):
    """Test Trotter Hamiltonian expectation value."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
    if calcdense:
        _ = local_ham.dense
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0)

    state = K.cast(random_complex((2 ** nqubits,)))
    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(local_ev, target_ev)

    state = random_complex((2 ** nqubits,))
    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(local_ev, target_ev)

    from qibo.core.states import VectorState
    state = VectorState.from_tensor(state)
    local_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    np.testing.assert_allclose(local_matmul, target_matmul)


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
