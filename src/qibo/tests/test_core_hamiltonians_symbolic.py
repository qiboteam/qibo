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
    # Give both form and terms
    with pytest.raises(ValueError):
        ham = hamiltonians.SymbolicHamiltonian(sympy.Symbol("x"), "test")
    # TODO: Complete this when `SymbolicHamiltonian.__init__` is completed


@pytest.mark.parametrize("nqubits", [3, 4])
#@pytest.mark.parametrize("model", ["TFIM", "XXZ", "Y", "MaxCut"])
def test_symbolic_hamiltonian_to_dense(nqubits):
    # TODO: Extend this to other models when `hamiltonians.py` is updated
    final_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1))
    target_ham = hamiltonians.TFIM(nqubits, h=1, numpy=True)
    np.testing.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


@pytest.mark.skip
def test_trotter_hamiltonian_scalar_mul(nqubits=3):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 * hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 * local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham * 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.skip
def test_trotter_hamiltonian_scalar_add(nqubits=4):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 + hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 + local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham + 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.skip
def test_symbolic_hamiltonian_scalar_sub(nqubits=3):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 - hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 - local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = hamiltonians.TFIM(nqubits, h=1.0, numpy=True) - 2
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham - 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.skip
def test_symbolic_hamiltonian_operator_add_and_sub(nqubits=3):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham1 = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    local_ham2 = hamiltonians.TFIM(nqubits, h=0.5, trotter=True)

    local_ham = local_ham1 + local_ham2
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, numpy=True) +
                  hamiltonians.TFIM(nqubits, h=0.5, numpy=True))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham = local_ham1 - local_ham2
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, numpy=True) -
                  hamiltonians.TFIM(nqubits, h=0.5, numpy=True))
    dense = local_ham.dense
    np.testing.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
def test_symbolic_hamiltonian_matmul(nqubits, normalize):
    """Test Trotter Hamiltonian expectation value."""
    local_ham = hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0))
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
