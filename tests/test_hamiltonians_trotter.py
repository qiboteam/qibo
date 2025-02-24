"""Test Trotter Hamiltonian methods from `qibo/core/hamiltonians.py`."""

import numpy as np
import pytest

from qibo import hamiltonians, symbols
from qibo.backends import NumpyBackend
from qibo.quantum_info import random_hermitian, random_statevector


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("model", ["TFIM", "XXZ", "Y", "MaxCut"])
def test_trotter_hamiltonian_to_dense(backend, nqubits, model):
    """Test that Trotter Hamiltonian dense form agrees with normal Hamiltonian."""
    local_ham = getattr(hamiltonians, model)(nqubits, dense=False, backend=backend)
    target_ham = getattr(hamiltonians, model)(nqubits, backend=backend)
    final_ham = local_ham.dense
    backend.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


def test_trotter_hamiltonian_scalar_mul(backend, nqubits=3):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    target_ham = 2 * hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    local_dense = (2 * local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    local_dense = (local_ham * 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_scalar_add(backend, nqubits=4):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    target_ham = 2 + hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    local_dense = (2 + local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    local_dense = (local_ham + 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_scalar_sub(backend, nqubits=3):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    target_ham = 2 - hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    local_dense = (2 - local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) - 2
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    local_dense = (local_ham - 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_operator_add_and_sub(backend, nqubits=3):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham1 = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    local_ham2 = hamiltonians.TFIM(nqubits, h=0.5, dense=False, backend=backend)

    local_ham = local_ham1 + local_ham2
    target_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) + hamiltonians.TFIM(
        nqubits, h=0.5, backend=backend
    )
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham = local_ham1 - local_ham2
    target_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) - hamiltonians.TFIM(
        nqubits, h=0.5, backend=backend
    )
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
def test_trotter_hamiltonian_matmul(backend, nqubits, normalize):
    """Test Trotter Hamiltonian expectation value."""
    state = random_statevector(2**nqubits, backend=backend)

    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)

    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(trotter_ev, target_ev)

    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(trotter_ev, target_ev)

    trotter_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    backend.assert_allclose(trotter_matmul, target_matmul)


def test_symbolic_hamiltonian_circuit_different_dts(backend):
    """Issue: https://github.com/qiboteam/qibo/issues/1357."""
    ham = hamiltonians.SymbolicHamiltonian(symbols.Z(0), backend=backend)
    a = ham.circuit(0.1)
    b = ham.circuit(0.1)
    matrix1 = ham.circuit(0.2).unitary(backend)
    matrix2 = (a + b).unitary(backend)
    backend.assert_allclose(matrix1, matrix2)
