"""Test Trotter Hamiltonian methods from `qibo/core/hamiltonians.py`."""
import pytest
import numpy as np
from qibo import hamiltonians
from qibo.tests.utils import random_state, random_complex, random_hermitian


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
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, backend=backend) +
                  hamiltonians.TFIM(nqubits, h=0.5, backend=backend))
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham = local_ham1 - local_ham2
    target_ham = (hamiltonians.TFIM(nqubits, h=1.0, backend=backend) -
                  hamiltonians.TFIM(nqubits, h=0.5, backend=backend))
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
def test_trotter_hamiltonian_matmul(backend, nqubits, normalize):
    """Test Trotter Hamiltonian expectation value."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, dense=False, backend=backend)
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)

    state = backend.cast(random_complex((2 ** nqubits,)))
    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(trotter_ev, target_ev)

    state = random_complex((2 ** nqubits,))
    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(trotter_ev, target_ev)

    trotter_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    backend.assert_allclose(trotter_matmul, target_matmul)


def test_trotter_hamiltonian_three_qubit_term(backend):
    """Test creating ``TrotterHamiltonian`` with three qubit term."""
    from scipy.linalg import expm
    from qibo.hamiltonians.terms import HamiltonianTerm
    m1 = random_hermitian(3)
    m2 = random_hermitian(2)
    m3 = random_hermitian(1)

    terms = [HamiltonianTerm(m1, 0, 1, 2), HamiltonianTerm(m2, 2, 3),
             HamiltonianTerm(m3, 1)]
    ham = hamiltonians.SymbolicHamiltonian(backend=backend)
    ham.terms = terms

    # Test that the `TrotterHamiltonian` dense matrix is correct
    eye = np.eye(2, dtype=m1.dtype)
    mm1 = np.kron(m1, eye)
    mm2 = np.kron(np.kron(eye, eye), m2)
    mm3 = np.kron(np.kron(eye, m3), np.kron(eye, eye))
    target_ham = hamiltonians.Hamiltonian(4, mm1 + mm2 + mm3, backend=backend)
    backend.assert_allclose(ham.matrix, target_ham.matrix)

    dt = 1e-2
    initial_state = random_state(4)
    circuit = ham.circuit(dt=dt)
    final_state = backend.execute_circuit(circuit, np.copy(initial_state))
    u = [expm(-0.5j * dt * (mm1 + mm3)), expm(-0.5j * dt * mm2)]
    target_state = u[1].dot(u[0].dot(initial_state))
    target_state = u[0].dot(u[1].dot(target_state))
    backend.assert_allclose(final_state, target_state)


def test_old_trotter_hamiltonian_errors():
    """Check errors when creating the deprecated ``TrotterHamiltonian`` object."""
    with pytest.raises(NotImplementedError):
        h = hamiltonians.TrotterHamiltonian()
    with pytest.raises(NotImplementedError):
        h = hamiltonians.TrotterHamiltonian.from_symbolic(0, 1)
