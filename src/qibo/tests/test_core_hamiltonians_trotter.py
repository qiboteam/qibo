"""Test Trotter Hamiltonian methods from `qibo/core/hamiltonians.py`."""
import pytest
import numpy as np
import qibo
from qibo import hamiltonians, K
from qibo.tests.utils import random_state, random_complex, random_hermitian


def test_trotter_hamiltonian_init():
    # Wrong type of terms
    with pytest.raises(TypeError):
        ham = hamiltonians.TrotterHamiltonian({(0, 1): "abc"})
    # Wrong type of parts
    with pytest.raises(TypeError):
        ham = hamiltonians.TrotterHamiltonian([(0, 1)])
    # Wrong number of target qubits
    with pytest.raises(ValueError):
        ham = hamiltonians.TrotterHamiltonian({(0, 1): hamiltonians.TFIM(nqubits=3, numpy=True)})
    # Same targets multiple times
    h = hamiltonians.TFIM(nqubits=2, numpy=True)
    with pytest.raises(ValueError):
        ham = hamiltonians.TrotterHamiltonian({(0, 1): h}, {(0, 1): h})
    # Different term Hamiltonian types
    h2 = hamiltonians.TFIM(nqubits=2, numpy=False)
    with pytest.raises(TypeError):
        ham = hamiltonians.TrotterHamiltonian({(0, 1): h, (1, 2): h2})


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("model", ["TFIM", "XXZ", "Y", "MaxCut"])
def test_trotter_hamiltonian_to_dense(nqubits, model):
    """Test that Trotter Hamiltonian dense form agrees with normal Hamiltonian."""
    local_ham = getattr(hamiltonians, model)(nqubits, trotter=True)
    target_ham = getattr(hamiltonians, model)(nqubits, numpy=True)
    final_ham = local_ham.dense
    np.testing.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


def test_trotter_hamiltonian_scalar_mul(nqubits=3):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 * hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 * local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham * 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_scalar_add(nqubits=4):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 + hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 + local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham + 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_scalar_sub(nqubits=3):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    target_ham = 2 - hamiltonians.TFIM(nqubits, h=1.0, numpy=True)
    local_dense = (2 - local_ham).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = hamiltonians.TFIM(nqubits, h=1.0, numpy=True) - 2
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    local_dense = (local_ham - 2).dense
    np.testing.assert_allclose(local_dense.matrix, target_ham.matrix)


def test_trotter_hamiltonian_operator_add_and_sub(nqubits=3):
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
def test_trotter_hamiltonian_matmul(nqubits, normalize):
    """Test Trotter Hamiltonian expectation value."""
    local_ham = hamiltonians.TFIM(nqubits, h=1.0, trotter=True)
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0)

    state = K.cast(random_complex((2 ** nqubits,)))
    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(trotter_ev, target_ev)

    state = random_complex((2 ** nqubits,))
    trotter_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    np.testing.assert_allclose(trotter_ev, target_ev)

    from qibo.core.states import VectorState
    state = VectorState.from_tensor(state)
    trotter_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    np.testing.assert_allclose(trotter_matmul, target_matmul)


def test_trotter_hamiltonian_operation_errors():
    """Test errors in ``TrotterHamiltonian`` addition and subtraction."""
    # test addition with different number of parts
    h1 = hamiltonians.TFIM(nqubits=5, trotter=True)
    term = hamiltonians.TFIM(nqubits=2, numpy=True)
    h2 = hamiltonians.TrotterHamiltonian({(0, 1): term, (2, 3): term, (4, 0): term},
                                         {(1, 2): term, (3, 4): term})
    with pytest.raises(ValueError):
        h = h1 + h2
    # test subtraction with incompatible parts
    h2 = hamiltonians.TrotterHamiltonian({(0, 1): term, (2, 3): term},
                                         {(1, 2): term}, {(4, 0): term})
    with pytest.raises(ValueError):
        h = h1 - h2
    # test matmul with bad type
    with pytest.raises(NotImplementedError):
        s = h1 @ "abc"
    # test matmul with bad shape
    with pytest.raises(ValueError):
        s = h1 @ np.zeros((2, 2))
    # test ``make_compatible`` with non-Trotter Hamiltonian
    with pytest.raises(TypeError):
        h2 = h1.make_compatible("test")
    # test ``make_compatible`` with interacting Hamiltonian
    with pytest.raises(NotImplementedError):
        h2 = h1.make_compatible(h2)
    # test ``make_compatible`` with insufficient two-qubit terms
    h3 = hamiltonians.X(nqubits=7, trotter=True)
    with pytest.raises(ValueError):
        h3 = h1.make_compatible(h3)


def test_trotter_hamiltonian_three_qubit_term(backend):
    """Test creating ``TrotterHamiltonian`` with three qubit term."""
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    from scipy.linalg import expm
    m1 = random_hermitian(3)
    m2 = random_hermitian(2)
    m3 = random_hermitian(1)

    term1 = hamiltonians.Hamiltonian(3, m1, numpy=True)
    term2 = hamiltonians.Hamiltonian(2, m2, numpy=True)
    term3 = hamiltonians.Hamiltonian(1, m3, numpy=True)
    parts = [{(0, 1, 2): term1}, {(2, 3): term2, (1,): term3}]
    trotter_h = hamiltonians.TrotterHamiltonian(*parts)

    # Test that the `TrotterHamiltonian` dense matrix is correct
    eye = np.eye(2, dtype=m1.dtype)
    mm1 = np.kron(m1, eye)
    mm2 = np.kron(np.kron(eye, eye), m2)
    mm3 = np.kron(np.kron(eye, m3), np.kron(eye, eye))
    target_h = hamiltonians.Hamiltonian(4, mm1 + mm2 + mm3)
    np.testing.assert_allclose(trotter_h.dense.matrix, target_h.matrix)

    dt = 1e-2
    initial_state = random_state(4)
    if K.op is not None:
        with pytest.raises(NotImplementedError):
            circuit = trotter_h.circuit(dt=dt)
    else:
        circuit = trotter_h.circuit(dt=dt)
        final_state = circuit(np.copy(initial_state))

        u = [expm(-0.5j * dt * m) for m in [mm1, mm2, mm3]]
        target_state = u[2].dot(u[1].dot(u[0])).dot(initial_state)
        target_state = u[0].dot(u[1].dot(u[2])).dot(target_state)
        np.testing.assert_allclose(final_state, target_state)

    qibo.set_backend(original_backend)


def test_trotter_hamiltonian_make_compatible_simple():
    """Test ``make_compatible`` on a simple 3-qubit example."""
    h0target = hamiltonians.X(3)
    h0 = hamiltonians.X(3, trotter=True)
    term1 = hamiltonians.Y(1, numpy=True)
    term2 = hamiltonians.TFIM(2, numpy=True)
    parts = [{(0, 1): term2, (1, 2): term2, (0, 2): term2, (2,): term1}]
    h1 = hamiltonians.TrotterHamiltonian(*parts)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    np.testing.assert_allclose(h0c.matrix, h0target.matrix)


def test_trotter_hamiltonian_make_compatible_redundant():
    """Test ``make_compatible`` with redudant two-qubit terms."""
    h0 = hamiltonians.X(2, trotter=True)
    target_matrix = K.to_numpy(h0.dense.matrix)
    target_matrix = np.kron(target_matrix, np.eye(2, dtype=target_matrix.dtype))
    parts = [{(0, 1, 2): hamiltonians.TFIM(3, numpy=True)}]
    h1 = hamiltonians.TrotterHamiltonian(*parts)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    np.testing.assert_allclose(h0c.matrix, target_matrix)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_trotter_hamiltonian_make_compatible(nqubits):
    """Test that ``make_compatible`` method works for ``X`` Hamiltonian."""
    h0target = hamiltonians.X(nqubits)
    h0 = hamiltonians.X(nqubits, trotter=True)
    h1 = hamiltonians.XXZ(nqubits, delta=0.5, trotter=True)
    assert not h1.is_compatible(h0)
    assert not h0.is_compatible(h1)
    np.testing.assert_allclose(h0.matrix, h0target.matrix)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    assert h0c.is_compatible(h1)
    np.testing.assert_allclose(h0.matrix, h0target.matrix)
    np.testing.assert_allclose(h0c.matrix, h0target.matrix)
    # for coverage
    h0c = h1.make_compatible(h0c)
    assert not h1.is_compatible("test")
    h2 = hamiltonians.XXZ(nqubits, delta=0.5, trotter=True)
    h2.parts[0].pop((0, 1))
    assert not h1.is_compatible(h2)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_trotter_hamiltonian_make_compatible_repeating(nqubits):
    """Check ``make_compatible`` when first target is repeated in parts."""
    h0target = hamiltonians.X(nqubits)
    h0 = hamiltonians.X(nqubits, trotter=True)
    term = hamiltonians.TFIM(2, numpy=True)
    parts = [{(0, i): term} for i in range(1, nqubits)]
    parts.extend(({(i, 0): term} for i in range(1, nqubits)))
    h1 = hamiltonians.TrotterHamiltonian(*parts)

    h0c = h1.make_compatible(h0)
    assert not h1.is_compatible(h0)
    assert h1.is_compatible(h0c)
    np.testing.assert_allclose(h0c.matrix, h0target.matrix)


def test_trotter_hamiltonian_make_compatible_onequbit_terms():
    """Check ``make_compatible`` when the two-qubit Hamiltonian has one-qubit terms."""
    from qibo import matrices
    term1 = hamiltonians.Hamiltonian(1, matrices.Z, numpy=True)
    term2 = hamiltonians.Hamiltonian(2, np.kron(matrices.Z, matrices.Z), numpy=True)
    terms = {(0, 1): term2,
             (0, 2): -0.5 * term2,
             (1, 2): 2 * term2,
             (1,): 0.35 * term1,
             (2, 3): 0.25 * term2,
             (2,): 0.5 * term1,
             (3,): term1}
    tham = hamiltonians.TrotterHamiltonian.from_dictionary(terms) + 1.5
    xham = hamiltonians.X(nqubits=4, trotter=True)
    cxham = tham.make_compatible(xham)
    assert not tham.is_compatible(xham)
    assert tham.is_compatible(cxham)
    np.testing.assert_allclose(xham.dense.matrix, cxham.dense.matrix)
