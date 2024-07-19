"""Test methods of :class:`qibo.core.hamiltonians.SymbolicHamiltonian`."""

import numpy as np
import pytest
import sympy
from pytest import approx

from qibo import Circuit, gates, hamiltonians
from qibo.quantum_info.random_ensembles import random_density_matrix, random_statevector
from qibo.symbols import I, Y, Z


def symbolic_tfim(nqubits, h=1.0):
    """Constructs symbolic Hamiltonian for TFIM."""
    from qibo.symbols import X, Z

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
        ham = hamiltonians.SymbolicHamiltonian(
            Z * X, symbol_map=symbol_map, backend=backend
        )
    # Invalid operation in Hamiltonian expresion
    ham = hamiltonians.SymbolicHamiltonian(
        sympy.cos(Z), symbol_map=symbol_map, backend=backend
    )
    with pytest.raises(TypeError):
        dense = ham.dense


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("calcterms", [False, True])
def test_symbolictfim_hamiltonian_to_dense(backend, nqubits, calcterms):
    final_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1), backend=backend
    )
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
    local_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    target_ham = 2 * hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 * local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
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
    local_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    target_ham = 2 + hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 + local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
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
    local_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    target_ham = 2 - hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (2 - local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) - 2
    local_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    local_dense = (local_ham - 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_operator_add_and_sub(
    backend, nqubits, calcterms, calcdense
):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham1 = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    local_ham2 = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=0.5), backend=backend
    )
    if calcterms:
        _ = local_ham1.terms
        _ = local_ham2.terms
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_ham = local_ham1 + local_ham2
    target_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) + hamiltonians.TFIM(
        nqubits, h=0.5, backend=backend
    )
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham1 = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    local_ham2 = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=0.5), backend=backend
    )
    if calcterms:
        _ = local_ham1.terms
        _ = local_ham2.terms
    if calcdense:
        _ = local_ham1.dense
        _ = local_ham2.dense
    local_ham = local_ham1 - local_ham2
    target_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) - hamiltonians.TFIM(
        nqubits, h=0.5, backend=backend
    )
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [5])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_hamiltonianmatmul(backend, nqubits, calcterms, calcdense):
    local_ham1 = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    local_ham2 = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=0.5), backend=backend
    )
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
    state = (
        random_density_matrix(2**nqubits, backend=backend)
        if density_matrix
        else random_statevector(2**nqubits, backend=backend)
    )
    local_ham = hamiltonians.SymbolicHamiltonian(
        symbolic_tfim(nqubits, h=1.0), backend=backend
    )
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend)
    if calcterms:
        _ = local_ham.terms
    local_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    backend.assert_allclose(local_matmul, target_matmul)


@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_state_expectation(
    backend, nqubits, normalize, calcterms, calcdense
):
    local_ham = (
        hamiltonians.SymbolicHamiltonian(symbolic_tfim(nqubits, h=1.0), backend=backend)
        + 2
    )
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense
    dense_ham = hamiltonians.TFIM(nqubits, h=1.0, backend=backend) + 2

    state = random_statevector(2**nqubits, backend=backend)

    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(local_ev, target_ev)

    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(local_ev, target_ev)


@pytest.mark.parametrize("give_nqubits", [False, True])
@pytest.mark.parametrize("calcterms", [False, True])
@pytest.mark.parametrize("calcdense", [False, True])
def test_symbolic_hamiltonian_state_expectation_different_nqubits(
    backend, give_nqubits, calcterms, calcdense
):
    expr = symbolic_tfim(3, h=1.0)
    if give_nqubits:
        local_ham = hamiltonians.SymbolicHamiltonian(expr, nqubits=5, backend=backend)
    else:
        local_ham = hamiltonians.SymbolicHamiltonian(expr, backend=backend)
    if calcterms:
        _ = local_ham.terms
    if calcdense:
        _ = local_ham.dense

    dense_ham = hamiltonians.TFIM(3, h=1.0, backend=backend)
    dense_matrix = np.kron(backend.to_numpy(dense_ham.matrix), np.eye(4))
    dense_ham = hamiltonians.Hamiltonian(5, dense_matrix, backend=backend)

    state = random_statevector(2**5, backend=backend)

    if give_nqubits:
        local_ev = local_ham.expectation(state)
        target_ev = dense_ham.expectation(state)
        backend.assert_allclose(local_ev, target_ev)

        local_ev = local_ham.expectation(state)
        target_ev = dense_ham.expectation(state)
        backend.assert_allclose(local_ev, target_ev)
    else:
        with pytest.raises(ValueError):
            local_ev = local_ham.expectation(state)
        with pytest.raises(ValueError):
            local_ev = local_ham.expectation(state)


def test_hamiltonian_expectation_from_samples(backend):
    """Test Hamiltonian expectation value calculation."""
    backend.set_seed(0)
    obs0 = 2 * Z(0) * Z(1) + Z(0) * Z(2)
    obs1 = 2 * Z(0) * Z(1) + Z(0) * Z(2) * I(3)
    h0 = hamiltonians.SymbolicHamiltonian(obs0, backend=backend)
    h1 = hamiltonians.SymbolicHamiltonian(obs1, backend=backend)
    c = Circuit(4)
    c.add(gates.RX(0, np.random.rand()))
    c.add(gates.RX(1, np.random.rand()))
    c.add(gates.RX(2, np.random.rand()))
    c.add(gates.RX(3, np.random.rand()))
    c.add(gates.M(0, 1, 2, 3))
    nshots = 10**5
    result = backend.execute_circuit(c, nshots=nshots)
    freq = result.frequencies(binary=True)
    ev0 = h0.expectation_from_samples(freq, qubit_map=None)
    ev1 = h1.expectation(result.state())
    backend.assert_allclose(ev0, ev1, atol=20 / np.sqrt(nshots))


def test_hamiltonian_expectation_from_samples_errors(backend):
    obs = [Z(0) * Y(1), Z(0) * Z(1) ** 3]
    h1 = hamiltonians.SymbolicHamiltonian(obs[0], backend=backend)
    h2 = hamiltonians.SymbolicHamiltonian(obs[1], backend=backend)
    with pytest.raises(NotImplementedError):
        h1.expectation_from_samples(None, qubit_map=None)
    with pytest.raises(NotImplementedError):
        h2.expectation_from_samples(None, qubit_map=None)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("calcterms", [False, True])
def test_symbolic_hamiltonian_abstract_symbol_ev(backend, density_matrix, calcterms):
    from qibo.symbols import Symbol, X

    matrix = np.random.random((2, 2))
    form = X(0) * Symbol(1, matrix) + Symbol(0, matrix) * X(1)
    local_ham = hamiltonians.SymbolicHamiltonian(form, backend=backend)
    if calcterms:
        _ = local_ham.terms

    state = (
        random_density_matrix(4, backend=backend)
        if density_matrix
        else random_statevector(4, backend=backend)
    )
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


def test_symbolic_hamiltonian_with_constant(backend):
    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.M(0))
    h = hamiltonians.SymbolicHamiltonian(1e6 - Z(0), backend=backend)

    result = c.execute(nshots=10000)
    assert result.expectation_from_samples(h) == approx(1e6, rel=1e-5, abs=0.0)
