"""Test methods of :class:`qibo.core.SymbolicHamiltonian`."""

import numpy as np
import pytest
import sympy

from qibo import Circuit, gates
from qibo.hamiltonians import TFIM, XXZ, Hamiltonian, SymbolicHamiltonian
from qibo.quantum_info.random_ensembles import random_density_matrix, random_statevector
from qibo.symbols import I, Symbol, X, Y, Z


def symbolic_tfim(nqubits, backend, h=1.0):
    """Constructs symbolic Hamiltonian for TFIM."""
    from qibo.symbols import X, Z

    sham = -sum(
        Z(i, backend=backend) * Z(i + 1, backend=backend) for i in range(nqubits - 1)
    )
    sham -= Z(0, backend=backend) * Z(nqubits - 1, backend=backend)
    sham -= h * sum(X(i, backend=backend) for i in range(nqubits))
    return sham


def test_symbolic_hamiltonian_errors(backend):
    # Wrong type of Symbol matrix
    with pytest.raises(TypeError):
        s = Symbol(0, "test")
    # Wrong type of symbolic expression
    with pytest.raises(TypeError):
        ham = SymbolicHamiltonian("test", backend=backend)
    # Passing form with symbol that is not a ``qibo.symbols.Symbol``
    from qibo import matrices

    z, x = sympy.Symbol("z"), sympy.Symbol("x")
    with pytest.raises(RuntimeError):
        ham = SymbolicHamiltonian(z * x, backend=backend)
    # Invalid operation in Hamiltonian expresion
    ham = SymbolicHamiltonian(sympy.cos(Z(0, backend=backend)), backend=backend)
    with pytest.raises(TypeError):
        dense = ham.dense


def test_symbolic_hamiltonian_form_setter(backend):
    h = SymbolicHamiltonian(Z(0), backend=backend)
    new_form = Z(0) * X(1) * Y(3)
    h.form = new_form
    assert h.form == new_form
    assert h.nqubits == 4


def test_symbolic_hamiltonian_dense(backend):
    target_matrix = backend.cast(
        Z(0, backend=backend).matrix @ Z(0, backend=backend).matrix
        + X(0, backend=backend).matrix @ Y(0, backend=backend).matrix
        + backend.matrices.I()
    )
    form = Z(0) ** 2 + X(0) * Y(0) + 1
    sham = SymbolicHamiltonian(form, nqubits=1, backend=backend)
    backend.assert_allclose(sham.dense.matrix, target_matrix)


@pytest.mark.parametrize("nqubits", [3, 4])
def test_symbolictfim_hamiltonian_to_dense(backend, nqubits):
    final_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1), backend=backend
    )
    target_ham = TFIM(nqubits, h=1, backend=backend)
    backend.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


@pytest.mark.parametrize("nqubits", [3, 4])
def test_symbolicxxz_hamiltonian_to_dense(backend, nqubits):
    sham = sum(
        X(i, backend=backend) * X(i + 1, backend=backend) for i in range(nqubits - 1)
    )
    sham += sum(
        Y(i, backend=backend) * Y(i + 1, backend=backend) for i in range(nqubits - 1)
    )
    sham += 0.5 * sum(
        Z(i, backend=backend) * Z(i + 1, backend=backend) for i in range(nqubits - 1)
    )
    sham += (
        X(0, backend=backend) * X(nqubits - 1, backend=backend)
        + Y(0, backend=backend) * Y(nqubits - 1, backend=backend)
        + 0.5 * Z(0, backend=backend) * Z(nqubits - 1, backend=backend)
    )
    final_ham = SymbolicHamiltonian(sham, backend=backend)
    target_ham = XXZ(nqubits, backend=backend)
    backend.assert_allclose(final_ham.matrix, target_ham.matrix, atol=1e-15)


@pytest.mark.parametrize("nqubits", [3])
def test_symbolic_hamiltonian_scalar_mul(backend, nqubits):
    """Test multiplication of Trotter Hamiltonian with scalar."""
    local_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    target_ham = 2 * TFIM(nqubits, h=1.0, backend=backend)
    local_dense = (2 * local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    local_dense = (local_ham * 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [4])
def test_symbolic_hamiltonian_scalar_add(backend, nqubits):
    """Test addition of Trotter Hamiltonian with scalar."""
    local_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    target_ham = 2 + TFIM(nqubits, h=1.0, backend=backend)
    local_dense = (2 + local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    local_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    local_dense = (local_ham + 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [3])
def test_symbolic_hamiltonian_scalar_sub(backend, nqubits):
    """Test subtraction of Trotter Hamiltonian with scalar."""
    local_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    target_ham = 2 - TFIM(nqubits, h=1.0, backend=backend)
    local_dense = (2 - local_ham).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)

    target_ham = TFIM(nqubits, h=1.0, backend=backend) - 2
    local_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    local_dense = (local_ham - 2).dense
    backend.assert_allclose(local_dense.matrix, target_ham.matrix)


@pytest.mark.parametrize("nqubits", [3])
def test_symbolic_hamiltonian_operator_add_and_sub(backend, nqubits):
    """Test addition and subtraction between Trotter Hamiltonians."""
    local_ham1 = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    local_ham2 = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=0.5), backend=backend
    )
    local_ham = local_ham1 + local_ham2
    target_ham = TFIM(nqubits, h=1.0, backend=backend) + TFIM(
        nqubits, h=0.5, backend=backend
    )
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)

    local_ham1 = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    local_ham2 = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=0.5), backend=backend
    )
    local_ham = local_ham1 - local_ham2
    target_ham = TFIM(nqubits, h=1.0, backend=backend) - TFIM(
        nqubits, h=0.5, backend=backend
    )
    dense = local_ham.dense
    backend.assert_allclose(dense.matrix, target_ham.matrix)

    # Test multiplication and sum
    target = XXZ(nqubits, backend=backend)
    term_1 = SymbolicHamiltonian(
        X(0, backend=backend) * X(1, backend=backend)
        + X(1, backend=backend) * X(2, backend=backend)
        + X(0, backend=backend) * X(2, backend=backend),
        backend=backend,
    )
    term_2 = SymbolicHamiltonian(
        Y(0, backend=backend) * Y(1, backend=backend)
        + Y(1, backend=backend) * Y(2, backend=backend)
        + Y(0, backend=backend) * Y(2, backend=backend),
        backend=backend,
    )
    term_3 = SymbolicHamiltonian(
        Z(0, backend=backend) * Z(1, backend=backend)
        + Z(1, backend=backend) * Z(2, backend=backend)
        + Z(0, backend=backend) * Z(2, backend=backend),
        backend=backend,
    )
    hamiltonian = term_1 + term_2 + 0.5 * term_3
    matrix = hamiltonian.dense.matrix

    backend.assert_allclose(matrix, target.matrix)


@pytest.mark.parametrize("nqubits", [5])
def test_symbolic_hamiltonian_hamiltonianmatmul(backend, nqubits):
    local_ham1 = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    local_ham2 = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=0.5), backend=backend
    )
    dense_ham1 = TFIM(nqubits, h=1.0, backend=backend)
    dense_ham2 = TFIM(nqubits, h=0.5, backend=backend)
    local_matmul = local_ham1 @ local_ham2
    target_matmul = dense_ham1 @ dense_ham2
    backend.assert_allclose(local_matmul.matrix, target_matmul.matrix)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("density_matrix", [False, True])
def test_symbolic_hamiltonian_matmul(backend, nqubits, density_matrix):
    state = (
        random_density_matrix(2**nqubits, backend=backend)
        if density_matrix
        else random_statevector(2**nqubits, backend=backend)
    )
    local_ham = SymbolicHamiltonian(
        symbolic_tfim(nqubits, backend, h=1.0), backend=backend
    )
    dense_ham = TFIM(nqubits, h=1.0, backend=backend)
    local_matmul = local_ham @ state
    target_matmul = dense_ham @ state
    backend.assert_allclose(local_matmul, target_matmul)


@pytest.mark.parametrize("nqubits,normalize", [(3, False), (4, False)])
def test_symbolic_hamiltonian_state_expectation(backend, nqubits, normalize):
    local_ham = (
        SymbolicHamiltonian(symbolic_tfim(nqubits, backend, h=1.0), backend=backend) + 2
    )
    dense_ham = TFIM(nqubits, h=1.0, backend=backend) + 2

    state = random_statevector(2**nqubits, backend=backend)

    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(local_ev, target_ev)

    local_ev = local_ham.expectation(state, normalize)
    target_ev = dense_ham.expectation(state, normalize)
    backend.assert_allclose(local_ev, target_ev)


@pytest.mark.parametrize("give_nqubits", [False, True])
def test_symbolic_hamiltonian_state_expectation_different_nqubits(
    backend, give_nqubits
):
    expr = symbolic_tfim(3, backend, h=1.0)
    if give_nqubits:
        local_ham = SymbolicHamiltonian(expr, nqubits=5, backend=backend)
    else:
        local_ham = SymbolicHamiltonian(expr, backend=backend)

    dense_ham = TFIM(3, h=1.0, backend=backend)
    dense_matrix = np.kron(backend.to_numpy(dense_ham.matrix), np.eye(4))
    dense_ham = Hamiltonian(5, dense_matrix, backend=backend)

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


@pytest.mark.parametrize(
    "observable,qubit_map",
    [
        (2 * Z(0) * Z(3) + Z(0) * Z(2), [0, 1, 2, 3]),
        (Z(1) + Z(3), [0, 1, 3]),
    ],
)
def test_hamiltonian_expectation_from_samples(backend, observable, qubit_map):
    """Test Hamiltonian expectation value calculation."""
    backend.set_seed(0)
    hamiltonian = SymbolicHamiltonian(observable, backend=backend)
    c = Circuit(4)
    c.add(gates.RX(0, np.random.rand()))
    c.add(gates.RX(1, np.random.rand()))
    c.add(gates.RX(2, np.random.rand()))
    c.add(gates.RX(3, np.random.rand()))
    c.add(gates.M(*qubit_map))
    nshots = 10**5
    result = backend.execute_circuit(c, nshots=nshots)
    freq = result.frequencies(binary=True)
    from_samples = hamiltonian.expectation_from_samples(freq, qubit_map=qubit_map)
    from_state = hamiltonian.expectation(result.state())
    backend.assert_allclose(from_samples, from_state, atol=20 / np.sqrt(nshots))


@pytest.mark.parametrize("density_matrix", [False, True])
def test_symbolic_hamiltonian_abstract_symbol_ev(backend, density_matrix):
    from qibo.symbols import Symbol, X

    matrix = np.random.random((2, 2))
    form = X(0, backend=backend) * Symbol(1, matrix, backend=backend) + Symbol(
        0, matrix, backend=backend
    ) * X(1, backend=backend)
    local_ham = SymbolicHamiltonian(form, backend=backend)

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
    h1 = SymbolicHamiltonian(symbolic_tfim(3, backend, h=1.0), backend=backend)
    h2 = SymbolicHamiltonian(symbolic_tfim(4, backend, h=1.0), backend=backend)
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
    h2 = XXZ(3, dense=False, backend=backend)


def test_symbolic_hamiltonian_with_constant(backend):
    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.M(0))
    h = SymbolicHamiltonian(1e6 - Z(0), backend=backend)

    result = c.execute(nshots=10000)
    assert float(result.expectation_from_samples(h)) == pytest.approx(
        1e6, rel=1e-5, abs=0.0
    )
