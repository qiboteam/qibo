"""Test symbolic Hamiltonian methods from `qibo/core/hamiltonians.py`."""
import pytest
import numpy as np
import sympy
from qibo import hamiltonians, matrices, K
from qibo.tests.utils import random_hermitian


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("trotter", [False, True])
def test_tfim_hamiltonian_from_symbols(nqubits, trotter):
    """Check creating TFIM Hamiltonian using sympy."""
    import sympy
    h = 0.5
    z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(nqubits))))
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(nqubits))))

    symham = sum(z_symbols[i] * z_symbols[i + 1] for i in range(nqubits - 1))
    symham += z_symbols[0] * z_symbols[-1]
    symham += h * sum(x_symbols)
    symmap = {z: (i, matrices.Z) for i, z in enumerate(z_symbols)}
    symmap.update({x: (i, matrices.X) for i, x in enumerate(x_symbols)})

    target_matrix = hamiltonians.TFIM(nqubits, h=h).matrix
    if trotter:
        trotter_ham = hamiltonians.TrotterHamiltonian.from_symbolic(-symham, symmap)
        final_matrix = trotter_ham.dense.matrix
    else:
        full_ham = hamiltonians.Hamiltonian.from_symbolic(-symham, symmap)
        final_matrix = full_ham.matrix
    np.testing.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("trotter", [False, True])
def test_from_symbolic_with_power(trotter):
    """Check ``from_symbolic`` when the expression contains powers."""
    z = sympy.symbols(" ".join((f"Z{i}" for i in range(3))))
    symham =  z[0] ** 2 - z[1] ** 2 + 3 * z[1] - 2 * z[0] * z[2] + + 1
    matrix = random_hermitian(1)
    symmap = {x: (i, matrix) for i, x in enumerate(z)}
    if trotter:
        ham = hamiltonians.TrotterHamiltonian.from_symbolic(symham, symmap)
        final_matrix = ham.dense.matrix
    else:
        ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap)
        final_matrix = ham.matrix

    matrix2 = matrix.dot(matrix)
    eye = np.eye(2, dtype=matrix.dtype)
    target_matrix = np.kron(np.kron(matrix2, eye), eye)
    target_matrix -= np.kron(np.kron(eye, matrix2), eye)
    target_matrix += 3 * np.kron(np.kron(eye, matrix), eye)
    target_matrix -= 2 * np.kron(np.kron(matrix, eye), matrix)
    target_matrix += np.eye(8, dtype=matrix.dtype)
    np.testing.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("trotter", [False, True])
def test_from_symbolic_with_complex_numbers(trotter):
    """Check ``from_symbolic`` when the expression contains imaginary unit."""
    import sympy
    x = sympy.symbols(" ".join((f"X{i}" for i in range(2))))
    y = sympy.symbols(" ".join((f"Y{i}" for i in range(2))))
    symham = (1 + 2j) * x[0] * x[1] + 2 * y[0] * y[1] - 3j * x[0] * y[1] + 1j * y[0] * x[1]
    symmap = {s: (i, matrices.X) for i, s in enumerate(x)}
    symmap.update({s: (i, matrices.Y) for i, s in enumerate(y)})
    if trotter:
        ham = hamiltonians.TrotterHamiltonian.from_symbolic(symham, symmap)
        final_matrix = ham.dense.matrix
    else:
        ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap)
        final_matrix = ham.matrix

    target_matrix = (1 + 2j) * np.kron(matrices.X, matrices.X)
    target_matrix += 2 * np.kron(matrices.Y, matrices.Y)
    target_matrix -= 3j * np.kron(matrices.X, matrices.Y)
    target_matrix += 1j * np.kron(matrices.Y, matrices.X)
    np.testing.assert_allclose(final_matrix, target_matrix)


def test_from_symbolic_application_hamiltonian():
    """Check ``from_symbolic`` for a specific four-qubit Hamiltonian."""
    z1, z2, z3, z4 = sympy.symbols("z1 z2 z3 z4")
    symmap = {z: (i, matrices.Z) for i, z in enumerate([z1, z2, z3, z4])}
    symham = (z1 * z2 - 0.5 * z1 * z3 + 2 * z2 * z3 + 0.35 * z2
              + 0.25 * z3 * z4 + 0.5 * z3 + z4 - z1)
    # Check that Trotter dense matrix agrees will full Hamiltonian matrix
    fham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap)
    tham = hamiltonians.TrotterHamiltonian.from_symbolic(symham, symmap)
    np.testing.assert_allclose(tham.dense.matrix, fham.matrix)
    # Check that no one-qubit terms exist in the Trotter Hamiltonian
    # (this means that merging was successful)
    first_targets = set()
    for part in tham.parts:
        for targets, term in part.items():
            first_targets.add(targets[0])
            assert len(targets) == 2
            assert term.nqubits == 2
    assert first_targets == set(range(4))
    # Check making an ``X`` Hamiltonian compatible with ``tham``
    xham = hamiltonians.X(nqubits=4, trotter=True)
    cxham = tham.make_compatible(xham)
    assert not tham.is_compatible(xham)
    assert tham.is_compatible(cxham)
    np.testing.assert_allclose(xham.dense.matrix, cxham.dense.matrix)


@pytest.mark.parametrize("nqubits", [4, 5])
@pytest.mark.parametrize("trotter", [False, True])
def test_x_hamiltonian_from_symbols(nqubits, trotter):
    """Check creating sum(X) Hamiltonian using sympy."""
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(nqubits))))
    symham =  -sum(x_symbols)
    symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}

    target_matrix = hamiltonians.X(nqubits).matrix
    if trotter:
        trotter_ham = hamiltonians.TrotterHamiltonian.from_symbolic(symham, symmap)
        final_matrix = trotter_ham.dense.matrix
    else:
        full_ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap)
        final_matrix = full_ham.matrix
    np.testing.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("trotter", [False, True])
def test_three_qubit_term_hamiltonian_from_symbols(trotter):
    """Check creating Hamiltonian with three-qubit interaction using sympy."""
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(4))))
    y_symbols = sympy.symbols(" ".join((f"Y{i}" for i in range(4))))
    z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(4))))
    symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}
    symmap.update({x: (i, matrices.Y) for i, x in enumerate(y_symbols)})
    symmap.update({x: (i, matrices.Z) for i, x in enumerate(z_symbols)})

    symham = x_symbols[0] * y_symbols[1] * z_symbols[2]
    symham += 0.5 * y_symbols[0] * z_symbols[1] * x_symbols[3]
    symham += z_symbols[0] * x_symbols[2]
    symham += -3 * x_symbols[1] * y_symbols[3]
    symham += y_symbols[2]
    symham += 1.5 * z_symbols[1]
    symham -= 2

    target_matrix = np.kron(np.kron(matrices.X, matrices.Y),
                            np.kron(matrices.Z, matrices.I))
    target_matrix += 0.5 * np.kron(np.kron(matrices.Y, matrices.Z),
                                   np.kron(matrices.I, matrices.X))
    target_matrix += np.kron(np.kron(matrices.Z, matrices.I),
                             np.kron(matrices.X, matrices.I))
    target_matrix += -3 * np.kron(np.kron(matrices.I, matrices.X),
                             np.kron(matrices.I, matrices.Y))
    target_matrix += np.kron(np.kron(matrices.I, matrices.I),
                             np.kron(matrices.Y, matrices.I))
    target_matrix += 1.5 * np.kron(np.kron(matrices.I, matrices.Z),
                                   np.kron(matrices.I, matrices.I))
    target_matrix -= 2 * np.eye(2**4, dtype=target_matrix.dtype)
    if trotter:
        trotter_ham = hamiltonians.TrotterHamiltonian.from_symbolic(symham, symmap)
        final_matrix = trotter_ham.dense.matrix
    else:
        full_ham = hamiltonians.Hamiltonian.from_symbolic(symham, symmap)
        final_matrix = full_ham.matrix
    np.testing.assert_allclose(final_matrix, target_matrix)


@pytest.mark.parametrize("sufficient", [True, False])
def test_symbolic_hamiltonian_merge_one_qubit(sufficient):
    """Check that ``merge_one_qubit`` works both when two-qubit are sufficient and no."""
    from qibo.hamiltonians import TrotterHamiltonian
    x_symbols = sympy.symbols(" ".join((f"X{i}" for i in range(5))))
    z_symbols = sympy.symbols(" ".join((f"Z{i}" for i in range(5))))
    symmap = {x: (i, matrices.X) for i, x in enumerate(x_symbols)}
    symmap.update({x: (i, matrices.Z) for i, x in enumerate(z_symbols)})
    symham = sum(z_symbols[i] * z_symbols[i + 1] for i in range(4))
    symham += sum(x_symbols)
    if sufficient:
        symham += z_symbols[0] * z_symbols[-1]
    merged, _ = TrotterHamiltonian.symbolic_terms(symham, symmap)

    two_qubit_keys = {(i, i + 1) for i in range(4)}
    if sufficient:
        target_matrix = (np.kron(matrices.Z, matrices.Z) +
                         np.kron(matrices.X, matrices.I))
        two_qubit_keys.add((4, 0))
        assert set(merged.keys()) == two_qubit_keys
        for matrix in merged.values():
            np.testing.assert_allclose(matrix, target_matrix)
    else:
        one_qubit_keys = {(i,) for i in range(5)}
        assert set(merged.keys()) == one_qubit_keys | two_qubit_keys
        target_matrix = matrices.X
        for t in one_qubit_keys:
            np.testing.assert_allclose(merged[t], target_matrix)
        target_matrix = np.kron(matrices.Z, matrices.Z)
        for t in two_qubit_keys:
            np.testing.assert_allclose(merged[t], target_matrix)


def test_symbolic_hamiltonian_errors():
    """Check errors raised by `SymbolicHamiltonian`."""
    from qibo.core.symbolic import SymbolicHamiltonian
    a, b = sympy.symbols("a b")
    ham = a * b
    # Bad hamiltonian type
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian("test", "test")
    # Bad symbol map type
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian(ham, "test")
    # Bad symbol map key
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian(ham, {"a": 2})
    # Bad symbol map value
    with pytest.raises(TypeError):
        sh = SymbolicHamiltonian(ham, {a: 2})
    with pytest.raises(ValueError):
        sh = SymbolicHamiltonian(ham, {a: (1, 2, 3)})
    # Missing symbol
    with pytest.raises(ValueError):
        sh = SymbolicHamiltonian(ham, {a: (0, matrices.X)})
    # Factor that cannot be parsed
    ham = a * b + sympy.cos(a) * b
    with pytest.raises(ValueError):
        sh = SymbolicHamiltonian(ham, {a: (0, matrices.X), b: (1, matrices.Z)})
