from functools import reduce

import numpy as np
import pytest
from scipy.linalg import sqrtm

from qibo import Circuit, gates, matrices
from qibo.quantum_info._linalg_operations import (
    _gram_schmidt_process,
    _vector_projection,
)
from qibo.quantum_info.linalg_operations import (
    anticommutator,
    commutator,
    lanczos,
    lie_closure,
    matrix_exponentiation,
    matrix_logarithm,
    matrix_power,
    matrix_sqrt,
    partial_trace,
    partial_transpose,
    schmidt_decomposition,
    singular_value_decomposition,
)
from qibo.quantum_info.metrics import infidelity, purity
from qibo.quantum_info.random_ensembles import (
    random_density_matrix,
    random_hermitian,
    random_statevector,
)


def test_commutator(backend):
    matrix_1 = np.random.rand(2, 2, 2)
    matrix_1 = backend.cast(matrix_1, dtype=matrix_1.dtype)

    matrix_2 = np.random.rand(2, 2)
    matrix_2 = backend.cast(matrix_2, dtype=matrix_2.dtype)

    matrix_3 = np.random.rand(3, 3)
    matrix_3 = backend.cast(matrix_3, dtype=matrix_3.dtype)

    with pytest.raises(TypeError):
        commutator(matrix_1, matrix_2)
    with pytest.raises(TypeError):
        commutator(matrix_2, matrix_1)
    with pytest.raises(TypeError):
        commutator(matrix_2, matrix_3)

    I, X, Y, Z = (
        backend.matrices.I(2),
        backend.matrices.X,
        backend.matrices.Y,
        backend.matrices.Z,
    )

    comm = commutator(X, I)
    backend.assert_allclose(comm, backend.zeros((2, 2)))

    comm = commutator(X, X)
    backend.assert_allclose(comm, backend.zeros((2, 2)))

    comm = commutator(X, Y)
    backend.assert_allclose(comm, 2j * Z)

    comm = commutator(X, Z)
    backend.assert_allclose(comm, -2j * Y)


def test_anticommutator(backend):
    matrix_1 = np.random.rand(2, 2, 2)
    matrix_1 = backend.cast(matrix_1, dtype=matrix_1.dtype)

    matrix_2 = np.random.rand(2, 2)
    matrix_2 = backend.cast(matrix_2, dtype=matrix_2.dtype)

    matrix_3 = np.random.rand(3, 3)
    matrix_3 = backend.cast(matrix_3, dtype=matrix_3.dtype)

    with pytest.raises(TypeError):
        anticommutator(matrix_1, matrix_2)
    with pytest.raises(TypeError):
        anticommutator(matrix_2, matrix_1)
    with pytest.raises(TypeError):
        anticommutator(matrix_2, matrix_3)

    I, X, Y, Z = matrices.I, matrices.X, matrices.Y, matrices.Z
    I = backend.cast(I, dtype=I.dtype)
    X = backend.cast(X, dtype=X.dtype)
    Y = backend.cast(Y, dtype=Y.dtype)
    Z = backend.cast(Z, dtype=Z.dtype)

    anticomm = anticommutator(X, I)
    backend.assert_allclose(anticomm, 2 * X)

    anticomm = anticommutator(X, X)
    backend.assert_allclose(anticomm, 2 * I)

    anticomm = anticommutator(X, Y)
    backend.assert_allclose(anticomm, backend.zeros((2, 2)))

    anticomm = anticommutator(X, Z)
    backend.assert_allclose(anticomm, backend.zeros((2, 2)))


@pytest.mark.parametrize("density_matrix", [False, True])
def test_partial_trace(backend, density_matrix):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2).astype(complex)
        state += 1j * np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        partial_trace(state, 1, backend=backend)
    with pytest.raises(ValueError):
        state = (
            random_density_matrix(5, backend=backend)
            if density_matrix
            else random_statevector(5, backend=backend)
        )
        partial_trace(state, 1, backend=backend)

    nqubits = 4
    circuit = Circuit(nqubits, density_matrix=density_matrix)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, qubit + 1) for qubit in range(1, nqubits - 1))
    state = backend.execute_circuit(circuit).state()

    traced = partial_trace(state, (1, 2, 3), backend=backend)

    Id = backend.maximally_mixed_state(1)

    backend.assert_allclose(traced, Id)


def _werner_state(p, backend):
    zero, one = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    psi = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)
    psi = np.outer(psi, np.conj(psi.T))
    psi = backend.cast(psi, dtype=psi.dtype)

    state = p * psi + (1 - p) * backend.maximally_mixed_state(2)

    # partial transpose of two-qubit werner state is known analytically
    transposed = (1 / 4) * np.array(
        [
            [1 - p, 0, 0, -2 * p],
            [0, p + 1, 0, 0],
            [0, 0, p + 1, 0],
            [-2 * p, 0, 0, 1 - p],
        ],
        dtype=complex,
    )
    transposed = backend.cast(transposed, dtype=transposed.dtype)

    return state, transposed


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("statevector", [False, True])
@pytest.mark.parametrize("p", [1 / 5, 1 / 3, 1.0])
def test_partial_transpose(backend, p, statevector, batch):
    with pytest.raises(ValueError):
        state = random_density_matrix(3, backend=backend)
        partial_transpose(state, [0], backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2, 2).astype(complex)
        state += 1j * np.random.rand(2, 2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        partial_transpose(state, [1], backend=backend)

    if statevector:
        zero, one = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
        psi = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)

        # testing statevector
        target = np.zeros((4, 4), dtype=complex)
        target[0, 3] = -1 / 2
        target[1, 1] = 1 / 2
        target[2, 2] = 1 / 2
        target[3, 0] = -1 / 2
        target = backend.cast(target, dtype=target.dtype)

        psi = backend.cast(psi, dtype=psi.dtype)

        if batch:
            # the inner cast is required because of torch
            psi = backend.cast([backend.cast([psi]) for _ in range(2)])

        transposed = partial_transpose(psi, [0], backend=backend)

        if batch:
            for j in range(2):
                backend.assert_allclose(transposed[j], target)
        else:
            backend.assert_allclose(transposed, target)
    else:
        state, target = _werner_state(p, backend)
        if batch:
            state = backend.cast([state for _ in range(2)])

        # partial transpose of two-qubit werner state is known analytically
        target = (1 / 4) * np.array(
            [
                [1 - p, 0, 0, -2 * p],
                [0, p + 1, 0, 0],
                [0, 0, p + 1, 0],
                [-2 * p, 0, 0, 1 - p],
            ],
            dtype=complex,
        )
        target = backend.cast(target, dtype=target.dtype)

        transposed = partial_transpose(state, [1], backend)

        if batch:
            for j in range(2):
                backend.assert_allclose(transposed[j], target)
        else:
            backend.assert_allclose(transposed, target)


def test_matrix_exponentiation(backend):
    phase = 0.1
    target = (
        backend.cos(0.1) * backend.matrices.I()
        + 1j * backend.sin(phase) * backend.matrices.X
    )

    matrix = matrix_exponentiation(backend.matrices.X, 1j * phase, backend=backend)

    backend.assert_allclose(matrix, target)


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("power", [-0.5, 0.5, 2, 2.0, "2"])
def test_matrix_power(backend, power, singular):
    nqubits = 2
    dims = 2**nqubits

    state = random_density_matrix(dims, pure=singular, backend=backend)

    if isinstance(power, str):
        with pytest.raises(TypeError):
            matrix_power(state, power, backend=backend)
    elif power == -0.5 and singular:
        # When the singular matrix is a state, this power should be itself
        backend.assert_allclose(matrix_power(state, power, backend=backend), state)
    elif abs(power) == 0.5 and not singular:
        # Should be equal to the (inverse) square root
        sqrt = sqrtm(backend.to_numpy(state)).astype(complex)
        if power == -0.5:
            sqrt = np.linalg.inv(sqrt)
        sqrt = backend.cast(sqrt)

        backend.assert_allclose(matrix_power(state, power, backend=backend), sqrt)
    else:
        power = matrix_power(state, power, backend=backend)

        target = float(backend.real(backend.trace(power)))

        assert abs(purity(state, backend=backend) - target) < 1e-5


def test_matrix_sqrt(backend):
    nqubits = 2
    dims = 2**nqubits

    state = random_density_matrix(dims, pure=False, backend=backend)

    eigvals, eigvecs = backend.eigenvectors(state)
    target = backend.zeros_like(state)
    for eigval, eigvec in zip(eigvals, eigvecs.T):
        target += backend.sqrt(eigval) * backend.outer(eigvec, backend.conj(eigvec))

    sqrt = matrix_sqrt(state, backend=backend)

    backend.assert_allclose(sqrt, target)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_matrix_log(backend, base):
    nqubits = 2
    dims = 2**nqubits

    state = random_density_matrix(dims, pure=False, backend=backend)

    eigvals, eigvecs = backend.eigenvectors(state)
    target = backend.zeros_like(state)
    for eigval, eigvec in zip(eigvals, eigvecs.T):
        target += (backend.log(eigval) / float(np.log(base))) * backend.outer(
            eigvec, backend.conj(eigvec)
        )

    sqrt = matrix_logarithm(state, base=base, backend=backend)
    backend.assert_allclose(sqrt, target)

    sqrt = matrix_logarithm(
        state, base=base, eigenvectors=eigvecs, eigenvalues=eigvals, backend=backend
    )
    backend.assert_allclose(sqrt, target)


def test_singular_value_decomposition(backend):
    zero = np.array([1, 0], dtype=complex)
    one = np.array([0, 1], dtype=complex)
    plus = (zero + one) / np.sqrt(2)
    minus = (zero - one) / np.sqrt(2)
    plus = backend.cast(plus, dtype=plus.dtype)
    minus = backend.cast(minus, dtype=minus.dtype)
    base = [plus, minus]

    coeffs = np.random.rand(4)
    coeffs /= np.sum(coeffs)
    coeffs = backend.cast(coeffs, dtype=coeffs.dtype)

    state = np.zeros((4, 4), dtype=complex)
    state = backend.cast(state, dtype=state.dtype)
    for k, coeff in enumerate(coeffs):
        bitstring = f"{k:0{2}b}"
        a, b = int(bitstring[0]), int(bitstring[1])
        ket = backend.kron(base[a], base[b])
        state = state + coeff * backend.outer(ket, ket.T)

    _, S, _ = singular_value_decomposition(state, backend=backend)

    S_sorted = backend.sort(S)
    coeffs_sorted = backend.sort(coeffs)

    backend.assert_allclose(S_sorted, coeffs_sorted)


def test_schmidt_decomposition(backend):
    with pytest.raises(ValueError):
        test = random_statevector(3, backend=backend)
        test = schmidt_decomposition(test, [0], backend=backend)

    state_A = random_statevector(4, seed=10, backend=backend)
    state_B = random_statevector(4, seed=11, backend=backend)
    state = backend.kron(state_A, state_B)

    U, S, Vh = schmidt_decomposition(state, [0, 1], backend=backend)

    # recovering original state
    recovered = backend.zeros_like(state, dtype=backend.complex128)
    for coeff, u, vh in zip(S, U.T, Vh):
        if abs(coeff) > 1e-10:
            recovered = recovered + coeff * backend.kron(u, vh)

    backend.assert_allclose(recovered, state)

    # entropy test
    coeffs = backend.abs(S) ** 2
    entropy = backend.where(backend.abs(S) < 1e-10, 0.0, backend.log(coeffs))
    entropy = -backend.sum(coeffs * entropy)

    assert entropy < 1e-14


@pytest.mark.parametrize("seed", [None, 10])
@pytest.mark.parametrize("initial_vector", [None, True])
@pytest.mark.parametrize("nqubits", [4, 5])
def test_lanczos(backend, nqubits, initial_vector, seed):
    # Since Lanczos is designed for extremal eigenvalues,
    # only the first 5 eigenvalues and eigenvectors are tested.
    dims = 2**nqubits
    hamiltonian = random_hermitian(dims, seed=seed, backend=backend)

    eigvals_target, eigvectors_target = backend.eigh(hamiltonian)

    if initial_vector:
        initial_vector = random_statevector(dims, seed=20, backend=backend)

    tridiag, ortho_matrix = lanczos(
        hamiltonian, initial_vector=initial_vector, seed=seed, backend=backend
    )

    backend.assert_allclose(
        tridiag, backend.conj(ortho_matrix.T) @ hamiltonian @ ortho_matrix
    )

    eigvals, eigvectors = backend.eigh(tridiag)
    eigs = list(zip(eigvals, eigvectors.T))
    eigs.sort()
    eigvectors = [row[1] for row in eigs]
    eigvals = [float(row[0]) for row in eigs]

    eigvals = backend.cast(eigvals)
    eigvectors = backend.cast(eigvectors)

    backend.assert_allclose(eigvals[:5], eigvals_target[:5], atol=1e-2, rtol=1e-2)

    infidelities = [
        float(infidelity(ortho_matrix @ eigvec, eigvec_target, backend=backend))
        for eigvec, eigvec_target in zip(eigvectors, eigvectors_target.T)
    ]
    assert all(inf < 1e-3 for inf in infidelities[:5])


@pytest.mark.parametrize("seed", [10])
@pytest.mark.parametrize("nqubits", [4, 5])
def test_vector_projection_and_gram_schmidt_process(backend, nqubits, seed):
    dims = 2**nqubits
    state = random_statevector(dims, seed=seed, backend=backend)
    directions = [
        random_statevector(dims, seed=seed + k, backend=backend)
        for k in range(1, 5 + 1)
    ]

    # testing several projections
    target = backend.cast(
        [
            backend.dot(backend.conj(state), direction) * direction
            for direction in directions
        ]
    )
    projection = _vector_projection(state, directions, backend=backend)
    backend.assert_allclose(projection, target)

    # test gram-schmidt
    target_gs = backend.cast(state, copy=True)
    for direction in target:
        target_gs -= direction
    vector = _gram_schmidt_process(state, directions, backend=backend)
    backend.assert_allclose(vector, target_gs)

    # testing one projection
    projection = _vector_projection(state, directions[0], backend=backend)
    backend.assert_allclose(projection, target[0])

    # test gram-schmidt
    target_gs = state - target[0]
    vector = _gram_schmidt_process(state, directions[0], backend=backend)
    backend.assert_allclose(vector, target_gs)


def test_lie_closure_errors(backend):
    X, Z = backend.matrices.X, backend.matrices.Z

    with pytest.raises(TypeError):
        lie_closure([], backend=backend)
    with pytest.raises(TypeError):
        lie_closure([backend.zeros((2, 3))], backend=backend)
    with pytest.raises(TypeError):
        lie_closure([X, backend.kron(X, Z)], backend=backend)
    with pytest.raises(ValueError):
        lie_closure([backend.cast([[0.0, 1.0], [0.0, 0.0]])], backend=backend)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_lie_closure(backend, nqubits, skew):
    I, X, Z = backend.matrices.I(2), backend.matrices.X, backend.matrices.Z

    # transverse-field Ising chain, whose DLA is so(2 * nqubits)
    generators = []
    for qubit in range(nqubits):
        operators = [I] * nqubits
        operators[qubit] = Z
        generators.append(reduce(backend.kron, operators))
    for qubit in range(nqubits - 1):
        operators = [I] * nqubits
        operators[qubit] = X
        operators[qubit + 1] = X
        generators.append(reduce(backend.kron, operators))

    if skew:
        generators = [1j * gen for gen in generators]

    basis = lie_closure(generators, backend=backend)
    dims = 2**nqubits

    assert basis.shape == (nqubits * (2 * nqubits - 1), dims, dims)

    # basis is Hermitian and orthonormal under the Hilbert-Schmidt inner product
    flat = backend.reshape(basis, (basis.shape[0], -1))
    gram = backend.conj(flat) @ backend.transpose(flat)
    backend.assert_allclose(gram, backend.matrices.I(basis.shape[0]), atol=1e-8)
    for element in basis:
        backend.assert_allclose(element, backend.conj(backend.transpose(element)))

    # basis spans the generators and is closed under commutators,
    # hence adding them (and their commutators) does not enlarge the algebra
    extended = list(basis)
    extended.extend(generators)
    extended.extend(commutator(basis[0], element) for element in basis)
    assert lie_closure(extended, backend=backend).shape == basis.shape


def test_lie_closure_max_iterations(backend):
    I, X, Z = backend.matrices.I(2), backend.matrices.X, backend.matrices.Z
    generators = [backend.kron(X, X), backend.kron(Z, I), backend.kron(I, Z)]

    # one extra nesting level at a time: 3 generators, then 5, then 6
    for max_iterations, dim in zip((0, 1, 2, 3), (3, 5, 6, 6)):
        basis = lie_closure(generators, max_iterations=max_iterations, backend=backend)
        assert basis.shape[0] == dim


def test_lie_closure_pauli_errors(backend):
    X = backend.matrices.X

    with pytest.raises(TypeError):
        lie_closure(["XX", X], backend=backend)
    with pytest.raises(ValueError):
        lie_closure(["XX", "Z"], backend=backend)
    with pytest.raises(ValueError):
        lie_closure(["XA"], backend=backend)


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_lie_closure_pauli(backend, nqubits):
    # transverse-field Ising chain (with a repeated generator), whose DLA is so(2 * nqubits)
    generators = ["I" * q + "Z" + "I" * (nqubits - q - 1) for q in range(nqubits)]
    generators.extend(
        "I" * q + "XX" + "I" * (nqubits - q - 2) for q in range(nqubits - 1)
    )
    generators.append(generators[0])

    paulis = lie_closure(generators, backend=backend)

    assert len(paulis) == nqubits * (2 * nqubits - 1)
    assert len(set(paulis)) == len(paulis)
    assert paulis[: len(generators) - 1] == generators[:-1]

    # same algebra as the one obtained from matrices
    matrices = {
        "I": backend.matrices.I(2),
        "X": backend.matrices.X,
        "Y": backend.matrices.Y,
        "Z": backend.matrices.Z,
    }
    to_matrix = lambda pauli: reduce(backend.kron, [matrices[p] for p in pauli])
    basis = lie_closure([to_matrix(gen) for gen in generators], backend=backend)
    assert basis.shape[0] == len(paulis)

    extended = list(basis)
    extended.extend(to_matrix(pauli) for pauli in paulis)
    assert lie_closure(extended, backend=backend).shape == basis.shape


def test_lie_closure_pauli_max_iterations(backend):
    generators = ["XX", "ZI", "IZ"]

    for max_iterations, dim in zip((0, 1, 2, 3), (3, 5, 6, 6)):
        paulis = lie_closure(generators, max_iterations=max_iterations, backend=backend)
        assert len(paulis) == dim

    # commuting Pauli strings generate an abelian algebra
    assert lie_closure(["ZI", "IZ", "ZZ"], backend=backend) == ["ZI", "IZ", "ZZ"]
    # the identity commutes with everything
    assert lie_closure(["II", "XX"], backend=backend) == ["II", "XX"]


def test_lie_closure_pauli_sum_errors(backend):
    with pytest.raises(ValueError):
        lie_closure([{"XX": 1.0j}], backend=backend)
    with pytest.raises(ValueError):
        lie_closure([{"XX": 1.0, "Z": 1.0}], backend=backend)
    with pytest.raises(ValueError):
        lie_closure([{}], backend=backend)
    with pytest.raises(TypeError):
        lie_closure([{"XX": 1.0}, backend.matrices.X], backend=backend)


@pytest.mark.parametrize(
    "generators",
    [
        [{"XXI": 1.0, "YYI": 1.0, "ZZI": 1.0}, {"IXX": 1.0, "IYY": 1.0, "IZZ": 1.0}],
        [{"XXI": 1.0, "YYI": 1.0, "ZZI": 0.7}, {"IXX": 1.0, "IYY": 1.0, "IZZ": 0.7}],
        [{"ZII": 1.0, "IZI": 1.0, "IIZ": 1.0}, {"XXI": 1.0, "IXX": 1.0}],
        [{"XI": 0.5, "ZZ": -1.0}, "IY", "ZX"],
    ],
)
def test_lie_closure_pauli_sum(backend, generators):
    matrices = {
        "I": backend.matrices.I(2),
        "X": backend.matrices.X,
        "Y": backend.matrices.Y,
        "Z": backend.matrices.Z,
    }
    to_matrix = lambda term: sum(
        coeff * reduce(backend.kron, [matrices[p] for p in pauli])
        for pauli, coeff in ({term: 1.0} if isinstance(term, str) else term).items()
    )

    operators = lie_closure(generators, backend=backend)
    basis = lie_closure([to_matrix(gen) for gen in generators], backend=backend)

    assert len(operators) == basis.shape[0]

    # orthonormal in the coefficient space
    paulis = sorted({pauli for operator in operators for pauli in operator})
    coefficients = np.array(
        [[operator.get(pauli, 0.0) for pauli in paulis] for operator in operators]
    )
    np.testing.assert_allclose(
        coefficients @ coefficients.T, np.eye(len(operators)), atol=1e-8
    )

    # same algebra as the one obtained from matrices
    extended = list(basis)
    extended.extend(to_matrix(operator) for operator in operators)
    assert lie_closure(extended, backend=backend).shape == basis.shape


def test_lie_closure_pauli_sum_max_iterations(backend):
    generators = [{"XX": 1.0, "ZI": 1.0}, {"IZ": 1.0}]

    # one extra nesting level at a time, until the algebra is closed
    dims = [
        len(lie_closure(generators, max_iterations=max_iterations, backend=backend))
        for max_iterations in range(5)
    ]
    assert dims[0] == 2
    assert dims == sorted(dims)
    assert dims[-1] == dims[-2] == len(lie_closure(generators, backend=backend))

    # a single Pauli string is the same as a dictionary with unit coefficient
    operators = lie_closure(["XX", {"ZI": 1.0}, {"IZ": 1.0}], backend=backend)
    assert len(operators) == 6


# Dynamical Lie algebras of translation-invariant 2-local spin chains, classified in
# R. Wiersema, E. Kokcu, A. F. Kemper and B. N. Bakalov, npj Quantum Inf. 10, 110 (2024).
# Generating sets of Table I, on pairs of neighbouring qubits.
CLASSIFICATION = {
    "a0": ["XX"],
    "a1": ["XY"],
    "a2": ["XY", "YX"],
    "a3": ["XX", "YZ"],
    "a4": ["XX", "YY"],
    "a5": ["XY", "YZ"],
    "a6": ["XX", "YZ", "ZY"],
    "a7": ["XX", "YY", "ZZ"],
    "a8": ["XX", "XZ"],
    "a9": ["XY", "XZ"],
    "a10": ["XY", "YZ", "ZX"],
    "a11": ["XY", "YX", "YZ"],
    "a12": ["XX", "XY", "YZ"],
    "a13": ["XX", "YY", "YZ"],
    "a14": ["XX", "YY", "XY"],
    "a15": ["XX", "XY", "XZ"],
    "a16": ["XY", "YX", "YZ", "ZY"],
    "a17": ["XX", "XY", "ZX"],
    "a18": ["XX", "XZ", "YY", "ZY"],
    "a19": ["XX", "XY", "ZX", "YZ"],
    "a20": ["XX", "YY", "ZZ", "ZY"],
    "a21": ["XX", "YY", "XY", "ZX"],
    "a22": ["XX", "XY", "XZ", "YX"],
    "b0": ["XI", "IX"],
    "b1": ["XX", "XI", "IX"],
    "b2": ["XY", "XI", "IX"],
    "b3": ["XI", "YI", "IX", "IY"],
    "b4": ["XX", "XY", "XZ", "XI", "IX", "IY", "IZ"],
}

# Bases for three qubits, supplement B V of the same reference
CLASSIFICATION_BASES = {
    "open": {
        "a0": "IXX XXI",
        "a1": "IXY XYI XZY",
        "a2": "IXY IYX XYI XZY YXI YZX",
        "a3": "IXX IYZ XXI XZZ YIY YXZ YYX YZI ZIZ ZXY",
        "a4": "IXX IYY XXI XZY YYI YZX",
        "a5": "IXY IYZ XYI XZY YIX YXZ YYY YZI ZIY ZYX",
        "a7": "IXX IYY IZZ XIX XXI XYZ XZY YIY YXZ YYI YZX ZIZ ZXY ZYX ZZI",
        "a8": "IIY IXX IXZ IYI IZX IZZ XXI XYX XYZ XZI",
        "a9": "IIX IXI IXY IXZ XYI XYY XYZ XZI XZY XZZ",
        "a10": "IXY IYZ IZX XIZ XXX XYI XZY YIX YXZ YYY YZI ZIY ZXI ZYX ZZZ",
        "a14": "IIZ IXX IXY IYX IYY IZI XXI XYI XZX XZY YXI YYI YZX YZY ZII",
        "a15": "IIX IIY IIZ IXI IXX IXY IXZ IYI IYX IYY IYZ IZI IZX IZY IZZ XIX XIY XIZ XXI XXX XXY XXZ XYI XYX XYY XYZ XZI XZX XZY XZZ",
    },
    "periodic": {
        "a0": "IXX XIX XXI",
        "a1": "IXY XYI XZY YIX YXZ ZYX",
        "a2": "IXY IYX XIY XYI XYZ XZY YIX YXI YXZ YZX ZXY ZYX",
        "a3": "IIX IXI IXX IYY IYZ IZY IZZ XII XIX XXI XYY XYZ XZY XZZ YIY YIZ YXY YXZ YYI YYX YZI YZX ZIY ZIZ ZXY ZXZ ZYI ZYX ZZI ZZX",
        "a4": "IXX IYY IZZ XIX XXI XYZ XZY YIY YXZ YYI YZX ZIZ ZXY ZYX ZZI",
        "a8": "IIY IXX IXZ IYI IYY IZX IZZ XIX XIZ XXI XXY XYX XYZ XZI XZY YII YIY YXX YXZ YYI YZX YZZ ZIX ZIZ ZXI ZXY ZYX ZYZ ZZI ZZY",
        "a9": "IIX IXI IXY IXZ XII XYI XYY XYZ XZI XZY XZZ YIX YXY YXZ YYX YZX ZIX ZXY ZXZ ZYX ZZX",
        "a14": "IIZ IXX IXY IYX IYY IZI IZZ XIX XIY XXI XXZ XYI XYZ XZX XZY YIX YIY YXI YXZ YYI YYZ YZX YZY ZII ZIZ ZXX ZXY ZYX ZYY ZZI",
    },
}


@pytest.mark.parametrize("label", list(CLASSIFICATION))
@pytest.mark.parametrize("nqubits", [3, 4, 5])
@pytest.mark.parametrize("topology", ["open", "periodic", "permutation"])
def test_lie_closure_classification(backend, topology, nqubits, label):
    dim = _classification_dimension(label, nqubits, topology)
    if dim is None:
        pytest.skip("Case not covered by the classification.")

    generators = _classification_generators(label, nqubits, topology)
    assert len(lie_closure(generators, backend=backend)) == dim


@pytest.mark.parametrize("topology", ["open", "periodic"])
def test_lie_closure_classification_bases(backend, topology):
    for label, basis in CLASSIFICATION_BASES[topology].items():
        generators = _classification_generators(label, 3, topology)
        assert sorted(lie_closure(generators, backend=backend)) == sorted(basis.split())


@pytest.mark.parametrize("nqubits", [3, 4, 5, 6])
def test_lie_closure_classification_closed_forms(backend, nqubits):
    def string(i, j, first, last):
        paulis = ["I"] * nqubits
        paulis[i : j + 1] = [first] + ["Z"] * (j - i - 1) + [last]
        return "".join(paulis)

    pairs = [(i, j) for i in range(nqubits) for j in range(i + 1, nqubits)]
    a_0 = ["I" * j + "XX" + "I" * (nqubits - j - 2) for j in range(nqubits - 1)]
    a_1 = [string(i, j, "X", "Y") for i, j in pairs]
    a_2 = a_1 + [string(i, j, "Y", "X") for i, j in pairs]

    for label, basis in (("a0", a_0), ("a1", a_1), ("a2", a_2)):
        generators = _classification_generators(label, nqubits, "open")
        assert sorted(lie_closure(generators, backend=backend)) == sorted(basis)


@pytest.mark.parametrize("label", list(CLASSIFICATION))
@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("topology", ["open", "periodic", "permutation"])
def test_lie_closure_classification_sums(backend, topology, nqubits, label):
    dim = _classification_dimension(label, nqubits, topology)
    if dim is None:
        pytest.skip("Case not covered by the classification.")

    # the algebra only depends on the real span of the generators, which is
    # unchanged by an invertible real recombination of them (here, a rotation)
    strings = _classification_generators(label, nqubits, topology)
    random_matrix = np.random.default_rng(1234).normal(size=(len(strings),) * 2)
    mixing, _ = np.linalg.qr(random_matrix)
    generators = [
        {string: float(row[j]) for j, string in enumerate(strings)} for row in mixing
    ]

    assert len(lie_closure(generators, backend=backend)) == dim


@pytest.mark.parametrize("label", list(CLASSIFICATION))
def test_lie_closure_classification_matrices(backend, label):
    nqubits = 3
    dim = _classification_dimension(label, nqubits, "open")
    if dim is None:
        pytest.skip("Case not covered by the classification.")

    paulis = {
        "I": backend.matrices.I(2),
        "X": backend.matrices.X,
        "Y": backend.matrices.Y,
        "Z": backend.matrices.Z,
    }
    generators = [
        reduce(backend.kron, [paulis[pauli] for pauli in string])
        for string in _classification_generators(label, nqubits, "open")
    ]

    assert lie_closure(generators, backend=backend).shape[0] == dim


@pytest.mark.parametrize("tol", [1e-10, 1e-8, 1e-6])
def test_lie_closure_pauli_sum_tolerance(backend, tol):
    # the result must not depend on ``tol`` as long as it separates numerical noise
    # from genuinely new operators
    strings = _classification_generators("a11", 4, "open")
    random_matrix = np.random.default_rng(11).normal(size=(len(strings),) * 2)
    mixing, _ = np.linalg.qr(random_matrix)
    generators = [
        {string: float(row[j]) for j, string in enumerate(strings)} for row in mixing
    ]

    assert len(lie_closure(generators, tol=tol, backend=backend)) == 120


def _classification_dimension(label: str, nqubits: int, topology: str):
    """Dimension of the algebras of Theorems IV.1, IV.2 and IV.3 of the reference,
    ``None`` if the case is not covered."""
    n = nqubits
    index = int(label[1:])
    su = lambda dims: dims**2 - 1
    so = lambda dims: dims * (dims - 1) // 2
    sp = lambda dims: dims * (2 * dims + 1)

    if label[0] == "b":
        dims = {
            "open": [
                n,
                2 * n - 1,
                sp(2 ** (n - 2)) + 1,
                3 * n,
                2 * su(2 ** (n - 1)) + 1,
            ],
            "periodic": [n, 2 * n, so(2**n) if n >= 4 else None, 3 * n, su(2**n)],
            "permutation": [n, n * (n + 1) // 2, None, 3 * n, None],
        }
        return dims[topology][index]

    # classes of n modulo 8 (modulo 6) for a_3 (a_5), labelled by min(n % 8, 8 - n % 8)
    a_3 = {
        0: 4 * so(2 ** (n - 2)),
        1: so(2 ** (n - 1)),
        2: 2 * su(2 ** (n - 2)),
        3: sp(2 ** (n - 2)),
        4: 4 * sp(2 ** (n - 3)),
    }[min(n % 8, 8 - n % 8)]
    a_5 = {
        0: 4 * so(2 ** (n - 2)),
        1: so(2 ** (n - 1)),
        2: 2 * su(2 ** (n - 2)),
        3: sp(2 ** (n - 2)),
    }[min(n % 6, 6 - n % 6)]
    a_7 = su(2 ** (n - 1)) if n % 2 else 4 * su(2 ** (n - 2))
    big = n >= 4
    open_dims = {
        0: n - 1,
        1: so(n),
        2: 2 * so(n),
        3: a_3,
        4: 2 * so(n),
        5: a_5,
        6: a_7,
        7: a_7,
        8: (n - 1) * (2 * n - 1),
        9: sp(2 ** (n - 2)),
        10: a_7,
        11: so(2**n) if big else None,
        13: 2 * su(2 ** (n - 1)),
        14: so(2 * n),
        15: 2 * su(2 ** (n - 1)),
        16: so(2**n) if big else None,
        20: 2 * su(2 ** (n - 1)),
    }
    open_dims.update({k: su(2**n) if big else None for k in (12, 17, 18, 19, 21, 22)})

    if topology == "open":
        return open_dims[index]

    if topology == "periodic":
        if index in (7, 13, 16, 20):
            return open_dims[index]
        if index in (12, 15, 17, 18, 19, 21, 22):
            return su(2**n)
        periodic_3 = {0: 4 * so(2 ** (n - 2)), 4: 4 * sp(2 ** (n - 3))}
        periodic_dims = {
            0: n,
            1: 2 * so(n),
            2: 4 * so(n),
            3: (
                2 * su(2 ** (n - 1))
                if n % 2
                else periodic_3.get(n % 8, 4 * su(2 ** (n - 2)))
            ),
            4: so(2 * n) if n % 2 else 4 * so(n),
            5: (
                so(2**n)
                if n % 3
                else (4 * so(2 ** (n - 2)) if n % 6 == 0 else sp(2 ** (n - 2)))
            ),
            6: 2 * su(2 ** (n - 1)) if n % 2 else 4 * su(2 ** (n - 2)),
            8: 2 * so(2 * n),
            9: so(2**n) if big else None,
            10: (
                su(2**n)
                if n % 3
                else (4 * su(2 ** (n - 2)) if n % 6 == 0 else su(2 ** (n - 1)))
            ),
            11: so(2**n) if big else None,
            14: 2 * so(2 * n),
        }
        return periodic_dims[index]

    permutation_dims = {
        0: n * (n - 1) // 2,
        2: 2 * so(2 ** (n - 1)),
        4: a_7,
        6: 2 * su(2 ** (n - 1)),
        7: a_7,
        14: 2 * su(2 ** (n - 1)),
        16: so(2**n) if big else None,
        20: 2 * su(2 ** (n - 1)),
        22: su(2**n) if big else None,
    }
    return permutation_dims.get(index)


def _classification_generators(label: str, nqubits: int, topology: str) -> list[str]:
    """Generators of the spin chain: on neighbouring qubits for ``open`` and
    ``periodic`` topologies, and on all pairs of qubits for ``permutation``."""
    if topology == "permutation":
        pairs = [(i, j) for i in range(nqubits) for j in range(nqubits) if i != j]
    else:
        pairs = [(i, i + 1) for i in range(nqubits - 1)]
        if topology == "periodic":
            pairs.append((nqubits - 1, 0))

    generators = []
    for pair in CLASSIFICATION[label]:
        for first, second in pairs:
            paulis = ["I"] * nqubits
            paulis[first], paulis[second] = pair
            generators.append("".join(paulis))

    return list(dict.fromkeys(generators))
