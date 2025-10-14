import numpy as np
import pytest
from scipy.linalg import sqrtm

from qibo import Circuit, gates, matrices
from qibo.quantum_info.linalg_operations import (
    _gram_schmidt_process,
    _vector_projection,
    anticommutator,
    commutator,
    lanczos,
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
        test = commutator(matrix_1, matrix_2)
    with pytest.raises(TypeError):
        test = commutator(matrix_2, matrix_1)
    with pytest.raises(TypeError):
        test = commutator(matrix_2, matrix_3)

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
        test = anticommutator(matrix_1, matrix_2)
    with pytest.raises(TypeError):
        test = anticommutator(matrix_2, matrix_1)
    with pytest.raises(TypeError):
        test = anticommutator(matrix_2, matrix_3)

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
        test = partial_trace(state, 1, backend=backend)
    with pytest.raises(ValueError):
        state = (
            random_density_matrix(5, backend=backend)
            if density_matrix
            else random_statevector(5, backend=backend)
        )
        test = partial_trace(state, 1, backend=backend)

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
        test = partial_transpose(state, [0], backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2, 2).astype(complex)
        state += 1j * np.random.rand(2, 2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        test = partial_transpose(state, [1], backend=backend)

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
            test = matrix_power(state, power, backend=backend)
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
    if backend.platform == "pytorch":
        S_sorted, coeffs_sorted = S_sorted[0], coeffs_sorted[0]

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
    recovered = backend.zeros_like(state.shape, dtype=complex)
    for coeff, u, vh in zip(S, U.T, Vh):
        if abs(coeff) > 1e-10:
            recovered = recovered + coeff * backend.kron(u, vh)

    backend.assert_allclose(recovered, state)

    # entropy test
    coeffs = backend.abs(S) ** 2
    entropy = backend.engine.where(backend.abs(S) < 1e-10, 0.0, backend.log(coeffs))
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
