"""Tests for the HHL algorithm in `qibo/models/algorithms/hhl.py`."""

import pytest

from qibo import Circuit, gates, hamiltonians
from qibo.hamiltonians import Hamiltonian
from qibo.models.algorithms import HHL
from qibo.models.algorithms.hhl import _hhl_original, _hhl_qsvt
from qibo.models.encodings import binary_encoder
from qibo.quantum_info import random_statevector, random_unitary

SPECTRA = {
    "positive": [1.0, 0.5, 0.8, 0.6],
    "indefinite": [-1.0, 0.5, 0.8, -0.6],
    "cutoff": [1.0, 0.0, 0.5, 0.05],
    "singular": [1.0, 0.0, 0.5, 0.0],
}
OPTIONS = {"original": {"nclock": 8}, "qsvt": {"method": "qsvt", "epsilon": 0.05}}


def _matrix(values, dimension, backend):
    unitary = random_unitary(dimension, seed=1, backend=backend)
    values = backend.cast(values[:dimension], dtype=backend.complex128)

    return backend.matmul(unitary * values, backend.conj(backend.transpose(unitary)))


@pytest.mark.parametrize(
    "method, kind",
    [
        ("original", "positive"),
        ("original", "indefinite"),
        ("original", "cutoff"),
        ("original", "singular"),
        ("qsvt", "positive"),
        ("qsvt", "indefinite"),
        ("qsvt", "singular"),
    ],
)
@pytest.mark.parametrize("nqubits", [1, 2])
def test_hhl(backend, nqubits, method, kind):
    """The solution is in the input register if the outcome of the other qubits is the right one."""
    dimension = 2**nqubits
    matrix = _matrix(SPECTRA[kind], dimension, backend)
    vector = random_statevector(dimension, seed=2, backend=backend)
    singular = kind in ("cutoff", "singular")

    circuit, constant = HHL(
        matrix,
        vector,
        condition_number=2.0 if singular else None,
        backend=backend,
        **OPTIONS[method],
    )

    # the eigenvalues that are smaller than the constant are not inverted by the original method
    eigenvalues, eigenvectors = backend.eigh(matrix)
    coefficients = backend.matmul(backend.conj(backend.transpose(eigenvectors)), vector)
    cutoff = constant * (1 - 1e-9) if method == "original" else 1e-9
    inverted = backend.abs(eigenvalues) >= cutoff
    inverse = backend.where(
        inverted, 1 / backend.where(inverted, eigenvalues, 1.0), 0.0
    )
    exact = backend.matmul(eigenvectors, coefficients * inverse)

    state = backend.execute_circuit(circuit).state()
    if method == "original":
        # the flag is one, and the clock register is zero
        nauxiliary = 1 + OPTIONS[method]["nclock"]
        state = backend.reshape(state, (2, 2 ** OPTIONS[method]["nclock"], dimension))
        solution, success = state[1, 0], state[1]
    else:
        # the signal qubit and the auxiliary qubit of the block encoding are zero
        nauxiliary = 2
        state = backend.reshape(state, (2, 2, dimension))
        solution, success = state[0, 0], state[0, 0]

    fidelity = backend.abs(backend.sum(backend.conj(exact) * solution)) ** 2 / (
        backend.sum(backend.abs(exact) ** 2) * backend.sum(backend.abs(solution) ** 2)
    )
    # the norm of the solution is in the probability of the outcome
    probability = backend.sum(backend.abs(success) ** 2)
    norm = constant**2 * backend.sum(backend.abs(exact) ** 2)

    assert circuit.nqubits == nauxiliary + nqubits
    # the rotation of the flag is a single multiplexed rotation
    assert sum(gate.name == "multiplexor" for gate in circuit.queue) == int(
        method == "original"
    )
    assert fidelity > 0.99
    backend.assert_allclose(probability, norm, rtol=0.1)


@pytest.mark.parametrize("epsilon", [0.1, 0.45])
def test_hhl_qsvt_unitary(backend, epsilon):
    """A Hermitian unitary is its own inverse, and the polynomial can be the identity."""
    matrix = backend.cast([[0.0, 1.0], [1.0, 0.0]], dtype=backend.complex128)
    vector = random_statevector(2, seed=2, backend=backend)

    circuit, constant = HHL(
        matrix, vector, method="qsvt", epsilon=epsilon, backend=backend
    )

    state = backend.execute_circuit(circuit).state()
    solution = backend.reshape(state, (2, 2, 2))[0, 0]

    # the polynomial is the identity for epsilon = 0.45, whose degree is one
    assert (constant == 1.0) == (epsilon == 0.45)
    backend.assert_allclose(
        solution, constant * backend.matmul(matrix, vector), atol=1e-5
    )


@pytest.mark.parametrize("method", ["original", "qsvt"])
def test_hhl_inputs(backend, method):
    """The matrix can be a Hamiltonian, and the vector can be a circuit."""
    hamiltonian = hamiltonians.TFIM(2, h=0.5, dense=False, backend=backend)
    vector = random_statevector(4, seed=2, backend=backend)
    loading = binary_encoder(2, data=vector, backend=backend)

    target, _ = HHL(hamiltonian.matrix, vector, backend=backend, **OPTIONS[method])
    target = backend.execute_circuit(target).state()

    for matrix, state in (
        (Hamiltonian(2, hamiltonian.matrix, backend=backend), vector),
        (hamiltonian, vector),
        (hamiltonian.matrix, loading),
    ):
        circuit, _ = HHL(matrix, state, backend=backend, **OPTIONS[method])

        backend.assert_allclose(
            backend.execute_circuit(circuit).state(), target, atol=1e-10
        )


def test_hhl_methods(backend):
    """The method chooses the function, and the original one is the default."""
    matrix = _matrix(SPECTRA["positive"], 4, backend)
    vector = random_statevector(4, seed=2, backend=backend)

    original, constant = _hhl_original(matrix, vector, 4, backend=backend)
    qsvt, qsvt_constant = _hhl_qsvt(matrix, vector, 0.1, backend=backend)

    for circuit, target in (
        (HHL(matrix, vector, 4, backend=backend), original),
        (HHL(matrix, vector, 4, method="original", backend=backend), original),
        (HHL(matrix, vector, method="qsvt", epsilon=0.1, backend=backend), qsvt),
    ):
        assert circuit[1] in (constant, qsvt_constant)
        backend.assert_allclose(
            backend.execute_circuit(circuit[0]).state(),
            backend.execute_circuit(target).state(),
            atol=1e-10,
        )


@pytest.mark.parametrize("method", ["original", "qsvt"])
def test_hhl_density_matrix(backend, method):
    """Keyword arguments are passed to the circuit."""
    matrix = _matrix(SPECTRA["positive"], 2, backend)
    circuit, _ = HHL(
        matrix, [1.0, 0.0], density_matrix=True, backend=backend, **OPTIONS[method]
    )

    assert circuit.density_matrix


def test_hhl_errors(backend):
    matrix = _matrix(SPECTRA["positive"], 2, backend)
    vector = [1.0, 0.0]
    measured = Circuit(1)
    measured.add(gates.M(0))

    with pytest.raises(ValueError):
        HHL(matrix, vector, 3, method="other", backend=backend)
    for nclock in (None, 1, 0, 2.5, True):
        with pytest.raises(ValueError):
            HHL(matrix, vector, nclock, backend=backend)
    for epsilon in (0.0, 0.5, 1.0, -0.1):
        with pytest.raises(ValueError):
            HHL(matrix, vector, method="qsvt", epsilon=epsilon, backend=backend)

    # the checks of the matrix, the vector and the condition number are the same
    for options in OPTIONS.values():
        for shape in ([1.0, 0.5], [[1.0]], [[1.0, 0.0, 0.0]], [[1.0] * 3] * 3):
            with pytest.raises(ValueError):
                HHL(shape, vector, backend=backend, **options)
        with pytest.raises(ValueError):
            HHL([[1.0, 0.5], [0.0, 1.0]], vector, backend=backend, **options)
        for wrong in ([1.0, 0.0, 0.0], [0.0, 0.0], [[1.0, 0.0]]):
            with pytest.raises(ValueError):
                HHL(matrix, wrong, backend=backend, **options)
        with pytest.raises(ValueError):
            HHL(matrix, Circuit(2), backend=backend, **options)
        with pytest.raises(ValueError):
            HHL(matrix, measured, backend=backend, **options)
        with pytest.raises(ValueError):
            HHL(
                _matrix(SPECTRA["cutoff"], 2, backend),
                vector,
                backend=backend,
                **options,
            )
        for condition_number in (0.5, -1.0):
            with pytest.raises(ValueError):
                HHL(
                    matrix,
                    vector,
                    condition_number=condition_number,
                    backend=backend,
                    **options,
                )
