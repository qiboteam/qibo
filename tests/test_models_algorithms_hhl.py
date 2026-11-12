"""Tests for the HHL algorithm in `qibo/models/algorithms/hhl.py`."""

import pytest

from qibo import Circuit, gates, hamiltonians
from qibo.hamiltonians import Hamiltonian
from qibo.models.algorithms import HHL
from qibo.models.encodings import binary_encoder
from qibo.quantum_info import random_statevector, random_unitary

SPECTRA = {
    "positive": [1.0, 0.5, 0.8, 0.6],
    "indefinite": [-1.0, 0.5, 0.8, -0.6],
    "singular": [1.0, 0.0, 0.5, 0.05],
}


def _matrix(values, dimension, backend):
    unitary = random_unitary(dimension, seed=1, backend=backend)
    values = backend.cast(values[:dimension], dtype=backend.complex128)

    return backend.matmul(unitary * values, backend.conj(backend.transpose(unitary)))


@pytest.mark.parametrize("kind", ["positive", "indefinite", "singular"])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_hhl(backend, nqubits, kind):
    """The solution is in the input register if the flag is one and the clock is zero."""
    dimension, nclock = 2**nqubits, 8
    matrix = _matrix(SPECTRA[kind], dimension, backend)
    vector = random_statevector(dimension, seed=2, backend=backend)

    # the eigenvalues that are smaller than the constant are not inverted
    circuit, constant = HHL(
        matrix,
        vector,
        nclock,
        condition_number=2.0 if kind == "singular" else None,
        backend=backend,
    )

    eigenvalues, eigenvectors = backend.eigh(matrix)
    coefficients = backend.matmul(backend.conj(backend.transpose(eigenvectors)), vector)
    inverted = backend.abs(eigenvalues) >= constant * (1 - 1e-9)
    inverse = backend.where(
        inverted, 1 / backend.where(inverted, eigenvalues, 1.0), 0.0
    )
    exact = backend.matmul(eigenvectors, coefficients * inverse)

    state = backend.execute_circuit(circuit).state()
    state = backend.reshape(state, (2, 2**nclock, dimension))
    solution = state[1, 0]

    fidelity = backend.abs(backend.sum(backend.conj(exact) * solution)) ** 2 / (
        backend.sum(backend.abs(exact) ** 2) * backend.sum(backend.abs(solution) ** 2)
    )
    # the norm of the solution is in the probability of the flag
    probability = backend.sum(backend.abs(state[1]) ** 2)
    norm = constant**2 * backend.sum(backend.abs(exact) ** 2)

    assert circuit.nqubits == 1 + nclock + nqubits
    assert fidelity > 0.99
    backend.assert_allclose(probability, norm, rtol=0.1)


def test_hhl_inputs(backend):
    """The matrix can be a Hamiltonian, and the vector can be a circuit."""
    hamiltonian = hamiltonians.TFIM(2, h=1.5, dense=False, backend=backend)
    vector = random_statevector(4, seed=2, backend=backend)
    loading = binary_encoder(2, data=vector, backend=backend)

    target, _ = HHL(hamiltonian.matrix, vector, 5, backend=backend)
    target = backend.execute_circuit(target).state()

    for matrix, state in (
        (Hamiltonian(2, hamiltonian.matrix, backend=backend), vector),
        (hamiltonian, vector),
        (hamiltonian.matrix, loading),
    ):
        circuit, _ = HHL(matrix, state, 5, backend=backend)

        backend.assert_allclose(
            backend.execute_circuit(circuit).state(), target, atol=1e-10
        )


def test_hhl_density_matrix(backend):
    """Keyword arguments are passed to the circuit."""
    matrix = _matrix(SPECTRA["positive"], 2, backend)
    circuit, _ = HHL(matrix, [1.0, 0.0], 3, density_matrix=True, backend=backend)

    assert circuit.density_matrix


def test_hhl_errors(backend):
    matrix = _matrix(SPECTRA["positive"], 2, backend)
    vector = [1.0, 0.0]
    measured = Circuit(1)
    measured.add(gates.M(0))

    for nclock in (1, 0, 2.5, True):
        with pytest.raises(ValueError):
            HHL(matrix, vector, nclock, backend=backend)
    for shape in ([1.0, 0.5], [[1.0]], [[1.0, 0.0, 0.0]], [[1.0] * 3] * 3):
        with pytest.raises(ValueError):
            HHL(shape, vector, 3, backend=backend)
    with pytest.raises(ValueError):
        HHL([[1.0, 0.5], [0.0, 1.0]], vector, 3, backend=backend)
    for wrong in ([1.0, 0.0, 0.0], [0.0, 0.0], [[1.0, 0.0]]):
        with pytest.raises(ValueError):
            HHL(matrix, wrong, 3, backend=backend)
    with pytest.raises(ValueError):
        HHL(matrix, Circuit(2), 3, backend=backend)
    with pytest.raises(ValueError):
        HHL(matrix, measured, 3, backend=backend)
    with pytest.raises(ValueError):
        HHL(_matrix(SPECTRA["singular"], 2, backend), vector, 3, backend=backend)
    for condition_number in (0.5, -1.0):
        with pytest.raises(ValueError):
            HHL(matrix, vector, 3, condition_number, backend=backend)
