"""Test methods defined in `qibo/models/circuit.py`."""

import numpy as np
import pytest

from qibo.models import QFT
from qibo.quantum_info import random_statevector


def qft_matrix(dimension: int, inverse: bool = False) -> np.ndarray:
    """Creates exact QFT matrix.

    Args:
        dimension: Dimension d of the matrix. The matrix will be d x d.
        inverse: Whether to construct matrix for the inverse QFT.

    Return:
        QFT transformation matrix as a numpy array with shape (d, d).
    """
    exponent = np.outer(np.arange(dimension), np.arange(dimension))
    sign = 1 - 2 * int(inverse)
    return np.exp(sign * 2 * np.pi * 1j * exponent / dimension) / np.sqrt(dimension)


def exact_qft(
    x: np.ndarray, density_matrix: bool = False, backend=None, inverse: bool = False
) -> np.ndarray:
    """Performs exact QFT to a given state vector."""
    dim = len(x)
    matrix = qft_matrix(dim, inverse)
    if backend is not None:
        matrix = backend.cast(matrix, dtype=matrix.dtype)
    if density_matrix:
        return matrix @ x @ backend.conj(matrix).T
    return matrix @ x


@pytest.mark.parametrize("nqubits", [4, 10, 100])
def test_qft_circuit_size(nqubits):
    circuit = QFT(nqubits)
    assert circuit.nqubits == nqubits
    assert circuit.depth == 2 * nqubits
    assert circuit.ngates == nqubits**2 // 2 + nqubits


@pytest.mark.parametrize("nqubits", [4, 5])
def test_qft_matrix(backend, nqubits):
    circuit = QFT(nqubits)
    dim = 2**nqubits
    target_matrix = qft_matrix(dim)
    backend.assert_allclose(
        circuit.unitary(backend), target_matrix, atol=1e-6, rtol=1e-6
    )
    circuit = circuit.invert()
    target_matrix = qft_matrix(dim, inverse=True)
    backend.assert_allclose(
        circuit.unitary(backend), target_matrix, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nqubits", [5, 6])
@pytest.mark.parametrize("random", [False, True])
def test_qft_execution(backend, nqubits, random, density_matrix):
    circuit = QFT(nqubits, density_matrix=density_matrix)
    initial_state = (
        random_statevector(2**nqubits, backend=backend)
        if random
        else backend.zero_state(nqubits)
    )
    if density_matrix:
        initial_state = backend.outer(initial_state, backend.conj(initial_state))

    final_state = backend.execute_circuit(c, backend.copy(initial_state))._state
    target_state = exact_qft(initial_state, density_matrix, backend)
    backend.assert_allclose(final_state, target_state, atol=1e-6, rtol=1e-6)


def test_qft_errors():
    """Check that ``_DistributedQFT`` raises error if not sufficient qubits."""
    from qibo.models.qft import _DistributedQFT

    with pytest.raises(NotImplementedError):
        circuit = QFT(10, with_swaps=False, accelerators={"/GPU:0": 2})
    with pytest.raises(NotImplementedError):
        circuit = _DistributedQFT(2, accelerators={"/GPU:0": 4})
