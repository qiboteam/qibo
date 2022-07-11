"""Test methods defined in `qibo/models/circuit.py`."""
import sys
import numpy as np
import pytest
from qibo import gates, models
from qibo.tests.utils import random_state


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


def exact_qft(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Performs exact QFT to a given state vector."""
    dim = len(x)
    return qft_matrix(dim, inverse).dot(x)


@pytest.mark.parametrize("nqubits", [4, 10, 100])
def test_qft_circuit_size(backend, nqubits):
    c = models.QFT(nqubits)
    assert c.nqubits == nqubits
    assert c.depth == 2 * nqubits
    assert c.ngates == nqubits ** 2 // 2 + nqubits


@pytest.mark.parametrize("nqubits", [4, 5])
def test_qft_matrix(backend, nqubits):
    c = models.QFT(nqubits)
    dim = 2 ** nqubits
    target_matrix = qft_matrix(dim)
    backend.assert_allclose(c.unitary(backend), target_matrix)
    c = c.invert()
    target_matrix = qft_matrix(dim, inverse=True)
    backend.assert_allclose(c.unitary(backend), target_matrix)


@pytest.mark.parametrize("nqubits", [5, 6, 12])
@pytest.mark.parametrize("random", [False, True])
def test_qft_execution(backend, accelerators, nqubits, random):
    c = models.QFT(nqubits)
    if random:
        initial_state = random_state(nqubits)
    else:
        initial_state = backend.zero_state(nqubits)
    final_state = backend.execute_circuit(c, np.copy(initial_state))
    target_state = exact_qft(backend.to_numpy(initial_state))
    backend.assert_allclose(final_state, target_state)


def test_qft_errors(backend):
    """Check that ``_DistributedQFT`` raises error if not sufficient qubits."""
    from qibo.models.qft import _DistributedQFT
    with pytest.raises(NotImplementedError):
        c = models.QFT(10, with_swaps=False, accelerators={"/GPU:0": 2})
    with pytest.raises(NotImplementedError):
        c = _DistributedQFT(2, accelerators={"/GPU:0": 4})
