"""
Testing Quantum Fourier Transform (QFT) circuit.
"""
import numpy as np
import pytest
import qibo
from qibo import gates, models
from qibo.tests import utils

_atol = 1e-7


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
    return np.exp(sign * 2 * np.pi * 1j * exponent / dimension)


def exact_qft(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Performs exact QFT to a given state vector."""
    dim = len(x)
    return qft_matrix(dim, inverse).dot(x) / np.sqrt(dim)


@pytest.mark.parametrize("nqubits", [4, 10, 100])
def test_qft_circuit_size(backend, nqubits):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.QFT(nqubits)
    assert c.nqubits == nqubits
    assert c.depth == 2 * nqubits
    assert c.ngates == nqubits ** 2 // 2 + nqubits
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [4, 5])
def test_qft_transformation(nqubits):
    """Check QFT transformation for |00...0>."""
    c = models.QFT(nqubits)
    final_state = c.execute().numpy()

    initial_state = np.zeros_like(final_state)
    initial_state[0] = 1.0
    exact_state = exact_qft(initial_state)

    np.testing.assert_allclose(final_state, exact_state, atol=_atol)


@pytest.mark.parametrize("nqubits", [4, 5, 11, 12])
def test_qft_transformation_random(nqubits):
    """Check QFT transformation for random initial state."""
    initial_state = utils.random_numpy_state(nqubits)
    exact_state = exact_qft(initial_state)

    c_init = models.Circuit(nqubits)
    c_init.add(gates.Flatten(initial_state))
    c = c_init + models.QFT(nqubits)
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, exact_state, atol=_atol)


@pytest.mark.parametrize("nqubits", [4, 5, 11, 12])
def test_distributed_qft_agreement(nqubits):
    """Check ``_DistributedQFT`` agrees with normal ``QFT``."""
    from qibo.models.circuit import _DistributedQFT
    initial_state = utils.random_numpy_state(nqubits)
    exact_state = exact_qft(initial_state)

    c = _DistributedQFT(nqubits)
    final_state = c(np.copy(initial_state)).numpy()

    np.testing.assert_allclose(final_state, exact_state, atol=_atol)


def test_distributed_qft_error():
    """Check that ``_DistributedQFT`` raises error if not sufficient qubits."""
    from qibo.models.circuit import _DistributedQFT
    with pytest.raises(NotImplementedError):
        c = _DistributedQFT(2, accelerators={"/GPU:0": 4})
    with pytest.raises(NotImplementedError):
        c = models.QFT(10, with_swaps=False, accelerators={"/GPU:0": 2})
