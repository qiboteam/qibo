"""
Testing Quantum Fourier Transform (QFT) circuit.
"""
import numpy as np
from qibo import gates, models


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


def test_qft_sanity():
    c = models.QFTCircuit(4)
    assert c.size == 4
    assert c.depth == 12


def test_qft_transformation():
    pass