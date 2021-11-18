"""Test methods defined in `qibo/models/circuit.py`."""
import sys
import numpy as np
import pytest
from qibo import gates, models, K
from qibo.tests.utils import random_state


def test_circuit_constructor():
    from qibo.core.circuit import Circuit, DensityMatrixCircuit
    from qibo.core.distcircuit import DistributedCircuit
    c = models.Circuit(5)
    assert isinstance(c, Circuit)
    c = models.Circuit(5, density_matrix=True)
    assert isinstance(c, DensityMatrixCircuit)
    if not K.supports_multigpu:  # pragma: no cover
        with pytest.raises(NotImplementedError):
            c = models.Circuit(5, accelerators={"/GPU:0": 2})
    else:
        c = models.Circuit(5, accelerators={"/GPU:0": 2})
        assert isinstance(c, DistributedCircuit)
    with pytest.raises(NotImplementedError):
        c = models.Circuit(5, accelerators={"/GPU:0": 2}, density_matrix=True)


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
    c = models.QFT(nqubits)
    assert c.nqubits == nqubits
    assert c.depth == 2 * nqubits
    assert c.ngates == nqubits ** 2 // 2 + nqubits


@pytest.mark.parametrize("nqubits", [5, 6, 12])
@pytest.mark.parametrize("random", [False, True])
def test_qft_execution(backend, accelerators, nqubits, random):
    c = models.QFT(nqubits)
    if random:
        initial_state = random_state(nqubits)
        final_state = c(K.cast(np.copy(initial_state)))
    else:
        initial_state = c.get_initial_state()
        final_state = c()
    target_state = exact_qft(K.to_numpy(initial_state))
    K.assert_allclose(final_state, target_state)


def test_qft_errors():
    """Check that ``_DistributedQFT`` raises error if not sufficient qubits."""
    from qibo.models.circuit import _DistributedQFT
    with pytest.raises(NotImplementedError):
        c = _DistributedQFT(2, accelerators={"/GPU:0": 4})
    with pytest.raises(NotImplementedError):
        c = models.QFT(10, with_swaps=False, accelerators={"/GPU:0": 2})
