"""Test methods defined in `qibo/models/circuit.py`."""
import numpy as np
import pytest
import qibo
from qibo import gates, models
from qibo.tests_new.utils import random_state


def test_circuit_constructor():
    from qibo.core.circuit import Circuit, DensityMatrixCircuit
    from qibo.core.distcircuit import DistributedCircuit
    c = models.Circuit(5)
    assert isinstance(c, Circuit)
    c = models.Circuit(5, density_matrix=True)
    assert isinstance(c, DensityMatrixCircuit)
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
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.QFT(nqubits)
    assert c.nqubits == nqubits
    assert c.depth == 2 * nqubits
    assert c.ngates == nqubits ** 2 // 2 + nqubits
    qibo.set_backend(original_backend)


@pytest.mark.parametrize("nqubits", [5, 6, 12])
@pytest.mark.parametrize("random", [False, True])
def test_qft_execution(backend, accelerators, nqubits, random):
    original_backend = qibo.get_backend()
    qibo.set_backend(backend)
    c = models.QFT(nqubits)
    if random:
        initial_state = random_state(nqubits)
        final_state = c(np.copy(initial_state))
    else:
        initial_state = c.get_initial_state()
        final_state = c()
    target_state = exact_qft(initial_state)
    np.testing.assert_allclose(final_state, target_state)
    qibo.set_backend(original_backend)


def test_qft_errors():
    """Check that ``_DistributedQFT`` raises error if not sufficient qubits."""
    from qibo.models.circuit import _DistributedQFT
    with pytest.raises(NotImplementedError):
        c = _DistributedQFT(2, accelerators={"/GPU:0": 4})
    with pytest.raises(NotImplementedError):
        c = models.QFT(10, with_swaps=False, accelerators={"/GPU:0": 2})
