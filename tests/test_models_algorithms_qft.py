"""Test methods defined in `qibo/models/circuit.py`."""

import math

import numpy as np
import pytest

from qibo import gates
from qibo.models import QFT, Circuit
from qibo.models.algorithms.qft import _DistributedQFT
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

    final_state = backend.execute_circuit(circuit, backend.copy(initial_state))._state
    target_state = exact_qft(initial_state, density_matrix, backend)
    backend.assert_allclose(final_state, target_state, atol=1e-6, rtol=1e-6)


def test_qft_errors():
    """Check that ``_DistributedQFT`` raises error if not sufficient qubits."""
    with pytest.raises(NotImplementedError):
        QFT(10, with_swaps=False, accelerators={"/GPU:0": 2})
    with pytest.raises(NotImplementedError):
        _DistributedQFT(2, accelerators={"/GPU:0": 4})


@pytest.mark.parametrize("with_swaps", [True, False])
@pytest.mark.parametrize("nqubits", [3, 4])
def test_approximate_qft_exact_equivalence(backend, nqubits, with_swaps):
    target = QFT(nqubits, with_swaps).unitary(backend)
    for degree in [None, nqubits, nqubits + 1, 3 * nqubits]:
        circuit = QFT(nqubits, with_swaps, approximation_degree=degree)
        backend.assert_allclose(circuit.unitary(backend), target, atol=1e-7)


@pytest.mark.parametrize("with_swaps", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4])
def test_approximate_qft_hadamards(backend, nqubits, with_swaps):
    circuit = QFT(nqubits, with_swaps, approximation_degree=1)
    target = backend.matrices.H
    expected = target
    for _ in range(nqubits - 1):
        expected = backend.kron(expected, target)
    if with_swaps:
        # the swaps commute with the Hadamards and reverse the order of the qubits
        reversal = Circuit(nqubits)
        for qubit in range(nqubits // 2):
            reversal.add(gates.SWAP(qubit, nqubits - qubit - 1))
        expected = expected @ reversal.unitary(backend)
    backend.assert_allclose(circuit.unitary(backend), expected, atol=1e-7)
    assert not [gate for gate in circuit.queue if isinstance(gate, gates.CU1)]


@pytest.mark.parametrize("nqubits", [1, 2, 3, 5, 8])
def test_approximate_qft_gate_count(nqubits):
    for degree in range(1, nqubits + 3):
        circuit = QFT(nqubits, approximation_degree=degree)
        ncu1 = len([gate for gate in circuit.queue if isinstance(gate, gates.CU1)])
        expected = sum(min(degree - 1, nqubits - 1 - i) for i in range(nqubits))
        assert ncu1 == expected
        if degree <= nqubits:
            assert ncu1 == nqubits * (degree - 1) - degree * (degree - 1) // 2
        assert circuit.ngates == nqubits + ncu1 + nqubits // 2
        for gate in circuit.queue:
            if isinstance(gate, gates.CU1):
                assert (
                    gate.target_qubits[0]
                    < gate.control_qubits[0]
                    < (gate.target_qubits[0] + degree)
                )


@pytest.mark.parametrize("nqubits", [5, 6])
def test_approximate_qft_error_bounds(backend, nqubits):
    exact = QFT(nqubits).unitary(backend)
    for degree in range(1, nqubits + 1):
        approximate = QFT(nqubits, approximation_degree=degree).unitary(backend)

        bound = 2 * math.pi * nqubits * 2.0 ** (-degree)

        # unitarity
        backend.assert_allclose(
            approximate @ backend.dagger(approximate),
            backend.identity(2**nqubits),
            atol=1e-8,
        )
        # operator norm
        backend.assert_allclose(
            backend.matrix_norm(approximate - exact, ord=2) <= bound + 1e-8, True
        )
        # elementwise phases
        mask = backend.abs(exact) > 1e-12
        phases = backend.angle(approximate[mask] / exact[mask])
        backend.assert_allclose(backend.max(np.abs(phases)) <= bound + 1e-8, True)
        backend.assert_allclose(backend.abs(approximate), backend.abs(exact), atol=1e-8)


@pytest.mark.parametrize("degree", [0, -1])
def test_approximate_qft_value_error():
    for degree in [0, -1]:
        with pytest.raises(ValueError):
            QFT(4, approximation_degree=degree)

    for degree in [2.5, True, "2"]:
        with pytest.raises(TypeError):
            QFT(4, approximation_degree=degree)


@pytest.mark.parametrize("degree", [1, 2, 3, 5, None])
@pytest.mark.parametrize("nqubits", [4, 5])
def test_approximate_distributed_qft(nqubits, degree):
    """Gate lists of the distributed QFT, which do not need GPUs, are filtered."""
    circuit = _DistributedQFT(nqubits, approximation_degree=degree)
    degree = nqubits if degree is None else degree
    ncu1 = len([gate for gate in circuit.queue if isinstance(gate, gates.CU1)])
    assert ncu1 == sum(min(degree - 1, nqubits - 1 - i) for i in range(nqubits))
    exact = _DistributedQFT(nqubits)
    assert exact.ngates >= circuit.ngates
    if degree >= nqubits:
        assert [g.name for g in circuit.queue] == [g.name for g in exact.queue]
