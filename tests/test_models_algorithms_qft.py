"""Test methods defined in `qibo/models/circuit.py`."""

import itertools
import math

import numpy as np
import pytest

from qibo import gates
from qibo.models import QFT, Circuit, OptimisticQFT
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
            backend.matrix_norm(approximate - exact, order=2) <= bound + 1e-8, True
        )
        # elementwise phases
        mask = backend.abs(exact) > 1e-12
        phases = backend.angle(approximate[mask] / exact[mask])
        backend.assert_allclose(backend.max(np.abs(phases)) <= bound + 1e-8, True)
        backend.assert_allclose(backend.abs(approximate), backend.abs(exact), atol=1e-8)


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


def _optimistic_qft_reference(nqubits: int, block_size: int, with_swaps: bool):
    """Algorithm 3 of arXiv:2505.00701 with all layers, including canceling QFT pairs."""
    blocks = [
        list(range(start, min(start + block_size, nqubits)))
        for start in range(0, nqubits, block_size)
    ]

    def window(index):
        return [qubit for block in blocks[index : index + 2] for qubit in block]

    def block_qft(qubits):
        return QFT(len(qubits), with_swaps=False).on_qubits(*qubits)

    circuit = Circuit(nqubits)
    # first layer: windows (0, 1), (2, 3), ..., with a lone last block if needed
    for index in range(0, len(blocks), 2):
        circuit.add(block_qft(window(index)))
    # second layer: inverse QFT on every block
    for block in blocks:
        circuit.add(QFT(len(block), with_swaps=False).invert().on_qubits(*block))
    # third layer: first block alone, windows (1, 2), (3, 4), ..., lone last block if needed
    circuit.add(block_qft(blocks[0]))
    for index in range(1, len(blocks), 2):
        circuit.add(block_qft(window(index)))
    if with_swaps:
        circuit.add(
            gates.SWAP(qubit, nqubits - qubit - 1) for qubit in range(nqubits // 2)
        )
    return circuit


@pytest.mark.parametrize("with_swaps", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5, 6])
def test_optimistic_qft_one_block(backend, nqubits, with_swaps):
    circuit = OptimisticQFT(nqubits, nqubits, with_swaps)
    target = QFT(nqubits, with_swaps)
    assert [gate.name for gate in circuit.queue] == [gate.name for gate in target.queue]
    backend.assert_allclose(
        circuit.unitary(backend), target.unitary(backend), atol=1e-7
    )


@pytest.mark.parametrize("with_swaps", [True, False])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5, 6, 7])
def test_optimistic_qft_reference(backend, nqubits, with_swaps):
    for block_size in range(1, nqubits + 1):
        circuit = OptimisticQFT(nqubits, block_size, with_swaps)
        reference = _optimistic_qft_reference(nqubits, block_size, with_swaps)
        backend.assert_allclose(
            circuit.unitary(backend), reference.unitary(backend), atol=1e-7
        )


@pytest.mark.parametrize("with_swaps", [True, False])
@pytest.mark.parametrize("nqubits,block_size", [(5, 2), (6, 3), (7, 1), (6, 4)])
def test_optimistic_qft_unitary(backend, nqubits, block_size, with_swaps):
    circuit = OptimisticQFT(nqubits, block_size, with_swaps)
    unitary = circuit.unitary(backend)
    identity = backend.identity(2**nqubits)
    backend.assert_allclose(unitary @ backend.dagger(unitary), identity, atol=1e-7)
    backend.assert_allclose(
        circuit.invert().unitary(backend) @ unitary, identity, atol=1e-7
    )


@pytest.mark.parametrize("nqubits,block_size", [(5, 2), (6, 3)])
def test_optimistic_qft_density_matrix(backend, nqubits, block_size):
    circuit = OptimisticQFT(nqubits, block_size, density_matrix=True)

    assert circuit.density_matrix

    state = random_statevector(2**nqubits, backend=backend)
    initial_state = backend.outer(state, backend.conj(state))

    final_state = backend.execute_circuit(circuit, backend.copy(initial_state)).state()

    unitary = OptimisticQFT(nqubits, block_size).unitary(backend)

    target = unitary @ initial_state @ backend.dagger(unitary)

    backend.assert_allclose(final_state, target, atol=1e-7)


@pytest.mark.parametrize("nqubits", [5, 6, 7, 8])
def test_optimistic_qft_average_error(backend, nqubits):
    exact = QFT(nqubits, with_swaps=False).unitary(backend)
    errors = []
    for block_size in range(1, nqubits + 1):
        approximate = OptimisticQFT(nqubits, block_size, False).unitary(backend)
        norm = backend.matrix_norm(approximate - exact, order="fro")
        errors.append(norm**2 / 2**nqubits)
    # the last error is for ``block_size == nqubits``, which is exact
    backend.assert_allclose(errors[-1], 0.0, atol=1e-12)
    errors = [float(error) for error in errors]
    assert all(math.isfinite(error) and error >= 0 for error in errors)
    # the squared distance between two unitaries, divided by the dimension, is at most 4
    assert max(errors) <= 4 + 1e-8
    # empirical regression check for these sizes, not a theorem
    assert all(
        later <= earlier + 1e-12 for earlier, later in itertools.pairwise(errors)
    )
    assert errors[0] > errors[nqubits // 2]


@pytest.mark.parametrize("nqubits", [7, 8])
def test_optimistic_qft_worst_versus_average(backend, nqubits):
    """Small blocks give a small average error, but a large error on some inputs."""
    exact = QFT(nqubits, with_swaps=False).unitary(backend)
    approximate = OptimisticQFT(nqubits, 2, False).unitary(backend)
    difference = approximate - exact
    dimension = 2**nqubits
    errors = [
        float(backend.vector_norm(difference[:, i], order=2) ** 2)
        for i in range(dimension)
    ]
    average = sum(errors) / dimension
    norm = backend.matrix_norm(difference, order="fro")
    backend.assert_allclose(average, norm**2 / dimension, atol=1e-8)
    assert average < 0.5
    assert max(errors) > 2 * average


def test_optimistic_qft_value_error():
    for block_size in [0, -1, 6]:
        with pytest.raises(ValueError):
            OptimisticQFT(5, block_size)

    for block_size in [1.5, True]:
        with pytest.raises(TypeError):
            OptimisticQFT(5, block_size)

    for with_swaps in [1, None]:
        with pytest.raises(TypeError):
            OptimisticQFT(4, 2, with_swaps)

    for devices in [None, {"/GPU:0": 2}]:
        with pytest.raises(NotImplementedError):
            OptimisticQFT(4, 2, accelerators=devices)
