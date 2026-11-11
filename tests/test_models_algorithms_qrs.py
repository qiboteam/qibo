"""Tests for the Quantum Rejection Sampling in `qibo/models/algorithms/qrs.py`."""

import math

import pytest

from qibo import Circuit, gates
from qibo.models import QRS, binary_encoder
from qibo.quantum_info import random_statevector


def _distribution(backend, nqubits, seed):
    return backend.abs(random_statevector(2**nqubits, seed=seed, backend=backend)) ** 2


def _source_circuit(backend, source):
    nqubits = int(math.log2(len(source)))
    return binary_encoder(nqubits, data=backend.sqrt(source), backend=backend)


def _flagged(backend, circuit, nqubits):
    # probability of the flag, and distribution of the other qubits if the flag is 1
    qubits = list(range(nqubits + 1))
    probabilities = backend.execute_circuit(circuit).probabilities(qubits)
    probability = backend.sum(probabilities[1::2])
    return probability, probabilities[1::2] / probability


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_rejection_sampling(backend, nqubits):
    source = _distribution(backend, nqubits, seed=1)
    target = _distribution(backend, nqubits, seed=2)
    algorithm = _source_circuit(backend, source)

    circuit = QRS(algorithm, target, source, measure=False, backend=backend)
    assert circuit.nqubits == nqubits + 1

    # the amplified probability of the flag, and the target distribution if it is 1
    ratio = max(float(t) / float(s) for t, s in zip(target, source))
    theta = math.asin(math.sqrt(1 / ratio))
    iterations = math.floor(math.pi / (4 * theta))
    probability, distribution = _flagged(backend, circuit, nqubits)
    backend.assert_allclose(
        probability, math.sin((2 * iterations + 1) * theta) ** 2, atol=1e-8
    )
    backend.assert_allclose(distribution, target / backend.sum(target), atol=1e-8)


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_rejection_sampling_exact(backend, nqubits):
    source = _distribution(backend, nqubits, seed=3)
    target = _distribution(backend, nqubits, seed=4)
    algorithm = _source_circuit(backend, source)

    # without the source, which is calculated, and with an unnormalized target
    circuit = QRS(algorithm, 5 * target, exact=True, measure=False, backend=backend)
    assert circuit.nqubits == nqubits + 1

    probability, distribution = _flagged(backend, circuit, nqubits)
    backend.assert_allclose(probability, 1.0, atol=1e-8)
    backend.assert_allclose(distribution, target, atol=1e-8)


def test_rejection_sampling_zeros(backend):
    # the source is zero in some states, where the target is also zero
    source = backend.cast([0.5, 0.0, 0.25, 0.25], dtype=backend.float64)
    target = backend.cast([0.1, 0.0, 0.0, 0.9], dtype=backend.float64)
    algorithm = _source_circuit(backend, source)

    circuit = QRS(algorithm, target, exact=True, measure=False, backend=backend)

    probability, distribution = _flagged(backend, circuit, 2)
    backend.assert_allclose(probability, 1.0, atol=1e-8)
    backend.assert_allclose(distribution, target, atol=1e-8)

    # the target is the source, so there is nothing to reject
    circuit = QRS(algorithm, source, measure=False, backend=backend)
    probability, distribution = _flagged(backend, circuit, 2)
    backend.assert_allclose(probability, 1.0, atol=1e-8)
    backend.assert_allclose(distribution, source, atol=1e-8)


@pytest.mark.parametrize("measure", [False, True])
def test_rejection_sampling_measure(backend, measure):
    algorithm = Circuit(2)
    algorithm.add(gates.H(qubit) for qubit in range(2))

    circuit = QRS(
        algorithm,
        [0, 0, 0, 1],
        exact=True,
        measure=measure,
        density_matrix=True,
        backend=backend,
    )
    assert circuit.density_matrix
    assert len(circuit.measurements) == int(measure)

    if measure:
        frequencies = backend.execute_circuit(circuit, nshots=100).frequencies()
        assert set(frequencies) == {"111"}


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("probability", [0.8, 0.9, 0.97])
def test_rejection_sampling_probability(backend, probability, exact):
    # the overlap of the prepared state with the target one is the given probability
    source = backend.cast([0.55, 0.3, 0.1, 0.05], dtype=backend.float64)
    target = backend.cast([0.1, 0.2, 0.3, 0.4], dtype=backend.float64)
    algorithm = _source_circuit(backend, source)

    circuit = QRS(
        algorithm,
        target,
        probability=probability,
        exact=exact,
        measure=False,
        backend=backend,
    )
    flag, distribution = _flagged(backend, circuit, 2)
    overlap = backend.sum(backend.sqrt(distribution * target))
    backend.assert_allclose(overlap**2, probability, atol=1e-6)
    if exact:
        backend.assert_allclose(flag, 1.0, atol=1e-8)


def test_rejection_sampling_probability_limits(backend):
    # the target has weight where the source is zero, so that the overlap cannot be 1
    source = backend.cast([0.5, 0.0, 0.25, 0.25], dtype=backend.float64)
    target = backend.cast([0.1, 0.2, 0.3, 0.4], dtype=backend.float64)
    algorithm = _source_circuit(backend, source)

    with pytest.raises(ValueError):
        QRS(algorithm, target, backend=backend)

    circuit = QRS(
        algorithm, target, probability=0.8, exact=True, measure=False, backend=backend
    )
    flag, distribution = _flagged(backend, circuit, 2)
    backend.assert_allclose(flag, 1.0, atol=1e-8)
    overlap = backend.sum(backend.sqrt(distribution * target))
    backend.assert_allclose(overlap**2, 0.8, atol=1e-6)

    # for the smallest overlap, the source is only copied, with the flag in the state 1
    smallest = float(backend.sum(backend.sqrt(source * target)) ** 2)
    circuit = QRS(
        algorithm,
        target,
        probability=smallest,
        exact=True,
        measure=False,
        backend=backend,
    )
    flag, distribution = _flagged(backend, circuit, 2)
    backend.assert_allclose(flag, 1.0, atol=1e-8)
    backend.assert_allclose(distribution, source, atol=1e-8)


@pytest.mark.parametrize("exact", [False, True])
def test_rejection_sampling_index_qubits(backend, exact):
    # the state is sqrt(p_0) |0>|0> + sqrt(p_1) |1>|+>, and the second qubit is not an index
    def prepare(probabilities):
        circuit = Circuit(2)
        circuit.add(gates.RY(0, theta=2 * math.acos(math.sqrt(probabilities[0]))))
        circuit.add(gates.H(1).controlled_by(0))
        return circuit

    circuit = QRS(
        prepare([0.8, 0.2]),
        [0.3, 0.7],
        qubits=[0],
        exact=exact,
        measure=False,
        backend=backend,
    )
    assert circuit.nqubits == 3

    # if the flag is 1, the state has the target amplitudes and the same states of the second
    # qubit, and the flag is 1 with certainty if exact
    expected = backend.execute_circuit(prepare([0.3, 0.7])).state()
    state = backend.execute_circuit(circuit).state().reshape(4, 2)
    flagged = state[:, 1]
    weight = backend.sum(backend.abs(flagged) ** 2)
    overlap = backend.abs(backend.sum(backend.conj(expected) * flagged)) ** 2
    backend.assert_allclose(overlap / weight, 1.0, atol=1e-8)
    if exact:
        backend.assert_allclose(weight, 1.0, atol=1e-8)


def test_rejection_sampling_errors(backend):
    algorithm = Circuit(2)
    algorithm.add(gates.H(qubit) for qubit in range(2))
    uniform = [0.25] * 4

    with pytest.raises(TypeError):
        QRS(gates.H(0), uniform, backend=backend)

    for target in ([1, 0, 0], [0, 0, 0, 0], [-1, 1, 0, 1]):
        with pytest.raises(ValueError):
            QRS(algorithm, target, backend=backend)

    with pytest.raises(ValueError):
        QRS(algorithm, uniform, source=[1.0] * 3, backend=backend)

    # the target has to be zero where the source is zero
    with pytest.raises(ValueError):
        QRS(algorithm, uniform, source=[0.0, 1.0, 1.0, 1.0], backend=backend)

    for probability in (0, -0.5, 1.5):
        with pytest.raises(ValueError):
            QRS(algorithm, uniform, probability=probability, backend=backend)

    for qubits in ([], [0, 0], [0, 2], [-1], [0.5]):
        with pytest.raises(ValueError):
            QRS(algorithm, uniform, qubits=qubits, backend=backend)
