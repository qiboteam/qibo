"""Tests for direct fidelity estimation with Pauli measurements."""

import math
import tracemalloc
from functools import reduce
from types import SimpleNamespace

import numpy as np
import pytest

import qibo.tomography.direct_fidelity_estimation as module
from qibo import Circuit, gates
from qibo.tomography import DirectFidelityEstimation, Tomography


def test_abstract_interface():
    assert issubclass(DirectFidelityEstimation, Tomography)


def test_call_cache(backend, monkeypatch):
    """The transform of each different X part is computed once, for all its draws."""
    calls = []
    transform = module._fast_walsh_hadamard_transform

    def count(*args, **kwargs):
        calls.append(1)
        return transform(*args, **kwargs)

    monkeypatch.setattr(module, "_fast_walsh_hadamard_transform", count)
    _, observed = _spy(backend, monkeypatch)

    # the X parts of a GHZ state are only 000 and 111, among the 1000 draws
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    DirectFidelityEstimation()(circuit, seed=1, backend=backend)

    # two transforms for the marginal, and one for each different X part
    assert {tuple(pauli in "XY" for pauli in s) for s in observed} == {
        (False,) * 3,
        (True,) * 3,
    }
    assert len(calls) == 2 + 2

    # a state with all the X parts
    calls.clear()
    observed.clear()
    DirectFidelityEstimation()(_prepare(3), seed=1, backend=backend)

    patterns = {tuple(pauli in "XY" for pauli in s) for s in observed}
    assert len(patterns) == 8
    assert len(calls) == 2 + len(patterns)


@pytest.mark.parametrize("nqubits", [3, 4])
def test_call_distribution(backend, monkeypatch, nqubits):
    """The Pauli strings are drawn with the probabilities of the characteristic function,
    which tests the marginal of the X part, and the conditional of the Z part."""
    shots, observed = _spy(backend, monkeypatch)

    circuit = _prepare(nqubits)
    state = backend.execute_circuit(circuit).state()
    ideal = backend.cast(
        [
            backend.dot(
                backend.ravel(backend.conj(state)),
                backend.ravel(_pauli(index, nqubits) @ state),
            ).real
            for index in range(4**nqubits)
        ],
        dtype=backend.float64,
    )

    epsilon, delta = 0.01, 0.05
    nsamples = math.ceil(1 / (epsilon**2 * delta))
    DirectFidelityEstimation()(
        circuit, epsilon=epsilon, delta=delta, seed=1, backend=backend
    )

    indices, counts = _counts(observed, shots, ideal, epsilon, delta)
    frequencies = backend.zeros(4**nqubits, dtype=backend.float64)
    frequencies[indices] = counts / nsamples

    assert 0.5 * backend.sum(backend.abs(frequencies - ideal**2 / 2**nqubits)) < 0.03


def test_call_errors(backend):
    circuit = _prepare(2)
    state = backend.execute_circuit(circuit).state()

    for epsilon in (0, 1, -0.1, 1.5, True, "0.1"):
        with pytest.raises(ValueError):
            DirectFidelityEstimation()(circuit, epsilon=epsilon, backend=backend)

    for delta in (0, 1, -0.1, 1.5, True, "0.1"):
        with pytest.raises(ValueError):
            DirectFidelityEstimation()(circuit, delta=delta, backend=backend)

    # the target must be a normalized state vector of the qubits of the circuit
    for target in (state[:2], 2 * state, np.outer(state, state), _prepare(3)):
        with pytest.raises(ValueError):
            DirectFidelityEstimation()(circuit, target, backend=backend)

    # the target circuit must prepare a pure state, which excludes the density matrices,
    # and the noise channels that are sampled, since their state is not unique
    with pytest.raises(ValueError):
        DirectFidelityEstimation()(circuit, _prepare(2, noisy=True), backend=backend)
    with pytest.raises(ValueError):
        DirectFidelityEstimation()(_prepare(2, noisy=True), backend=backend)

    sampled = _prepare(2)
    sampled.add(gates.PauliNoiseChannel(0, [("X", 0.3)]))
    with pytest.raises(ValueError):
        DirectFidelityEstimation()(circuit, sampled, backend=backend)

    # the circuits must have no measurements
    measured = _prepare(2)
    measured.add(gates.M(0, 1))
    with pytest.raises(ValueError):
        DirectFidelityEstimation()(measured, circuit, backend=backend)
    with pytest.raises(ValueError):
        DirectFidelityEstimation()(circuit, measured, backend=backend)


@pytest.mark.parametrize(
    "nqubits,noisy", [(1, False), (2, True), (3, False), (3, True)]
)
def test_call_exact(backend, monkeypatch, nqubits, noisy):
    """With the exact probabilities, the estimator is the average of the ratios of the
    expectation values of the drawn Pauli strings, which tests the bases of the Pauli
    strings, the order of the qubits, the signs, the weights, and the number of shots.
    """
    shots, observed = _spy(backend, monkeypatch)

    target = _prepare(nqubits)
    prepared = _prepare(nqubits, noisy)

    state = backend.to_numpy(backend.execute_circuit(target).state())
    prepared_state = backend.to_numpy(backend.execute_circuit(prepared).state())
    sigma = prepared_state if noisy else np.outer(prepared_state, prepared_state.conj())

    strings = [_pauli(index, nqubits) for index in range(4**nqubits)]
    ideal = np.array([np.vdot(state, string @ state).real for string in strings])
    actual = np.array([np.trace(string @ sigma).real for string in strings])

    epsilon, delta = 0.2, 0.3
    nsamples = math.ceil(1 / (epsilon**2 * delta))
    estimate = DirectFidelityEstimation()(
        prepared, target, epsilon=epsilon, delta=delta, seed=3, backend=backend
    )

    # one circuit for each different string, in alphabetical order, with the shots of
    # all its repetitions, and only strings with a non-zero probability
    indices, counts = _counts(observed, shots, ideal, epsilon, delta)
    assert observed == sorted(set(observed))
    assert np.all(np.abs(ideal[indices]) > 1e-3)
    backend.assert_allclose(counts, np.round(counts), atol=1e-9)
    backend.assert_allclose(sum(counts), nsamples)

    backend.assert_allclose(
        estimate, np.sum(counts * actual[indices] / ideal[indices]) / nsamples
    )


def test_call_memory(backend, monkeypatch):
    """The table of the :math:`4^{n}` probabilities is never built, so the memory is
    far below its size, which is 128 MiB of floats for 12 qubits."""

    def execute(self, circuits, nshots=1000, backend=None):
        return [SimpleNamespace(frequencies=lambda binary=False: {0: nshots})]

    monkeypatch.setattr(Tomography, "execute", execute)

    nqubits = 12
    rng = np.random.default_rng(0)
    state = rng.normal(size=2**nqubits) + 1j * rng.normal(size=2**nqubits)
    state = backend.cast(state / np.linalg.norm(state), dtype=backend.complex128)

    # a first call to exclude the allocations that are made only once
    warmup = backend.cast(np.ones(8) / math.sqrt(8), dtype=backend.complex128)
    DirectFidelityEstimation()(Circuit(3), warmup, backend=backend)

    tracemalloc.start()
    estimate = DirectFidelityEstimation()(
        Circuit(nqubits), state, epsilon=0.2, delta=0.2, seed=1, backend=backend
    )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert math.isfinite(estimate)
    assert peak < 16 * 2**20


def test_call_sampling(backend):
    target = _prepare(3)
    prepared = _prepare(3, noisy=True)

    state = backend.execute_circuit(target).state()
    sigma = backend.execute_circuit(prepared).state()
    fidelity = float(backend.real(backend.conj(state) @ sigma @ state))

    # the error is within twice the additive error, with probability 1 - 2 * delta
    estimate = DirectFidelityEstimation()(
        prepared, target, epsilon=0.1, delta=0.1, seed=1, backend=backend
    )
    assert abs(estimate - fidelity) < 0.2

    # the estimator is unbiased, and its error is about 0.03 for each seed
    estimates = [
        DirectFidelityEstimation()(
            prepared, target, epsilon=0.1, delta=0.1, seed=seed, backend=backend
        )
        for seed in range(2, 22)
    ]
    backend.assert_allclose(np.mean(estimates), fidelity, atol=0.03)


def test_call_seed(backend):
    target = _prepare(3)
    prepared = _prepare(3, noisy=True)

    def estimate(seed):
        return DirectFidelityEstimation()(
            prepared, target, epsilon=0.2, delta=0.2, seed=seed, backend=backend
        )

    backend.assert_allclose(estimate(5), estimate(5))
    assert estimate(5) != estimate(6)

    # without a seed, it runs with the generator of the backend
    assert isinstance(estimate(None), float)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_call_stabilizer(backend, monkeypatch, seed):
    """The expectation values of a stabilizer state are one or minus one, so the exact
    estimator is one for any sampling, which tests the sampling of the Pauli strings."""
    monkeypatch.setattr(Tomography, "execute", _execute_exact(backend, []))

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.S(1))
    circuit.add(gates.X(2))
    circuit.add(gates.H(2))
    circuit.add(gates.S(2))

    estimate = DirectFidelityEstimation()(circuit, seed=seed, backend=backend)

    backend.assert_allclose(estimate, 1.0, atol=1e-12)


def test_call_target(backend):
    circuit = _prepare(2)
    state = backend.execute_circuit(circuit).state()
    prepared = _prepare(2, noisy=True)

    def estimate(target, circuit=prepared):
        return DirectFidelityEstimation()(
            circuit, target, epsilon=0.2, delta=0.2, seed=1, backend=backend
        )

    # the target can be a circuit or its state vector, which are the same
    backend.assert_allclose(estimate(circuit), estimate(state))
    backend.assert_allclose(estimate(circuit), estimate(backend.to_numpy(state)))

    # by default, the target is the state prepared by the circuit
    backend.assert_allclose(estimate(None, circuit), estimate(circuit, circuit))


def test_circuits():
    """The circuits are those of the Pauli strings of the abstract class."""
    circuit = Circuit(3)
    circuit.add(gates.RX(0, 0.3))

    circuits = DirectFidelityEstimation().circuits(circuit, ["XYZ", "IIY"])

    assert [[(gate.name, gate.qubits) for gate in c.queue] for c in circuits] == [
        [("rx", (0,)), ("h", (0,)), ("sdg", (1,)), ("h", (1,)), ("measure", (0, 1, 2))],
        [("rx", (0,)), ("sdg", (2,)), ("h", (2,)), ("measure", (0, 1, 2))],
    ]

    with pytest.raises(ValueError):
        DirectFidelityEstimation().circuits(circuit, ["XY"])


def _counts(observed, shots, ideal, epsilon, delta):
    """Indices in the Pauli basis, and number of times drawn, of the Pauli strings that
    were measured, from their shots and the expectation values ``ideal``."""
    nsamples = math.ceil(1 / (epsilon**2 * delta))
    indices = [
        sum("IXYZ".index(pauli) * 4 ** (len(s) - 1 - i) for i, pauli in enumerate(s))
        for s in observed
    ]
    counts = np.array(
        [
            nshots
            / math.ceil(
                2 * math.log(2 / delta) / (ideal[index] ** 2 * nsamples * epsilon**2)
            )
            for nshots, index in zip(shots, indices)
        ]
    )

    return indices, counts


def _execute_exact(backend, shots):
    """Exact probabilities of each circuit, as the frequencies of its shots.

    The shots of each circuit are appended to ``shots``.
    """

    def execute(self, circuits, nshots=1000, backend=backend):
        results = []
        for tomography_circuit in circuits:
            shots.append(nshots)
            without_measurements = Circuit(
                tomography_circuit.nqubits,
                density_matrix=tomography_circuit.density_matrix,
            )
            without_measurements.add(tomography_circuit.queue[:-1])
            probabilities = backend.to_numpy(
                backend.execute_circuit(without_measurements).probabilities()
            )
            results.append(
                SimpleNamespace(
                    frequencies=lambda binary=False, probabilities=probabilities: dict(
                        enumerate((nshots * probabilities).tolist())
                    )
                )
            )

        return results

    return execute


def _label(index, nqubits):
    """Pauli string at ``index`` of the Pauli basis, as a string of I, X, Y, Z."""
    return "".join(
        "IXYZ"[(int(index) // 4 ** (nqubits - 1 - qubit)) % 4]
        for qubit in range(nqubits)
    )


def _pauli(index, nqubits):
    """Pauli string at ``index`` of the Pauli basis, with the first qubit as the most
    significant digit, and I, X, Y, Z as 0, 1, 2, 3."""
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }

    return reduce(np.kron, [matrices[pauli] for pauli in _label(index, nqubits)])


def _prepare(nqubits, noisy=False):
    """State with real and imaginary parts, to test all the Pauli bases."""
    circuit = Circuit(nqubits, density_matrix=noisy)
    for qubit in range(nqubits):
        circuit.add(gates.RX(qubit, 0.4 + 0.5 * qubit))
        circuit.add(gates.RY(qubit, 1.1 - 0.3 * qubit))
        circuit.add(gates.RZ(qubit, 0.7 + qubit))
    for qubit in range(nqubits - 1):
        circuit.add(gates.CNOT(qubit, qubit + 1))
        circuit.add(gates.RY(qubit + 1, 0.9))
    if noisy:
        circuit.add(gates.DepolarizingChannel(tuple(range(nqubits)), 0.15))

    return circuit


def _spy(backend, monkeypatch):
    """Executes the circuits with their exact probabilities, and records the shots of
    each one, and the Pauli strings that they measure."""
    shots, observed = [], []
    monkeypatch.setattr(Tomography, "execute", _execute_exact(backend, shots))

    circuits = DirectFidelityEstimation.circuits

    def record(self, circuit, observables):
        observed.extend(observables)
        return circuits(self, circuit, observables)

    monkeypatch.setattr(DirectFidelityEstimation, "circuits", record)

    return shots, observed
