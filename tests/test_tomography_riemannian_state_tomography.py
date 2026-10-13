"""Tests for low-rank state tomography with Riemannian gradient descent."""

import math
from functools import reduce
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.tomography import RiemannianStateTomography, Tomography


def test_abstract_interface():
    assert issubclass(RiemannianStateTomography, Tomography)


def test_call_convergence(backend, monkeypatch):
    """With exact data, the error decreases by a constant factor at each iteration."""
    monkeypatch.setattr(Tomography, "execute", _execute_exact)

    circuit = _prepare(3)
    state = backend.execute_circuit(circuit).state()
    exact = backend.to_numpy(backend.outer(state, backend.conj(state)))

    errors = []
    for iterations in (1, 4, 7, 10):
        estimate = RiemannianStateTomography()(
            circuit, 1, 48, 1, iterations, 0, seed=1, backend=backend
        )
        errors.append(np.linalg.norm(backend.to_numpy(estimate) - exact))

    # the factor of the error is about one half at each iteration, so that of
    # three iterations is about one eighth
    assert all(later < 0.25 * earlier for earlier, later in pairwise(errors))
    assert errors[-1] < 1e-3


def test_call_errors(backend):
    circuit = _prepare(2)

    for nshots in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError):
            RiemannianStateTomography()(circuit, 1, 8, nshots=nshots, backend=backend)

    for rank in (0, 5, 2.5, True, "1"):
        with pytest.raises(ValueError):
            RiemannianStateTomography()(circuit, rank, 8, backend=backend)

    for nobservables in (0, 17, 2.5, True, "8"):
        with pytest.raises(ValueError):
            RiemannianStateTomography()(circuit, 1, nobservables, backend=backend)

    for iterations in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError):
            RiemannianStateTomography()(
                circuit, 1, 8, iterations=iterations, backend=backend
            )

    for tolerance in (-1e-3, True, "1e-3"):
        with pytest.raises(ValueError):
            RiemannianStateTomography()(
                circuit, 1, 8, tolerance=tolerance, backend=backend
            )

    measured = _prepare(2)
    measured.add(gates.M(0, 1))
    with pytest.raises(ValueError):
        RiemannianStateTomography()(measured, 1, 8, backend=backend)


@pytest.mark.parametrize(
    "nqubits,nobservables,rank,mixed",
    [
        (2, 16, 1, False),
        (3, 48, 1, False),
        (3, 64, 2, True),
        (4, 160, 1, False),
        (4, 224, 2, True),
    ],
)
def test_call_exact(backend, monkeypatch, nqubits, nobservables, rank, mixed):
    """With the exact probabilities of the Pauli strings, the state is recovered.

    It tests the bases of the Pauli strings, the order of the qubits, and the
    iterations, apart from the sampling of the outcomes. The mixed state has rank two.
    """
    monkeypatch.setattr(Tomography, "execute", _execute_exact)

    circuit = _prepare(nqubits, mixed)
    state = backend.execute_circuit(circuit).state()
    exact = state if mixed else backend.outer(state, backend.conj(state))

    estimate = RiemannianStateTomography()(
        circuit,
        rank,
        nobservables,
        nshots=1,
        iterations=500,
        tolerance=1e-13,
        seed=1,
        backend=backend,
    )

    assert estimate.shape == (2**nqubits, 2**nqubits)
    backend.assert_allclose(estimate, exact, atol=1e-6)


def test_call_nobservables(backend, monkeypatch):
    """One circuit is executed for each Pauli string."""
    calls = []
    execute = Tomography.execute

    def recorder(self, circuits, nshots=1000, backend=None):
        calls.append((len(circuits), nshots))
        return execute(self, circuits, nshots, backend)

    monkeypatch.setattr(Tomography, "execute", recorder)

    circuit = _prepare(3)
    RiemannianStateTomography()(circuit, 1, 20, 300, backend=backend)
    RiemannianStateTomography()(circuit, 1, 64, backend=backend)

    assert calls == [(20, 300), (64, 1000)]


@pytest.mark.parametrize("rank", [1, 2, 8])
def test_call_rank(backend, rank):
    """The estimate is positive semidefinite with a rank that is at most ``rank``.

    The full rank, which is :math:`8`, tests the truncation of the negative eigenvalues,
    since the adjoint of the data of a subset of the Pauli strings has some of them.
    """
    circuit = _prepare(3)

    estimate = RiemannianStateTomography()(
        circuit, rank, 48, 500, 100, seed=2, backend=backend
    )

    eigenvalues = backend.to_numpy(backend.eigvalsh(estimate))
    assert eigenvalues[0] > -1e-10
    assert np.sum(eigenvalues > 1e-10) <= rank
    backend.assert_allclose(estimate, backend.dagger(estimate), atol=1e-12)


@pytest.mark.parametrize("iterations", [1, 2, 3])
@pytest.mark.parametrize(
    "nqubits,nobservables,rank,mixed",
    [(3, 40, 1, False), (3, 48, 2, True), (4, 128, 1, False)],
)
def test_call_reference(
    backend, monkeypatch, nqubits, nobservables, rank, mixed, iterations
):
    """The iterates agree with a dense implementation of the algorithm.

    The reference builds the Pauli strings, the projection onto the tangent space, and
    the truncation as dense matrices, and it does not use the transforms between
    operators and Pauli coefficients, nor the basis of the tangent space.
    """
    monkeypatch.setattr(Tomography, "execute", _execute_exact)

    circuit = _prepare(nqubits, mixed)

    estimate = RiemannianStateTomography()(
        circuit, rank, nobservables, 1, iterations, 0, seed=3, backend=backend
    )

    reference = _reference(backend, circuit, rank, nobservables, iterations, seed=3)
    backend.assert_allclose(estimate, reference, atol=1e-8)


def test_call_sampling(backend):
    circuit = _prepare(4)
    state = backend.execute_circuit(circuit).state()

    estimate = RiemannianStateTomography()(
        circuit, rank=1, nobservables=128, nshots=2000, seed=1, backend=backend
    )

    fidelity = backend.real(backend.conj(state) @ estimate @ state)
    backend.assert_allclose(fidelity, 1.0, atol=0.03)
    # the trace is not constrained, but the errors of the expectation values are small
    backend.assert_allclose(backend.trace(estimate), 1.0, atol=0.05)
    backend.assert_allclose(estimate, backend.dagger(estimate), atol=1e-12)
    assert float(backend.eigvalsh(estimate)[0]) > -1e-10


def test_call_seed(backend):
    circuit = _prepare(3)

    def estimate(seed):
        return RiemannianStateTomography()(
            circuit, 1, 48, 200, seed=seed, backend=backend
        )

    backend.assert_allclose(estimate(5), estimate(5))
    assert float(backend.max(backend.abs(estimate(5) - estimate(6)))) > 0

    # without a seed, it runs with the generator of the backend
    assert estimate(None).shape == (8, 8)


def test_call_stopping(backend):
    """The iterations stop at the maximum number, or when the change is small."""
    circuit = _prepare(3)

    def estimate(**kwargs):
        return RiemannianStateTomography()(
            circuit, 1, 48, 200, seed=1, backend=backend, **kwargs
        )

    first = estimate(iterations=1)
    loose = estimate(tolerance=1e3)
    many = estimate(iterations=50, tolerance=0)

    backend.assert_allclose(loose, first)
    assert float(backend.max(backend.abs(many - first))) > 1e-6


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_circuits(nqubits):
    circuit = Circuit(nqubits)
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.CNOT(0, nqubits - 1) if nqubits > 1 else gates.RY(0, 0.2))
    nprepared = len(circuit.queue)

    observables = ["XYZ"[qubit % 3] for qubit in range(nqubits)]
    observables = ["".join(observables), "I" * nqubits, "Y" * nqubits, "Z" * nqubits]
    circuits = RiemannianStateTomography().circuits(circuit, observables)

    # the circuit given by the user is not modified
    assert len(circuit.queue) == nprepared
    assert not circuit.measurements

    assert len(circuits) == len(observables)

    for observable, tomography_circuit in zip(observables, circuits):
        queue = tomography_circuit.queue

        # the state preparation is kept as it is
        assert [(gate.name, gate.qubits) for gate in queue[:nprepared]] == [
            (gate.name, gate.qubits) for gate in circuit.queue
        ]

        expected = []
        for qubit, pauli in enumerate(observable):
            if pauli == "Y":
                expected.append(("sdg", (qubit,)))
            if pauli in "XY":
                expected.append(("h", (qubit,)))
        expected.append(("measure", tuple(range(nqubits))))

        assert [(gate.name, gate.qubits) for gate in queue[nprepared:]] == expected


def test_circuits_errors():
    circuit = Circuit(2)
    circuit.add(gates.H(0))

    for observable in ("X", "XYZ", "XA", "xy", 1):
        with pytest.raises(ValueError):
            RiemannianStateTomography().circuits(circuit, ["XX", observable])

    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        RiemannianStateTomography().circuits(circuit, ["XX"])


def _execute_exact(self, circuits, nshots=1000, backend=None):
    """Exact probabilities of each circuit, as the frequencies of one shot."""
    results = []
    for tomography_circuit in circuits:
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
                    enumerate(probabilities.tolist())
                )
            )
        )

    return results


def _prepare(nqubits, mixed=False):
    """State with real and imaginary parts, to test all the Pauli bases."""
    circuit = Circuit(nqubits, density_matrix=mixed)
    for qubit in range(nqubits):
        circuit.add(gates.RX(qubit, 0.4 + 0.5 * qubit))
        circuit.add(gates.RY(qubit, 1.1 - 0.3 * qubit))
        circuit.add(gates.RZ(qubit, 0.7 + qubit))
    for qubit in range(nqubits - 1):
        circuit.add(gates.CNOT(qubit, qubit + 1))
        circuit.add(gates.RY(qubit + 1, 0.9))
    if mixed:
        # a bit flip on one qubit gives a state of rank two
        circuit.add(gates.PauliNoiseChannel(0, [("X", 0.3)]))

    return circuit


def _reference(backend, circuit, rank, nobservables, iterations, seed):
    """Dense Riemannian gradient descent with the exact expectation values."""
    nqubits = circuit.nqubits
    dim = 2**nqubits
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }

    # the same Pauli strings as the protocol, which are sorted
    indices = backend.to_numpy(
        backend.sort(
            backend.random_choice(
                dim**2,
                size=nobservables,
                replace=False,
                seed=seed,
                dtype=backend.int64,
            )
        )
    )
    strings = [
        reduce(
            np.kron,
            [
                matrices["IXYZ"[(int(index) // 4 ** (nqubits - 1 - qubit)) % 4]]
                for qubit in range(nqubits)
            ],
        )
        for index in indices
    ]

    scale = math.sqrt(dim / nobservables)

    def sensing(matrix):
        return scale * np.array([np.trace(string @ matrix).real for string in strings])

    def adjoint(vector):
        return scale * sum(value * string for value, string in zip(vector, strings))

    def truncate(matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        kept = np.argsort(eigenvalues)[-rank:]
        vectors = eigenvectors[:, kept]
        return (vectors * np.maximum(eigenvalues[kept], 0)) @ vectors.conj().T, vectors

    state = backend.to_numpy(backend.execute_circuit(circuit).state())
    exact = np.outer(state, state.conj()) if state.ndim == 1 else state
    data = sensing(exact)

    estimate, vectors = truncate(adjoint(data))
    for _ in range(iterations):
        gradient = adjoint(data - sensing(estimate))
        projector = vectors @ vectors.conj().T
        tangent = (
            projector @ gradient
            + gradient @ projector
            - projector @ gradient @ projector
        )
        step = np.linalg.norm(tangent) ** 2 / np.linalg.norm(sensing(tangent)) ** 2
        estimate, vectors = truncate(estimate + step * tangent)

    return estimate
