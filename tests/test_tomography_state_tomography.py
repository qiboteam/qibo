"""Tests for state tomography."""

import math
from functools import reduce
from itertools import pairwise, product
from types import SimpleNamespace

import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.models.error_mitigation import get_response_matrix
from qibo.noise import NoiseModel, ReadoutError
from qibo.tomography import StateTomography, Tomography
from qibo.tomography.state_tomography import _project_physical

METHODS = ["linear-inversion", "physical-linear-inversion", "least-squares", "mle"]
PHYSICAL_METHODS = METHODS[1:]


def test_abstract_interface():
    assert issubclass(StateTomography, Tomography)


@pytest.mark.parametrize("method", METHODS)
def test_call_bell_state(backend, method):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    estimate = StateTomography()(
        circuit, nshots=20000, seed=3, method=method, backend=backend
    )

    bell = backend.cast([1, 0, 0, 1], dtype=backend.complex128) / math.sqrt(2)
    fidelity = backend.real(backend.conj(bell) @ estimate @ bell)
    backend.assert_allclose(fidelity, 1.0, atol=0.02)


def test_call_errors(backend):
    circuit = _prepare(2)

    for nshots in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError):
            StateTomography()(circuit, nshots=nshots, backend=backend)

    for method in ("mlee", "linear_inversion", None):
        with pytest.raises(ValueError):
            StateTomography()(circuit, method=method, backend=backend)

    for iterations in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError):
            StateTomography()(circuit, iterations=iterations, backend=backend)

    for tolerance in (-1e-3, True, "1e-3"):
        with pytest.raises(ValueError):
            StateTomography()(circuit, tolerance=tolerance, backend=backend)

    for shape in ((2, 2), (4, 3), (16,)):
        with pytest.raises(ValueError):
            StateTomography()(
                circuit, response_matrix=np.ones(shape) / 4, backend=backend
            )

    measured = _prepare(2)
    measured.add(gates.M(0, 1))
    with pytest.raises(ValueError):
        StateTomography()(measured, backend=backend)


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_call_exact(backend, monkeypatch, nqubits, mixed, method, noisy):
    """With the exact probabilities of the settings, the state is recovered exactly.

    It tests the bases of the settings, the order of the qubits, the estimators, and
    the response matrix, apart from the sampling. The maximum likelihood estimate of a
    pure state is on the boundary of the states, where it converges slowly.
    """
    response_matrix = None
    if noisy:
        # column-stochastic matrix of the readout of each qubit
        single = np.array([[0.95, 0.10], [0.05, 0.90]])
        response_matrix = reduce(np.kron, [single] * nqubits)

    # the exact probabilities of each setting, as the frequencies of one shot
    def execute(self, circuits, nshots=1000, backend=None):
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
            if noisy:
                probabilities = response_matrix @ probabilities
            results.append(
                SimpleNamespace(
                    frequencies=lambda binary=False, probabilities=probabilities: dict(
                        enumerate(probabilities.tolist())
                    )
                )
            )
        return results

    monkeypatch.setattr(Tomography, "execute", execute)

    circuit = _prepare(nqubits, mixed)
    state = backend.execute_circuit(circuit).state()
    exact = state if mixed else backend.outer(state, backend.conj(state))

    estimate = StateTomography()(
        circuit,
        nshots=1,
        method=method,
        response_matrix=response_matrix,
        iterations=3000,
        tolerance=1e-12,
        backend=backend,
    )

    assert estimate.shape == (2**nqubits, 2**nqubits)
    atol = 1e-3 if method == "mle" and not mixed else 1e-7
    backend.assert_allclose(estimate, exact, atol=atol)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_call_least_squares_minimum(backend, nqubits):
    """The sum of squares of least squares is between the unconstrained minimum,
    which is that of linear inversion, and that of the other physical estimators."""
    circuit = _prepare(nqubits)
    frequencies = _frequencies(backend, circuit, 100, 1)

    def cost(method):
        estimate = StateTomography()(
            circuit,
            100,
            1,
            method,
            iterations=5000,
            tolerance=1e-12,
            backend=backend,
        )
        probabilities = _probabilities(backend.to_numpy(estimate), nqubits)
        return np.sum((probabilities - frequencies) ** 2)

    minimum = cost("least-squares")

    assert cost("linear-inversion") <= minimum + 1e-9
    assert cost("physical-linear-inversion") >= minimum - 1e-9
    assert cost("mle") >= minimum - 1e-9


@pytest.mark.parametrize("nqubits", [1, 2])
def test_call_mle_likelihood(backend, nqubits):
    """The likelihood never decreases with the iterations, and it is the largest."""
    circuit = _prepare(nqubits)
    frequencies = _frequencies(backend, circuit, 100, 1)

    def log_likelihood(method, iterations=3000):
        estimate = StateTomography()(
            circuit,
            100,
            1,
            method,
            iterations=iterations,
            tolerance=0,
            backend=backend,
        )
        probabilities = _probabilities(backend.to_numpy(estimate), nqubits)
        return np.sum(frequencies * np.log(np.maximum(probabilities, 1e-300)))

    values = [log_likelihood("mle", iterations) for iterations in (1, 10, 100, 1000)]

    assert all(later >= earlier - 1e-12 for earlier, later in pairwise(values))
    assert log_likelihood("mle") >= log_likelihood("least-squares") - 1e-9
    assert log_likelihood("mle") >= log_likelihood("physical-linear-inversion") - 1e-9


def test_call_nshots(backend, monkeypatch):
    """The number of shots is that of each of the settings."""
    calls = []
    execute = Tomography.execute

    def recorder(self, circuits, nshots=1000, backend=None):
        calls.append((len(circuits), nshots))
        return execute(self, circuits, nshots, backend)

    monkeypatch.setattr(Tomography, "execute", recorder)

    circuit = _prepare(2)
    StateTomography()(circuit, backend=backend)
    StateTomography()(circuit, 300, backend=backend)

    assert calls == [(9, 1000), (9, 300)]


def test_call_not_positive_semidefinite(backend):
    """Linear inversion does not project the estimate onto the physical states."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    estimate = StateTomography()(circuit, nshots=100, seed=1, backend=backend)

    assert float(backend.eigvalsh(estimate)[0]) < 0


@pytest.mark.parametrize("method", PHYSICAL_METHODS)
def test_call_physical(backend, method):
    """The estimate is a density matrix, and its sampling is as in linear inversion."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    state = backend.execute_circuit(circuit).state()
    exact = backend.outer(state, backend.conj(state))

    linear = StateTomography()(circuit, nshots=100, seed=1, backend=backend)
    estimate = StateTomography()(circuit, 100, 1, method, backend=backend)

    assert float(backend.eigvalsh(estimate)[0]) > -1e-10
    backend.assert_allclose(estimate, backend.dagger(estimate), atol=1e-12)
    backend.assert_allclose(backend.trace(estimate), 1.0, atol=1e-12)

    # the projection onto a convex set that contains the state is never farther from it
    if method == "physical-linear-inversion":
        distance = float(backend.max(backend.abs(estimate - exact)))
        assert distance <= float(backend.max(backend.abs(linear - exact)))


@pytest.mark.parametrize("method", METHODS)
def test_call_response_matrix_identity(backend, method):
    """An ideal readout is the same as no response matrix."""
    circuit = _prepare(2)

    estimate = StateTomography()(circuit, 200, 1, method, backend=backend)
    ideal = StateTomography()(
        circuit, 200, 1, method, response_matrix=np.eye(4), backend=backend
    )

    backend.assert_allclose(ideal, estimate, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_call_response_matrix_noise(backend, monkeypatch, method):
    """The response matrix of the noise model of the readout mitigates its errors."""
    readout = np.array([[0.93, 0.07], [0.12, 0.88]])
    noise_model = NoiseModel()
    noise_model.add(ReadoutError(np.kron(readout, readout)), gates.M)

    def execute(self, circuits, nshots=1000, backend=None):
        results = []
        for circuit in circuits:
            noisy = noise_model.apply(circuit)
            noisy.density_matrix = True
            results.append(backend.execute_circuit(noisy, nshots=nshots))
        return results

    monkeypatch.setattr(Tomography, "execute", execute)

    backend.set_seed(7)
    response_matrix = get_response_matrix(
        2, noise_model=noise_model, nshots=100000, backend=backend
    )

    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    bell = backend.cast([1, 0, 0, 1], dtype=backend.complex128) / math.sqrt(2)

    def fidelity(**kwargs):
        estimate = StateTomography()(
            circuit, 20000, 1, method, backend=backend, **kwargs
        )
        return float(backend.real(backend.conj(bell) @ estimate @ bell))

    assert fidelity() < 0.85
    assert fidelity(response_matrix=response_matrix) > 0.97


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_call_sampling(backend, nqubits, method):
    circuit = _prepare(nqubits)
    state = backend.execute_circuit(circuit).state()
    exact = backend.outer(state, backend.conj(state))

    estimate = StateTomography()(circuit, 20000, 1, method, backend=backend)

    assert estimate.shape == (2**nqubits, 2**nqubits)
    backend.assert_allclose(estimate, exact, atol=0.03)
    # the estimate is exactly Hermitian with unit trace, whatever the outcomes are
    backend.assert_allclose(estimate, backend.dagger(estimate), atol=1e-12)
    backend.assert_allclose(backend.trace(estimate), 1.0, atol=1e-12)
    if method != "linear-inversion":
        assert float(backend.eigvalsh(estimate)[0]) > -1e-10


@pytest.mark.parametrize("method", METHODS)
def test_call_seed(backend, method):
    circuit = _prepare(2)

    def estimate(seed):
        return StateTomography()(circuit, 200, seed, method, backend=backend)

    backend.assert_allclose(estimate(5), estimate(5))
    assert float(backend.max(backend.abs(estimate(5) - estimate(6)))) > 0

    # without a seed, it runs with the generator of the backend
    assert estimate(None).shape == (4, 4)


@pytest.mark.parametrize("method", ["least-squares", "mle"])
def test_call_stopping(backend, method):
    """The iterations stop at the maximum number, or when the change is small."""
    circuit = _prepare(2)

    first = StateTomography()(circuit, 200, 1, method, iterations=1, backend=backend)
    loose = StateTomography()(circuit, 200, 1, method, tolerance=1e3, backend=backend)
    many = StateTomography()(
        circuit, 200, 1, method, iterations=50, tolerance=0, backend=backend
    )

    backend.assert_allclose(loose, first)
    assert float(backend.max(backend.abs(many - first))) > 1e-3


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_circuits(nqubits):
    circuit = Circuit(nqubits)
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.CNOT(0, nqubits - 1) if nqubits > 1 else gates.RY(0, 0.2))
    nprepared = len(circuit.queue)

    circuits = StateTomography().circuits(circuit)

    # the circuit given by the user is not modified
    assert len(circuit.queue) == nprepared
    assert not circuit.measurements

    assert len(circuits) == 3**nqubits

    for setting, tomography_circuit in zip(product("XYZ", repeat=nqubits), circuits):
        queue = tomography_circuit.queue

        # the state preparation is kept as it is
        assert [(gate.name, gate.qubits) for gate in queue[:nprepared]] == [
            (gate.name, gate.qubits) for gate in circuit.queue
        ]

        expected = []
        for qubit, pauli in enumerate(setting):
            if pauli == "Y":
                expected.append(("sdg", (qubit,)))
            if pauli != "Z":
                expected.append(("h", (qubit,)))
        expected.append(("measure", tuple(range(nqubits))))

        assert [(gate.name, gate.qubits) for gate in queue[nprepared:]] == expected


def test_circuits_errors():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with pytest.raises(ValueError):
        StateTomography().circuits(circuit)


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_project_physical(backend, dim):
    """It agrees with the algorithm of Smolin, Gambetta and Smith."""
    rng = np.random.default_rng(dim)
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    matrix = (matrix + matrix.conj().T) / 2
    matrix = matrix - (np.trace(matrix).real - 1) / dim * np.eye(dim)

    # eigenvalues in decreasing order, with the negative ones set to zero, and their
    # weight distributed among the others
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    nonzero, accumulated = dim, 0.0
    while eigenvalues[nonzero - 1] + accumulated / nonzero < 0:
        accumulated += eigenvalues[nonzero - 1]
        nonzero -= 1
    eigenvalues = np.array(
        [
            eigenvalue + accumulated / nonzero if index < nonzero else 0.0
            for index, eigenvalue in enumerate(eigenvalues)
        ]
    )
    expected = (eigenvectors * eigenvalues) @ eigenvectors.conj().T

    projected = _project_physical(
        backend.cast(matrix, dtype=backend.complex128), backend
    )

    assert np.linalg.eigvalsh(matrix)[0] < 0
    backend.assert_allclose(projected, expected, atol=1e-10)
    assert float(backend.eigvalsh(projected)[0]) > -1e-12
    backend.assert_allclose(backend.trace(projected), 1.0, atol=1e-12)

    # a density matrix is not changed
    backend.assert_allclose(
        _project_physical(projected, backend), projected, atol=1e-10
    )


def _frequencies(backend, circuit, nshots, seed):
    """Frequencies of the outcomes of the settings, one setting for each row."""
    backend.set_seed(seed)
    tomography = StateTomography()
    results = tomography.execute(tomography.circuits(circuit), nshots, backend)

    frequencies = np.zeros((3**circuit.nqubits, 2**circuit.nqubits))
    for index, result in enumerate(results):
        for outcome, frequency in result.frequencies(binary=False).items():
            frequencies[index, outcome] = frequency / nshots

    return frequencies


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
        circuit.add(gates.DepolarizingChannel(tuple(range(nqubits)), 0.3))

    return circuit


def _probabilities(state, nqubits):
    """Probabilities of the outcomes of the settings, one setting for each row."""
    hadamard = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    rotations = {"X": hadamard, "Y": hadamard @ np.diag([1, -1j]), "Z": np.eye(2)}

    probabilities = []
    for setting in product("XYZ", repeat=nqubits):
        unitary = reduce(np.kron, [rotations[pauli] for pauli in setting])
        probabilities.append(np.real(np.diag(unitary @ state @ unitary.conj().T)))

    return np.array(probabilities)
