"""Tests for vanilla state tomography."""

import math
from itertools import product
from types import SimpleNamespace

import pytest

from qibo import Circuit, gates
from qibo.tomography import StateTomography, Tomography


def test_abstract_interface():
    assert issubclass(StateTomography, Tomography)


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


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_call_exact(backend, monkeypatch, nqubits, mixed):
    """With the exact probabilities of the settings, the state is recovered exactly.

    It tests the bases of the settings, the order of the qubits, and the linear
    inversion, apart from the sampling.
    """

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

    estimate = StateTomography()(circuit, nshots=1, backend=backend)

    assert estimate.shape == (2**nqubits, 2**nqubits)
    backend.assert_allclose(estimate, exact, atol=1e-10)


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


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_call_sampling(backend, nqubits):
    circuit = _prepare(nqubits)
    state = backend.execute_circuit(circuit).state()
    exact = backend.outer(state, backend.conj(state))

    estimate = StateTomography()(circuit, nshots=20000, seed=1, backend=backend)

    assert estimate.shape == (2**nqubits, 2**nqubits)
    backend.assert_allclose(estimate, exact, atol=0.03)
    # the estimate is exactly Hermitian with unit trace, whatever the outcomes are
    backend.assert_allclose(estimate, backend.dagger(estimate), atol=1e-12)
    backend.assert_allclose(backend.trace(estimate), 1.0, atol=1e-12)


def test_call_bell_state(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    estimate = StateTomography()(circuit, nshots=20000, seed=3, backend=backend)

    bell = backend.cast([1, 0, 0, 1], dtype=backend.complex128) / math.sqrt(2)
    fidelity = backend.real(backend.conj(bell) @ estimate @ bell)
    backend.assert_allclose(fidelity, 1.0, atol=0.02)


def test_call_not_positive_semidefinite(backend):
    """Linear inversion does not project the estimate onto the physical states."""
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    estimate = StateTomography()(circuit, nshots=100, seed=1, backend=backend)

    assert float(backend.eigvalsh(estimate)[0]) < 0


def test_call_seed(backend):
    circuit = _prepare(2)

    def estimate(seed):
        return StateTomography()(circuit, nshots=200, seed=seed, backend=backend)

    backend.assert_allclose(estimate(5), estimate(5))
    assert float(backend.max(backend.abs(estimate(5) - estimate(6)))) > 0

    # without a seed, it runs with the generator of the backend
    assert estimate(None).shape == (4, 4)


def test_call_errors(backend):
    circuit = _prepare(2)

    for nshots in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError):
            StateTomography()(circuit, nshots=nshots, backend=backend)

    measured = _prepare(2)
    measured.add(gates.M(0, 1))
    with pytest.raises(ValueError):
        StateTomography()(measured, backend=backend)


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
