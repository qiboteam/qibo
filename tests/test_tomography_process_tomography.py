"""Tests for vanilla process tomography."""

import math
from itertools import product
from types import SimpleNamespace

import pytest

from qibo import Circuit, gates
from qibo.quantum_info import (
    kraus_to_choi,
    kraus_to_liouville,
    kraus_to_pauli,
    pauli_basis,
)
from qibo.tomography import ProcessTomography, Tomography

AMPLITUDE_DAMPING = 0.3


@pytest.fixture
def exact_probabilities(monkeypatch):
    """Executes the circuits with their exact probabilities, as one shot frequencies."""

    def execute(self, circuits, nshots=1000, backend=None):
        results = []
        for circuit in circuits:
            without_measurements = circuit.__class__(**circuit.init_kwargs)
            without_measurements.add(circuit.queue[:-1])
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


def test_abstract_interface():
    assert issubclass(ProcessTomography, Tomography)


@pytest.mark.parametrize("density_matrix", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_circuits(nqubits, density_matrix):
    circuit = Circuit(nqubits, density_matrix=density_matrix)
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.CNOT(0, nqubits - 1) if nqubits > 1 else gates.RY(0, 0.2))
    nprocess = len(circuit.queue)

    circuits = ProcessTomography().circuits(circuit)

    # the circuit given by the user is not modified
    assert len(circuit.queue) == nprocess
    assert not circuit.measurements

    assert len(circuits) == 12**nqubits
    assert all(
        tomography_circuit.density_matrix == density_matrix
        for tomography_circuit in circuits
    )

    # input states change the slowest, and each one has the 3^n settings
    expected = []
    for inputs in product(range(4), repeat=nqubits):
        preparation = []
        for qubit, state in enumerate(inputs):
            if state == 1:
                preparation.append(("x", (qubit,)))
            if state >= 2:
                preparation.append(("h", (qubit,)))
            if state == 3:
                preparation.append(("s", (qubit,)))
        for setting in product("XYZ", repeat=nqubits):
            rotation = []
            for qubit, pauli in enumerate(setting):
                if pauli == "Y":
                    rotation.append(("sdg", (qubit,)))
                if pauli != "Z":
                    rotation.append(("h", (qubit,)))
            expected.append(
                preparation
                + [(gate.name, gate.qubits) for gate in circuit.queue]
                + rotation
                + [("measure", tuple(range(nqubits)))]
            )

    assert [
        [(gate.name, gate.qubits) for gate in tomography_circuit.queue]
        for tomography_circuit in circuits
    ] == expected


def test_circuits_errors():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))

    with pytest.raises(ValueError):
        ProcessTomography().circuits(circuit)


@pytest.mark.parametrize(
    "process,nqubits",
    [("unitary", 1), ("unitary", 2), ("unitary", 3), ("channel", 1), ("channel", 2)],
)
def test_call_exact(backend, exact_probabilities, process, nqubits):
    """With the exact probabilities of the circuits, the process is recovered exactly.

    It tests the input states, the order of the qubits, the state tomography of the
    outputs, and the linear inversion, apart from the sampling.
    """
    circuit, kraus = _process(backend, process, nqubits)

    estimate = ProcessTomography()(circuit, nshots=1, backend=backend)

    assert estimate.shape == (4**nqubits, 4**nqubits)
    backend.assert_allclose(
        estimate, _transfer_matrix(backend, kraus, nqubits), atol=1e-10
    )
    # the process preserves the trace
    backend.assert_allclose(estimate[0], backend.identity(4**nqubits)[0], atol=1e-10)


@pytest.mark.parametrize("process,nqubits", [("unitary", 2), ("channel", 1)])
def test_call_representations(backend, exact_probabilities, process, nqubits):
    """The representations are those of ``qibo.quantum_info``."""
    circuit, kraus = _process(backend, process, nqubits)
    operators = [(tuple(range(nqubits)), operator) for operator in kraus]

    for representation, convert in (
        ("pauli", lambda ops: kraus_to_pauli(ops, normalize=True, backend=backend)),
        ("liouville", lambda ops: kraus_to_liouville(ops, backend=backend)),
        ("choi", lambda ops: kraus_to_choi(ops, backend=backend)),
    ):
        estimate = ProcessTomography()(
            circuit, nshots=1, representation=representation, backend=backend
        )

        backend.assert_allclose(estimate, convert(operators), atol=1e-10)


@pytest.mark.parametrize(
    "nqubits,nshots,atol", [(1, 20000, 0.05), (2, 5000, 0.12), (3, 20000, 0.12)]
)
def test_call_sampling(backend, nqubits, nshots, atol):
    circuit, kraus = _process(backend, "unitary", nqubits)

    estimate = ProcessTomography()(circuit, nshots=nshots, seed=1, backend=backend)

    backend.assert_allclose(
        estimate, _transfer_matrix(backend, kraus, nqubits), atol=atol
    )
    # the trace is preserved whatever the outcomes are
    backend.assert_allclose(estimate[0], backend.identity(4**nqubits)[0], atol=1e-12)


def test_call_not_completely_positive(backend):
    """Linear inversion does not project the estimate onto the physical processes."""
    circuit, _ = _process(backend, "unitary", 1)

    choi = ProcessTomography()(
        circuit, nshots=100, representation="choi", seed=1, backend=backend
    )

    assert float(backend.eigvalsh((choi + backend.dagger(choi)) / 2)[0]) < 0


def test_call_executes_circuits(backend, monkeypatch):
    """The circuits that are executed are those generated, with their number of shots."""
    executed = []
    execute = Tomography.execute

    def recorder(self, circuits, nshots=1000, backend=None):
        executed.append((circuits, nshots))
        return execute(self, circuits, nshots, backend)

    monkeypatch.setattr(Tomography, "execute", recorder)

    circuit, _ = _process(backend, "unitary", 2)
    ProcessTomography()(circuit, nshots=50, seed=1, backend=backend)

    def description(circuits):
        return [[(gate.name, gate.qubits) for gate in c.queue] for c in circuits]

    assert {nshots for _, nshots in executed} == {50}
    assert description(
        [c for circuits, _ in executed for c in circuits]
    ) == description(ProcessTomography().circuits(circuit))


def test_call_seed(backend):
    circuit, _ = _process(backend, "unitary", 1)

    def estimate(seed):
        return ProcessTomography()(circuit, nshots=200, seed=seed, backend=backend)

    backend.assert_allclose(estimate(5), estimate(5))
    assert float(backend.max(backend.abs(estimate(5) - estimate(6)))) > 0

    # without a seed, it runs with the generator of the backend
    assert estimate(None).shape == (4, 4)


def test_call_errors(backend):
    circuit, _ = _process(backend, "unitary", 1)

    for nshots in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError):
            ProcessTomography()(circuit, nshots=nshots, backend=backend)

    for representation in ("chi", "kraus", "PAULI", None):
        with pytest.raises(ValueError):
            ProcessTomography()(circuit, representation=representation, backend=backend)

    measured = Circuit(1)
    measured.add(gates.H(0))
    measured.add(gates.M(0))
    with pytest.raises(ValueError):
        ProcessTomography()(measured, backend=backend)


def _process(backend, process, nqubits):
    """Circuit of a process, with real and imaginary parts, and its Kraus operators.

    A ``"channel"`` is a unitary followed by an amplitude damping on the last qubit,
    in a circuit of density matrices.
    """
    circuit = Circuit(nqubits, density_matrix=process == "channel")
    for qubit in range(nqubits):
        circuit.add(gates.RX(qubit, 0.4 + 0.5 * qubit))
        circuit.add(gates.RY(qubit, 1.1 - 0.3 * qubit))
    for qubit in range(nqubits - 1):
        circuit.add(gates.CNOT(qubit, qubit + 1))
    unitary = circuit.unitary(backend)

    if process == "unitary":
        return circuit, [unitary]

    damping = [
        backend.cast(
            [[1, 0], [0, math.sqrt(1 - AMPLITUDE_DAMPING)]], dtype=backend.complex128
        ),
        backend.cast(
            [[0, math.sqrt(AMPLITUDE_DAMPING)], [0, 0]], dtype=backend.complex128
        ),
    ]
    circuit.add(gates.KrausChannel(nqubits - 1, damping))

    identity = backend.identity(2 ** (nqubits - 1))
    return circuit, [backend.kron(identity, operator) @ unitary for operator in damping]


def _transfer_matrix(backend, kraus, nqubits):
    """Pauli transfer matrix from its definition, with the Kraus operators."""
    dim = 2**nqubits
    paulis = pauli_basis(nqubits, backend=backend)

    return backend.cast(
        [
            [
                float(
                    backend.real(
                        backend.trace(
                            pauli_i
                            @ sum(
                                operator @ pauli_j @ backend.dagger(operator)
                                for operator in kraus
                            )
                        )
                    )
                )
                / dim
                for pauli_j in paulis
            ]
            for pauli_i in paulis
        ],
        dtype=backend.float64,
    )
