import pytest

from qibo import Circuit, gates
from qibo.tomography.abstract import Tomography


class Dummy(Tomography):
    """Minimal concrete protocol."""

    def __call__(self, circuit):
        return circuit

    def circuits(self, circuit):
        return [circuit]


def test_abstract_tomography(backend):
    with pytest.raises(TypeError):
        Tomography()

    protocol = Dummy()

    circuit = Circuit(1)
    circuit.add(gates.X(0))
    circuit.add(gates.M(0))

    with pytest.raises(ValueError):
        protocol._check_circuit(circuit)

    results = protocol.execute([circuit, circuit], nshots=5, backend=backend)
    assert len(results) == 2
    backend.assert_allclose(results[0].samples(binary=False), backend.ones(5))


@pytest.mark.parametrize("nqubits", [1, 2, 3])
def test_pauli_circuits(nqubits):
    circuit = Circuit(nqubits)
    circuit.add(gates.RX(0, 0.3))
    circuit.add(gates.CNOT(0, nqubits - 1) if nqubits > 1 else gates.RY(0, 0.2))
    nprepared = len(circuit.queue)

    observables = [("XYZ" * nqubits)[:nqubits], "I" * nqubits, "Y" * nqubits]
    circuits = Dummy()._pauli_circuits(circuit, observables)

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


def test_pauli_circuits_errors():
    circuit = Circuit(2)
    circuit.add(gates.H(0))

    for observable in ("X", "XYZ", "XA", "xy", 1):
        with pytest.raises(ValueError):
            Dummy()._pauli_circuits(circuit, ["XX", observable])

    circuit.add(gates.M(0))
    with pytest.raises(ValueError):
        Dummy()._pauli_circuits(circuit, ["XX"])
