import pytest

from qibo import Circuit, gates


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
