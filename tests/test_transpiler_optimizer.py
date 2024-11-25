import networkx as nx
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.optimizer import Preprocessing, Rearrange


def test_preprocessing_error(star_connectivity):
    circ = Circuit(7)
    preprocesser = Preprocessing(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circ)

    circ = Circuit(5, wire_names=[0, 1, 2, "q3", "q4"])
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circ)


def test_preprocessing_same(star_connectivity):
    circ = Circuit(5)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1


def test_preprocessing_add(star_connectivity):
    circ = Circuit(3)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 5


def test_fusion():
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.Z(0))
    circuit.add(gates.Y(0))
    circuit.add(gates.X(1))
    fusion = Rearrange(max_qubits=1)
    fused_circ = fusion(circuit)
    assert isinstance(fused_circ.queue[0], gates.Unitary)
