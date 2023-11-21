import networkx as nx
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.optimizer import Preprocessing, Rearrange


def star_connectivity():
    Q = ["q" + str(i) for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


def test_preprocessing_error():
    circ = Circuit(7)
    preprocesser = Preprocessing(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circ)


def test_preprocessing_same():
    circ = Circuit(5)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1


def test_preprocessing_add():
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
