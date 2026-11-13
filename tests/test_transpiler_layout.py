import copy
import pickle

import networkx as nx
import pytest

from qibo import Circuit, gates
from qibo.transpiler import (
    Passes,
    Preprocessing,
    Random,
    Sabre,
    ShortestPaths,
    TranspileLayout,
    transpile,
)
from qibo.transpiler.asserts import assert_circuit_equivalence


def test_transpile_layout():
    layout = TranspileLayout({"a": 1, "b": 0, "c": 2}, ["a", "b", "c"], 2)
    assert layout == {"a": 1, "b": 0, "c": 2}
    assert isinstance(layout, dict)
    assert type(layout.final_layout) is dict
    assert layout.final_layout == {"a": 1, "b": 0, "c": 2}
    assert layout.initial_layout == ["a", "b", "c"]
    assert layout.nqubits_input == 2
    assert layout.routing_permutation == [1, 0, 2]

    initial_layout = ["b", "a", "c"]
    layout = TranspileLayout({"a": 1, "b": 0, "c": 2}, initial_layout, 3)
    initial_layout.append("d")
    assert layout.initial_layout == ["b", "a", "c"]
    assert layout.routing_permutation == [0, 1, 2]


def test_transpile_layout_logical_layouts_without_auxiliary_qubits():
    layout = TranspileLayout({"a": 1, "b": 0}, ["b", "a"], 2)
    assert layout.auxiliary_qubits == []
    assert layout.initial_logical_layout(filter_auxiliary=True) == {0: "b", 1: "a"}
    assert layout.final_logical_layout(filter_auxiliary=True) == {0: 0, 1: 1}


def test_transpile_layout_copy():
    layout = TranspileLayout({0: 1, 1: 0}, [0, 1], 2)
    for new in (
        copy.copy(layout),
        copy.deepcopy(layout),
        pickle.loads(pickle.dumps(layout)),
    ):
        assert new == layout
        assert new.initial_layout == [0, 1]
        assert new.nqubits_input == 2


@pytest.mark.parametrize("router", [Sabre, ShortestPaths])
def test_passes_transpile_layout(backend, router):
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(1, 2))
    connectivity = nx.path_graph(4)
    pipeline = Passes(
        [Preprocessing(), router(seed=1)],
        connectivity=connectivity,
    )
    transpiled, layout = pipeline(circuit, backend=backend)
    assert isinstance(layout, TranspileLayout)
    assert layout.nqubits_input == 3
    assert layout.initial_layout == [0, 1, 2, 3]
    assert sorted(layout.routing_permutation) == [0, 1, 2, 3]
    assert layout.routing_permutation == list(layout.values())
    assert transpiled.nqubits == 4


def test_passes_transpile_layout_initial_layout(backend):
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    connectivity = nx.path_graph(4)
    pipeline = Passes(
        [Preprocessing(initial_layout=[2, 3]), ShortestPaths(seed=1)],
        connectivity=connectivity,
    )
    _, layout = pipeline(circuit, backend=backend)
    assert layout.nqubits_input == 2
    assert layout.initial_layout[:2] == [2, 3]
    assert sorted(layout.initial_layout) == [0, 1, 2, 3]
    assert set(layout) == set(layout.initial_layout)


def test_passes_transpile_layout_placer(backend):
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(0, 2))
    pipeline = Passes(
        [Preprocessing(), Random(seed=2), ShortestPaths(seed=1)],
        connectivity=nx.path_graph(3),
    )
    transpiled, layout = pipeline(circuit, backend=backend)
    assert sorted(layout.initial_layout) == [0, 1, 2]
    assert layout.routing_permutation == [
        layout[node] for node in layout.initial_layout
    ]
    placed = Circuit(**{**circuit.init_kwargs, "wire_names": layout.initial_layout})
    placed.add(circuit.queue)
    assert_circuit_equivalence(placed, transpiled, layout)


def test_passes_without_router_has_identity_layout(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    pipeline = Passes([Preprocessing()], connectivity=nx.path_graph(4))
    transpiled, layout = pipeline(circuit, backend=backend)
    assert isinstance(layout, TranspileLayout)
    assert layout == {0: 0, 1: 1, 2: 2, 3: 3}
    assert layout.initial_layout == [0, 1, 2, 3]
    assert layout.nqubits_input == 2
    assert layout.routing_permutation == [0, 1, 2, 3]
    assert transpiled.nqubits == 4


def test_passes_without_router_initial_layout(backend):
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    pipeline = Passes(
        [Preprocessing(initial_layout=[3, 1])], connectivity=nx.path_graph(4)
    )
    _, layout = pipeline(circuit, backend=backend)
    assert layout.initial_layout[:2] == [3, 1]
    assert sorted(layout.initial_layout) == [0, 1, 2, 3]
    assert layout.routing_permutation == [0, 1, 2, 3]
    assert layout == {name: index for index, name in enumerate(layout.initial_layout)}


def test_passes_without_connectivity_has_identity_layout(backend):
    circuit = Circuit(3, wire_names=["a", "b", "c"])
    circuit.add(gates.H(0))
    transpiled, layout = Passes()(circuit, backend=backend)
    assert transpiled is circuit
    assert layout == {"a": 0, "b": 1, "c": 2}
    assert layout.initial_layout == ["a", "b", "c"]
    assert layout.nqubits_input == 3
    assert layout.routing_permutation == [0, 1, 2]


def test_transpile_returns_transpile_layout(backend):
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 2))
    connectivity = nx.path_graph(3)
    _, layout = transpile(circuit, connectivity, seed=1, backend=backend)
    assert isinstance(layout, TranspileLayout)
    assert layout.nqubits_input == 3
    results = transpile([circuit, circuit], connectivity, seed=1, backend=backend)
    assert all(isinstance(layout, TranspileLayout) for _, layout in results)
