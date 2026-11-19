import math
from collections import Counter

import networkx as nx
import pytest

from qibo import Circuit, gates
from qibo.transpiler import (
    HardwareModel,
    NoiseAwarePlacer,
    NoiseAwarePostLayout,
    Passes,
    Preprocessing,
)
from qibo.transpiler._exceptions import ConnectivityError, TranspilerPipelineError
from qibo.transpiler.abstract import PostLayout
from qibo.transpiler.asserts import assert_transpiling
from qibo.transpiler.layout import TranspileLayout
from qibo.transpiler.optimizer import GateDirection
from qibo.transpiler.pipeline import generate_preset_passes, transpile
from qibo.transpiler.placer import (
    Dense,
    ReverseTraversal,
    _connectivity_weights,
    _constrained_order,
    _LayoutMatcher,
)
from qibo.transpiler.unroller import NativeGates


class Identity(PostLayout):
    """Post layout that changes nothing, and that records how it is used."""

    def __init__(self, connectivity=None, skip=False):
        self.connectivity = connectivity
        self.calls = 0
        self.placers = []
        self.skipping = skip

    def __call__(self, circuit, layout):
        self.calls += 1
        return layout

    def skip(self, placer):
        self.placers.append(placer)
        return self.skipping


def test_connectivity_weights():
    hardware = HardwareModel(
        (0, 1, 2),
        [(0, 1), (1, 2)],
        errors={"2q": {(0, 1): 0.1, (1, 2): 0.2, (2, 1): 0.4}},
        symmetric={(0, 1): False},
    )
    weights = hardware.weighted_connectivity()
    connectivity = nx.path_graph(3)
    first, second, third = (_weight(error) for error in (0.1, 0.2, 0.4))

    # the mean of the directions that a connection has
    mean = _connectivity_weights(connectivity, weights)
    assert mean[(0, 1)] == mean[(1, 0)] == pytest.approx(first)
    assert mean[(1, 2)] == mean[(2, 1)] == pytest.approx((second + third) / 2)

    # each direction that it has, and not the ones that it does not have
    directed = _connectivity_weights(connectivity, weights, directed=True)
    assert directed == pytest.approx({(0, 1): first, (1, 2): second, (2, 1): third})


@pytest.mark.parametrize("connectivity", [nx.path_graph(7), nx.cycle_graph(3)])
def test_connectivity_weights_error(connectivity):
    # a qubit that the hardware does not have, and a coupling that it does not have
    with pytest.raises(ConnectivityError):
        _connectivity_weights(connectivity, _line().weighted_connectivity())


def test_constrained_order():
    # the qubit with most gates first, then the one with most gates to the placed ones
    pairs = Counter({(0, 1): 1, (1, 2): 3, (2, 3): 1})
    ordered = _constrained_order(pairs)
    assert list(ordered.nodes) == [1, 2, 0, 3]
    assert set(map(frozenset, ordered.edges)) == set(map(frozenset, pairs))
    # a graph that is not connected goes on with the next group of qubits
    separate = _constrained_order(Counter({(0, 1): 1, (2, 3): 1}))
    assert list(separate.nodes) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "pairs, expected",
    [
        # a chain starts at its end, and not inside of it, whatever the number of gates is
        (Counter({(index, index + 1): 1 for index in range(5)}), [0, 1, 2, 3, 4, 5]),
        (Counter({(index, index + 1): 3 for index in range(5)}), [0, 1, 2, 3, 4, 5]),
        (Counter({(3, 5): 2}), [3, 5]),
        # a star starts at a leaf, and goes on with the center, which has gates with it
        (Counter({(0, qubit): 1 for qubit in range(1, 5)}), [1, 0, 2, 3, 4]),
        (Counter({(index, (index + 1) % 5): 1 for index in range(5)}), [0, 1, 2, 3, 4]),
        # each group of qubits starts at the end of its chain
        (Counter({(0, 1): 1, (1, 2): 1, (3, 4): 1, (4, 5): 1}), [0, 1, 2, 3, 4, 5]),
        # with pairs that have more gates, they go first, as in the other tests
        (Counter({(0, 1): 1, (1, 2): 1, (2, 3): 2, (3, 4): 1}), [2, 3, 1, 0, 4]),
        (Counter({(0, 1): 1, (1, 2): 2, (2, 3): 1, (3, 4): 2}), [1, 2, 3, 4, 0]),
    ],
)
def test_constrained_order_chains(pairs, expected):
    ordered = _constrained_order(pairs)
    assert list(ordered.nodes) == expected
    assert set(map(frozenset, ordered.edges)) == set(map(frozenset, pairs))


@pytest.mark.parametrize("side, chains", [(5, (25,)), (6, (17, 17))])
def test_noise_aware_placer_places_chains_on_a_grid(side, chains):
    """The chains are placed on a grid with few feasibility checks, because the search starts
    at their ends. Inside of them, it takes many thousands of checks, or it does not end."""
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    couplings = sorted(tuple(sorted(edge)) for edge in grid.edges)
    hardware = HardwareModel(tuple(range(side**2)), couplings)
    pairs, first = [], 0
    for length in chains:
        pairs += [(first + index, first + index + 1) for index in range(length - 1)]
        first += length
    circuit = _circuit(side**2, pairs)

    placer = NoiseAwarePlacer(hardware, call_limit=(1000, 1000))
    placer(circuit)
    assert placer.stop_reason == "optimal"
    names = circuit.wire_names
    assert all(grid.has_edge(names[first], names[second]) for first, second in pairs)


def test_layout_matcher_directed_cost():
    device, pattern = nx.path_graph(2), nx.Graph([(0, 1)])
    node_cost = {qubit: {0: 0.0, 1: 0.5} for qubit in (0, 1)}
    edge_weight = {(0, 1): 0.1, (1, 0): 0.1}
    # a gate of the first qubit on the first node and the second qubit on the second one
    matcher = _LayoutMatcher(
        device,
        pattern,
        100,
        node_cost,
        edge_weight,
        {},
        directed={(0, 1): 2},
        directed_weight={(0, 1): 0.3},
    )
    assert matcher.cost({0: 0, 1: 1}) == pytest.approx(0.5 + 2 * 0.3)
    # the coupling does not work in the other direction, so the gate costs infinity
    assert matcher.cost({0: 1, 1: 0}) == math.inf


def test_layout_matcher_limits():
    # the device and the pattern are the same coupling, so a check is feasible if it is allowed
    device, pattern = nx.path_graph(2), nx.Graph([(0, 1)])
    node_cost = {qubit: {0: 0.0, 1: 0.0} for qubit in (0, 1)}
    edge_weight, counts = {(0, 1): 0.1, (1, 0): 0.1}, {frozenset((0, 1)): 1}

    # a limit before the first layout, and a limit of the checks after it
    matcher = _LayoutMatcher(device, pattern, (5, 3), node_cost, edge_weight, counts)
    assert _checks(matcher, 5) == [True] * 5
    assert not matcher.limited
    matcher.best = 1.0
    assert _checks(matcher, 4) == [True] * 3 + [False]
    assert matcher.limited
    assert matcher.best == 1.0

    matcher = _LayoutMatcher(device, pattern, (5, 3), node_cost, edge_weight, counts)
    assert _checks(matcher, 6) == [True] * 5 + [False]
    assert matcher.limited

    # a single limit is the same before and after a layout
    for best in (False, True):
        matcher = _LayoutMatcher(device, pattern, 2, node_cost, edge_weight, counts)
        matcher.best = 1.0 if best else matcher.best
        assert _checks(matcher, 3) == [True, True, False]


def test_noise_aware_placer_dense():
    circuit = _circuit(6, [(0, 1), (1, 2)])
    placer = NoiseAwarePlacer(_line(), method="dense")
    placer(circuit)
    assert sorted(circuit.wire_names[:3]) == [3, 4, 5]
    assert placer.stop_reason == "dense"

    # without errors there is no cost, and it places the qubits as the dense placer
    plain = HardwareModel(tuple(range(6)), [(i, i + 1) for i in range(5)])
    noise_aware, dense = _circuit(6, [(0, 1), (1, 2)]), _circuit(6, [(0, 1), (1, 2)])
    NoiseAwarePlacer(plain, method="dense")(noise_aware)
    Dense(plain.connectivity)(dense)
    assert noise_aware.wire_names == dense.wire_names


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "other"},
        {"max_matches": 0},
        {"call_limit": 0},
        {"call_limit": ()},
        {"call_limit": (1, 2, 3)},
        {"call_limit": (0, 5)},
        {"call_limit": (5, 0)},
    ],
)
def test_noise_aware_placer_error(kwargs):
    with pytest.raises(ValueError):
        NoiseAwarePlacer(_line(), **kwargs)


@pytest.mark.parametrize("method", ["vf2", "dense"])
def test_noise_aware_placer_error_three_qubit_gate(method):
    circuit = Circuit(6)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ValueError, match="more than 2 qubits"):
        NoiseAwarePlacer(_line(), method=method)(circuit)


def test_noise_aware_placer_free_qubits():
    # the qubit that is only measured takes the free node with the best readout
    readout = {0: 0.30, 1: 0.02, 2: 0.03, 3: 0.02, 4: 0.01, 5: 0.02}
    hardware = _line((0.01,) * 5, errors={"readout": readout})
    for method in ("vf2", "dense"):
        circuit = _circuit(6, [(0, 1)], measured=(0, 1, 2))
        NoiseAwarePlacer(hardware, method=method)(circuit)
        used = set(circuit.wire_names[:2])
        free = {node: error for node, error in readout.items() if node not in used}
        assert readout[circuit.wire_names[2]] == min(free.values())
        assert sorted(circuit.wire_names) == list(range(6))

    # the qubits with operations choose the nodes together, with the unique assignment of
    # the lowest total cost, and the ones without operations take what is left in order
    readout = {0: 0.30, 1: 0.02, 2: 0.03, 3: 0.02, 4: 0.02, 5: 0.01}
    single = {0: 0.01, 1: 0.001, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.05}
    hardware = _line(errors={"readout": readout, "1q": single})
    circuit = _circuit(6, [(0, 1)], measured=(2, 3))
    for _ in range(3):
        circuit.add(gates.H(3))
    NoiseAwarePlacer(hardware)(circuit)
    assert sorted(circuit.wire_names[:2]) == [3, 4]
    # the qubit with three gates avoids the node with a poor single-qubit gate
    assert circuit.wire_names[2:4] == [5, 1]
    assert circuit.wire_names[4:] == [0, 2]


def test_noise_aware_placer_infinite_costs():
    # a readout that always fails, and couplings that always fail, cost infinity
    broken = _line((1.0,) * 5, errors={"readout": {node: 1.0 for node in range(6)}})
    circuit = _circuit(6, [(0, 1)], measured=(0, 1, 2))
    placer = NoiseAwarePlacer(broken)
    placer(circuit)
    assert placer.stop_reason == "none exists"
    assert sorted(circuit.wire_names) == list(range(6))

    # only the free qubit has an infinite cost, which does not make the assignment fail
    hardware = _line(errors={"readout": {node: 1.0 for node in range(6)}})
    circuit = _circuit(6, [(0, 1)], measured=(2,))
    NoiseAwarePlacer(hardware)(circuit)
    assert sorted(circuit.wire_names) == list(range(6))


def test_noise_aware_placer_keep_layout():
    hardware = _line()
    circuit = _circuit(6, [(0, 1), (1, 2)])
    placer = NoiseAwarePlacer(hardware, keep_layout=True)
    placer(circuit)
    assert placer.stop_reason == "kept"
    assert circuit.wire_names == list(range(6))

    # without it, the circuit is moved to the quietest qubits
    moved = _circuit(6, [(0, 1), (1, 2)])
    NoiseAwarePlacer(hardware)(moved)
    assert sorted(moved.wire_names[:3]) == [3, 4, 5]

    # a layout that needs SWAP gates is not kept
    triangle = _circuit(6, [(0, 1), (1, 2), (0, 2)])
    placer = NoiseAwarePlacer(hardware, keep_layout=True)
    placer(triangle)
    assert placer.stop_reason == "none exists"


def test_noise_aware_placer_limits():
    hardware = _line()
    chain = _circuit(6, [(0, 1), (1, 2), (2, 3), (3, 4)])
    # a limit of feasibility checks before the first layout
    placer = NoiseAwarePlacer(hardware, call_limit=3)
    placer(chain.copy())
    assert placer.stop_reason == "none found"

    # a limit of layouts, and a limit after the first layout
    for kwargs in ({"max_matches": 1}, {"call_limit": (10**6, 1)}):
        placer = NoiseAwarePlacer(hardware, **kwargs)
        circuit = _circuit(6, [(0, 1)])
        placer(circuit)
        assert placer.stop_reason == "limited"
        assert sorted(circuit.wire_names) == list(range(6))

    # the second limit does not change a search that is shorter
    reference, circuit = chain.copy(), chain.copy()
    NoiseAwarePlacer(hardware)(reference)
    placer = NoiseAwarePlacer(hardware, call_limit=(10**6, 10**6))
    placer(circuit)
    assert circuit.wire_names == reference.wire_names
    assert placer.stop_reason == "optimal"


def test_noise_aware_placer_no_errors():
    # without errors there is no cost, so the first layout of the search is the best one
    hardware = HardwareModel(tuple(range(4)), [(0, 1), (1, 2), (2, 3)])
    circuit = _circuit(4, [(0, 1), (1, 2)])
    placer = NoiseAwarePlacer(hardware)
    placer(circuit)
    assert placer.stop_reason == "optimal"
    assert all(
        hardware.connectivity.has_edge(circuit.wire_names[a], circuit.wire_names[b])
        for a, b in ((0, 1), (1, 2))
    )


def test_noise_aware_placer_no_gates():
    circuit = Circuit(6)
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    placer = NoiseAwarePlacer(_line(errors={"readout": {0: 0.3, 3: 0.01}}))
    placer(circuit)
    assert placer.stop_reason == "no two-qubit gates"
    # the measured qubit goes to the node with the best readout
    assert circuit.wire_names[0] == 3
    assert sorted(circuit.wire_names) == list(range(6))


def test_noise_aware_placer_strict_direction():
    natives = NativeGates.default() | NativeGates.CNOT
    couplings = {(0, 1): False, (1, 2): False, (2, 3): False}
    hardware = _line((0.05, 0.05, 0.001), native_gates=natives, symmetric=couplings)
    gate_pairs = [(0, 1), (0, 1), (1, 2)]

    # the cheapest layout puts the busiest pair on the quiet coupling, against its direction
    circuit = _circuit(4, gate_pairs)
    NoiseAwarePlacer(hardware)(circuit)
    assert circuit.wire_names == [3, 2, 1, 0]
    strict, circuit = (
        NoiseAwarePlacer(hardware, strict_direction=True),
        _circuit(4, gate_pairs),
    )
    strict(circuit)
    assert circuit.wire_names == [1, 2, 3, 0]
    assert strict.stop_reason == "optimal"

    # no layout respects the two directions of a coupling
    strict(_circuit(4, [(0, 1), (1, 0)]))
    assert strict.stop_reason == "none exists"

    # the gates that are symmetric, and the ones that are not native, are not constrained
    circuit = _circuit(4, gate_pairs, gate=gates.CZ)
    strict(circuit)
    assert circuit.wire_names == [3, 2, 1, 0]
    non_native = _line((0.05, 0.05, 0.001), symmetric=couplings)
    circuit = _circuit(4, gate_pairs)
    NoiseAwarePlacer(non_native, strict_direction=True)(circuit)
    assert circuit.wire_names == [3, 2, 1, 0]

    # nor is a connectivity that is not directed
    undirected = _line((0.05, 0.05, 0.001), native_gates=natives)
    circuit = _circuit(4, gate_pairs)
    NoiseAwarePlacer(undirected, strict_direction=True)(circuit)
    assert circuit.wire_names == [3, 2, 1, 0]


def test_noise_aware_placer_strict_direction_keep_and_initial():
    natives = NativeGates.default() | NativeGates.CNOT
    hardware = _line(
        (0.05, 0.05, 0.001),
        native_gates=natives,
        symmetric={(0, 1): False, (1, 2): False, (2, 3): False},
    )
    # the layout is kept if the gates follow the couplings, and searched if they do not
    for pairs, reason in (([(0, 1), (1, 2)], "kept"), ([(1, 0), (2, 1)], "optimal")):
        placer = NoiseAwarePlacer(hardware, keep_layout=True, strict_direction=True)
        placer(_circuit(4, pairs))
        assert placer.stop_reason == reason
    # without strict direction it is kept, because the nodes are connected
    placer = NoiseAwarePlacer(hardware, keep_layout=True)
    placer(_circuit(4, [(1, 0), (2, 1)]))
    assert placer.stop_reason == "kept"

    # an initial layout against the direction is not valid, and the best one is kept
    placer = NoiseAwarePlacer(hardware, strict_direction=True)
    placer.connectivity = hardware.connectivity
    circuit = _circuit(4, [(0, 1), (1, 2)])
    best = {0: 1, 1: 2, 2: 3}
    for initial in ({0: 3, 1: 2, 2: 1}, {0: 0, 1: 1, 2: 2}, best):
        assert placer._search(circuit, initial=initial) == best


def test_noise_aware_placer_vf2():
    circuit = _circuit(6, [(0, 1), (1, 2)])
    placer = NoiseAwarePlacer(_line())
    assert placer.stop_reason is None
    placer(circuit)
    assert sorted(circuit.wire_names[:3]) == [3, 4, 5]
    assert sorted(circuit.wire_names) == list(range(6))
    assert placer.stop_reason == "optimal"

    # a circuit that needs SWAP gates falls back to the dense placement
    triangle = _circuit(6, [(0, 1), (1, 2), (0, 2)])
    placer(triangle)
    assert placer.stop_reason == "none exists"
    region = triangle.wire_names[:3]
    assert nx.is_connected(_line().connectivity.subgraph(region))


def test_noise_aware_placer_vf2_pruned_search():
    # the search is exact, so it finds the quietest chain among the many of a large device
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4))
    couplings = list(graph.edges)
    errors = dict.fromkeys(couplings, 0.05) | {couplings[7]: 0.001, couplings[8]: 0.002}
    hardware = HardwareModel(tuple(graph.nodes), couplings, errors={"2q": errors})
    weights = hardware.weighted_connectivity()
    circuit = _circuit(12, [(0, 1), (1, 2)])
    placer = NoiseAwarePlacer(hardware)
    placer(circuit)
    assert placer.stop_reason == "optimal"

    placed = [circuit.wire_names[qubit] for qubit in range(3)]
    cost = weights.edges[placed[0], placed[1]]["weight"]
    cost += weights.edges[placed[1], placed[2]]["weight"]
    best = min(
        weights.edges[first, middle]["weight"] + weights.edges[middle, last]["weight"]
        for middle in graph.nodes
        for first in graph[middle]
        for last in graph[middle]
        if first != last
    )
    assert cost == pytest.approx(best)


def test_noise_aware_post_layout():
    hardware = _line()
    circuit = _circuit(3, [(0, 1), (1, 2), (0, 2)])
    circuit.add(gates.M(0, 1, 2))
    plain, moved = Identity(), NoiseAwarePostLayout(hardware)
    results = []
    for post in (plain, moved):
        transpiled, layout = generate_preset_passes(
            connectivity=hardware.connectivity, post_layout=post, seed=1
        )(circuit)
        results.append((transpiled, layout))
        assert_transpiling(circuit, transpiled, hardware.connectivity, layout)

    (before, _), (after, layout) = results
    # only the labels change: the same gates act on the same qubit indices
    assert [(g.name, g.qubits) for g in after.queue] == [
        (g.name, g.qubits) for g in before.queue
    ]
    used = {after.wire_names[q] for g in after.queue for q in g.qubits}
    assert used == {3, 4, 5}
    assert {before.wire_names[q] for g in before.queue for q in g.qubits} == {0, 1, 2}
    # the layout of the router has the new labels
    assert list(layout.initial_layout) == after.wire_names
    assert set(layout) == set(after.wire_names)
    assert moved.stop_reason == "optimal"


def test_noise_aware_post_layout_displaced_qubits():
    hardware = _line()
    circuit = Circuit(6, wire_names=list(range(6)))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.H(3))
    circuit.add(gates.M(3))
    layout = TranspileLayout({node: node for node in range(6)}, list(range(6)), 6)
    post = NoiseAwarePostLayout(hardware)

    new = post(circuit, layout)
    # the pair moves to the quietest coupling, the qubit that is only measured does not
    # move, the idle qubit keeps its label, and the ones that lose theirs take the free nodes
    assert sorted(circuit.wire_names[:2]) == [4, 5]
    assert circuit.wire_names[2:] == [2, 3, 0, 1]
    assert list(new.initial_layout) == circuit.wire_names
    assert new.routing_permutation == layout.routing_permutation
    assert post.stop_reason == "optimal"


def test_noise_aware_post_layout_error():
    for kwargs in ({"max_matches": 0}, {"call_limit": (3, 0)}):
        with pytest.raises(ValueError):
            NoiseAwarePostLayout(_line(), **kwargs)


def test_noise_aware_post_layout_no_change():
    hardware = _line()
    layout = TranspileLayout({node: node for node in range(6)}, list(range(6)), 6)
    post = NoiseAwarePostLayout(hardware)

    # a circuit that is already on the quietest qubits
    circuit = Circuit(6, wire_names=[3, 4, 5, 0, 1, 2])
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    assert post(circuit, layout) is layout
    assert circuit.wire_names == [3, 4, 5, 0, 1, 2]

    # a circuit without two-qubit gates has nothing to move
    single = Circuit(6, wire_names=list(range(6)))
    single.add(gates.H(0))
    assert post(single, layout) is layout
    assert post.stop_reason == "no two-qubit gates"


def test_noise_aware_post_layout_skip():
    hardware = _line()
    circuit = _circuit(6, [(0, 1), (1, 2)])
    post, trusting = (
        NoiseAwarePostLayout(hardware),
        NoiseAwarePostLayout(hardware, trust_placer=True),
    )
    assert not post.skip(None)

    optimal = NoiseAwarePlacer(hardware)
    optimal(circuit.copy())
    assert optimal.stop_reason == "optimal"
    assert not post.skip(optimal)
    assert trusting.skip(optimal)
    assert trusting.stop_reason == "skipped"

    # another hardware model, a search that a limit ended, and other placers are not trusted
    other = NoiseAwarePlacer(_line())
    other(circuit.copy())
    assert not trusting.skip(other)
    limited = NoiseAwarePlacer(hardware, max_matches=1)
    limited(circuit.copy())
    assert not trusting.skip(limited)
    assert not trusting.skip(NoiseAwarePlacer(hardware, method="dense"))
    assert not trusting.skip(Dense(hardware.connectivity))

    # a placer that kept the layout on purpose is always respected
    kept = NoiseAwarePlacer(hardware, keep_layout=True)
    kept(circuit.copy())
    assert kept.stop_reason == "kept"
    assert post.skip(kept)
    assert post.stop_reason == "skipped"


def test_post_layout_abstract():
    post = Identity()
    assert post.connectivity is None
    post.connectivity = nx.DiGraph([(0, 1)])
    assert not post.connectivity.is_directed()
    assert post.connectivity.has_edge(1, 0)
    assert not PostLayout.skip(post, None)


def test_post_layout_in_passes():
    hardware = _line()
    placer, called = NoiseAwarePlacer(hardware), []
    for skip in (True, False):
        post = Identity(skip=skip)
        called.clear()
        passes = generate_preset_passes(
            connectivity=hardware.connectivity,
            layout_method=placer,
            post_layout=post,
            seed=1,
            callback=lambda transpiler_pass, **kwargs: called.append(transpiler_pass),
        )
        passes(_circuit(6, [(0, 1), (1, 2), (0, 2)]))
        # it is asked with the last placer, and it is only run and reported if it does not skip
        assert post.placers == [placer]
        assert post.calls == int(not skip)
        assert (post in called) == (not skip)


@pytest.mark.parametrize(
    "pairs, decider",
    [
        # the layout is kept
        ([(0, 1), (1, 2)], NoiseAwarePlacer),
        # a layout without SWAP gates is found, and the reverse traversal keeps it
        ([(0, 2), (2, 1)], NoiseAwarePlacer),
        # there is none, so the reverse traversal decides it
        ([(0, 1), (1, 2), (0, 2)], ReverseTraversal),
    ],
)
def test_post_layout_in_default_layout_search(pairs, decider):
    """The post layout is told the placer that decided the layout, and not the last one."""
    hardware = _line()
    post = Identity()
    passes = generate_preset_passes(
        connectivity=hardware.connectivity,
        optimization_level=1,
        hardware=hardware,
        post_layout=post,
        seed=1,
    )
    passes(_circuit(6, pairs))
    assert [type(placer) for placer in post.placers] == [decider]


def test_noise_aware_post_layout_after_default_layout_search():
    hardware = _line()

    # the layout that was kept is not moved to the quietest qubits
    post = NoiseAwarePostLayout(hardware)
    passes = generate_preset_passes(
        connectivity=hardware.connectivity,
        optimization_level=1,
        hardware=hardware,
        post_layout=post,
    )
    transpiled, _ = passes(_circuit(6, [(0, 1), (1, 2)]))
    assert post.stop_reason == "skipped"
    assert transpiled.wire_names == list(range(6))

    # the best layout that the placer found is not searched again if the post layout trusts it
    post = NoiseAwarePostLayout(hardware, trust_placer=True)
    passes = generate_preset_passes(
        connectivity=hardware.connectivity,
        optimization_level=1,
        hardware=hardware,
        post_layout=post,
    )
    passes(_circuit(6, [(0, 2), (2, 1)]))
    assert post.stop_reason == "skipped"

    # a circuit that needs SWAP gates is routed after the reverse traversal, so it is searched
    passes(_circuit(6, [(0, 1), (1, 2), (0, 2)]))
    assert post.stop_reason != "skipped"


def test_post_layout_in_passes_error():
    hardware = _line()
    passes = Passes([Preprocessing(), Identity()], connectivity=hardware.connectivity)
    with pytest.raises(TranspilerPipelineError, match="needs a router"):
        passes(_circuit(6, [(0, 1)]))


def test_post_layout_in_preset():
    post = Identity()
    undirected = generate_preset_passes(
        connectivity=nx.path_graph(4), post_layout=post
    ).passes
    assert undirected[-1] is post

    # the pass that sets the direction of the gates depends on the qubits, so it is after
    directed = generate_preset_passes(
        connectivity=nx.DiGraph([(0, 1), (1, 2)]), post_layout=post
    ).passes
    assert isinstance(directed[directed.index(post) + 1], GateDirection)

    # there is no post layout by default
    assert post not in generate_preset_passes(connectivity=nx.path_graph(4)).passes


def test_post_layout_in_preset_error():
    with pytest.raises(ValueError, match="requires ``connectivity``"):
        generate_preset_passes(post_layout=Identity())


def test_transpile_post_layout():
    hardware = _line()
    circuit = _circuit(3, [(0, 1), (1, 2), (0, 2)])
    transpiled, layout = transpile(
        circuit,
        connectivity=hardware.connectivity,
        post_layout=NoiseAwarePostLayout(hardware),
        seed=1,
    )
    assert {transpiled.wire_names[q] for g in transpiled.queue for q in g.qubits} == {
        3,
        4,
        5,
    }
    assert list(layout.initial_layout) == transpiled.wire_names


def _checks(matcher, number):
    """Results of some feasibility checks of the first node of a matcher."""
    return [matcher.syntactic_feasibility(0, 0) for _ in range(number)]


def _circuit(nqubits, pairs, measured=(), gate=gates.CNOT):
    """Circuit with a two-qubit gate for each pair, and a measurement of some qubits."""
    circuit = Circuit(nqubits)
    for pair in pairs:
        circuit.add(gate(*pair))
    if measured:
        circuit.add(gates.M(*measured))
    return circuit


def _line(couplings=(0.05, 0.04, 0.02, 0.001, 0.003), errors=None, **kwargs):
    """Hardware model of qubits in a line, with the error of each coupling."""
    pairs = [(index, index + 1) for index in range(len(couplings))]
    errors = {"2q": dict(zip(pairs, couplings))} | (errors or {})
    return HardwareModel(
        tuple(range(len(couplings) + 1)), pairs, errors=errors, **kwargs
    )


def _weight(error):
    """Weight of an error, which is the negative logarithm of its success probability."""
    return -math.log1p(-error)
