import pickle
import random
from concurrent.futures.process import BrokenProcessPool

import networkx as nx
import numpy as np
import pytest
from qiskit.transpiler import PassManager, passes

from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibo.quantum_info import random_statevector, random_unitary
from qibo.transpiler import pipeline as pipeline_module
from qibo.transpiler import unroller as unroller_module
from qibo.transpiler._exceptions import (
    BlockingError,
    ConnectivityError,
    DecompositionError,
    PlacementError,
    TranspilerPipelineError,
)
from qibo.transpiler.abstract import Optimizer, Placer, Router
from qibo.transpiler.asserts import (
    _transpose_qubits,
    assert_circuit_equivalence,
    assert_decomposition,
    assert_transpiling,
)
from qibo.transpiler.hardware import HardwareModel
from qibo.transpiler.optimizer import (
    BarrierBeforeFinalMeasurements,
    CommutativeCancellation,
    ConsolidateBlocks,
    ElidePermutations,
    FilterGates,
    FixedPoint,
    GateDirection,
    Optimize1qGatesDecomposition,
    Preprocessing,
    RemoveIdentityEquivalent,
    Validation,
)
from qibo.transpiler.pipeline import (
    Passes,
    generate_preset_passes,
    restrict_connectivity_qubits,
    transpile,
)
from qibo.transpiler.placer import (
    Dense,
    NoiseAwarePlacer,
    NoiseAwarePostLayout,
    Random,
    ReverseTraversal,
    StarConnectivityPlacer,
    Subgraph,
)
from qibo.transpiler.qiskit import QiskitPasses
from qibo.transpiler.router import (
    NoRouting,
    Sabre,
    ShortestPaths,
    StarConnectivityRouter,
)
from qibo.transpiler.unroller import NativeGates, Unroller


def generate_random_circuit(nqubits, ngates, names=None, seed=42):
    """Generate a random circuit with RX and CZ gates."""
    np.random.seed(seed)
    one_qubit_gates = [gates.RX, gates.RY, gates.RZ, gates.X, gates.Y, gates.Z, gates.H]
    two_qubit_gates = [
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
        gates.iSWAP,
        gates.CRX,
        gates.CRY,
        gates.CRZ,
    ]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits, wire_names=names)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta, trainable=False))
        else:
            circuit.add(gate(*q))
    return circuit


def test_restrict_qubits_error_no_subset(star_connectivity):
    with pytest.raises(ConnectivityError) as excinfo:
        restrict_connectivity_qubits(star_connectivity(), [0, 1, 5])
    assert "Some qubits are not in the original connectivity." in str(excinfo.value)


def test_restrict_qubits_error_not_connected(star_connectivity):
    with pytest.raises(ConnectivityError) as excinfo:
        restrict_connectivity_qubits(star_connectivity(), [0, 1])
    assert "New connectivity graph is not connected." in str(excinfo.value)


def test_on_qubits_without_connectivity_error():
    with pytest.raises(ConnectivityError) as excinfo:
        Passes(passes=None, on_qubits=[0, 1, 2])
    assert "``connectivity`` must be provided" in str(excinfo.value)


def test_restrict_qubits(star_connectivity):
    new_connectivity = restrict_connectivity_qubits(
        star_connectivity(["A", "B", "C", "D", "E"]), ["A", "B", "C"]
    )
    assert list(new_connectivity.nodes) == ["A", "B", "C"]
    assert list(new_connectivity.edges) == [("A", "C"), ("B", "C")]


def test_restrict_qubits_directed(star_connectivity):
    directed = nx.DiGraph([("A", "C"), ("C", "B"), ("D", "C"), ("E", "C")])
    new_connectivity = restrict_connectivity_qubits(directed, ["A", "B", "C"])
    assert new_connectivity.is_directed()
    assert list(new_connectivity.nodes) == ["A", "B", "C"]
    assert list(new_connectivity.edges) == [("A", "C"), ("C", "B")]
    with pytest.raises(ConnectivityError, match="not connected"):
        restrict_connectivity_qubits(directed, ["A", "B", "D"])


def test_generate_preset_passes_directed_structure():
    connectivity = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    natives = NativeGates.ECR | NativeGates.U3 | NativeGates.M
    pipeline = generate_preset_passes(connectivity, natives, seed=3)
    assert pipeline.connectivity is connectivity
    *passes, direction, translator = pipeline.passes
    assert [type(transpiler_pass) for transpiler_pass in passes] == [
        Validation,
        Unroller,
        Preprocessing,
        BarrierBeforeFinalMeasurements,
        Sabre,
        FilterGates,
        Unroller,
    ]
    assert isinstance(direction, GateDirection)
    assert isinstance(translator, Unroller) and translator.native_gates == natives


def test_generate_preset_passes_too_many_qubits(star_connectivity):
    """A circuit larger than the hardware raises an error that says so."""
    passes = generate_preset_passes(connectivity=star_connectivity())
    circuit = Circuit(7)
    circuit.add(gates.CNOT(0, 6))
    with pytest.raises(
        ValueError, match="can't be greater than the number of physical"
    ):
        passes(circuit)


def test_generate_preset_passes_undirected_has_no_gate_direction():
    pipeline = generate_preset_passes(nx.path_graph(4))
    assert not any(isinstance(p, GateDirection) for p in pipeline.passes)


@pytest.mark.parametrize("routing_method", ["sabre", "shortest_paths"])
@pytest.mark.parametrize(
    "natives",
    [
        NativeGates.default(),
        NativeGates.U3
        | NativeGates.ECR
        | NativeGates.Z
        | NativeGates.RZ
        | NativeGates.M
        | NativeGates.I,
        NativeGates.RZ
        | NativeGates.SX
        | NativeGates.X
        | NativeGates.CNOT
        | NativeGates.M
        | NativeGates.I,
    ],
    ids=["default", "ECR", "CNOT"],
)
def test_generate_preset_passes_directed_connectivity(backend, natives, routing_method):
    # every edge goes in one direction only
    connectivity = nx.DiGraph([(0, 1), (2, 1), (2, 3), (4, 3), (4, 0)])
    circuit = generate_random_circuit(nqubits=5, ngates=15, seed=1)
    circuit.add(gates.M(*range(5)))
    pipeline = generate_preset_passes(
        connectivity, natives, seed=1, routing_method=routing_method, backend=backend
    )
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled_circuit)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=natives,
    )


def test_generate_preset_passes_directed_wire_names(backend):
    names = ["a", "b", "c", "d"]
    connectivity = nx.DiGraph([("b", "a"), ("b", "c"), ("d", "c")])
    natives = NativeGates.default() | NativeGates.CNOT
    circuit = generate_random_circuit(nqubits=4, ngates=12, names=names)
    pipeline = generate_preset_passes(connectivity, natives, seed=2, backend=backend)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled_circuit)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=natives,
    )


def test_generate_preset_passes_directed_on_qubits(backend):
    connectivity = nx.DiGraph([(0, 1), (2, 1), (2, 3), (4, 3)])
    natives = NativeGates.default() | NativeGates.CNOT
    circuit = generate_random_circuit(nqubits=3, ngates=10, names=[1, 2, 3])
    pipeline = generate_preset_passes(
        connectivity, natives, seed=2, on_qubits=[1, 2, 3], backend=backend
    )
    assert pipeline.connectivity.is_directed()
    transpiled_circuit, _ = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled_circuit)


def test_assert_circuit_equivalence_wrong_nqubits():
    circ1 = Circuit(1)
    circ2 = Circuit(2)
    final_layout = {0: 0, 1: 1}
    with pytest.raises(ValueError):
        assert_circuit_equivalence(circ1, circ2, final_layout=final_layout)


@pytest.mark.parametrize("qubits", [3, 5])
def test_is_satisfied(qubits, star_connectivity):
    default_transpiler = Passes(
        passes=None, connectivity=star_connectivity(), on_qubits=list(range(qubits))
    )
    circuit = Circuit(qubits, wire_names=list(range(qubits)))
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.Z(0))
    assert default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_placement(star_connectivity):
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5, wire_names=["A", "B", "C", "D", "E"])
    assert not default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_decomposition(star_connectivity):
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.X(0))
    assert not default_transpiler.is_satisfied(circuit)


def test_is_satisfied_false_connectivity(star_connectivity):
    default_transpiler = Passes(passes=None, connectivity=star_connectivity())
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.Z(0))
    assert not default_transpiler.is_satisfied(circuit)


def test_is_satisfied_without_connectivity(backend):
    # there is nothing to place and route, so only the native gates are checked
    pipeline = generate_preset_passes(seed=1, backend=backend)
    assert pipeline.connectivity is None
    circuit = Circuit(3)
    circuit.add([gates.H(0), gates.TOFFOLI(0, 1, 2)])
    transpiled_circuit, _ = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled_circuit)
    assert not pipeline.is_satisfied(circuit)


def test_is_satisfied_without_connectivity_not_native():
    default_transpiler = Passes(passes=None)
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.Z(0))
    assert default_transpiler.is_satisfied(circuit)
    circuit.add(gates.X(0))
    assert not default_transpiler.is_satisfied(circuit)


@pytest.mark.parametrize("nqubits", [2, 3, 5])
@pytest.mark.parametrize("ngates", [5, 20])
@pytest.mark.parametrize("placer", [Random, ReverseTraversal])
@pytest.mark.parametrize("router", [ShortestPaths, Sabre])
def test_custom_passes(backend, router, placer, ngates, nqubits, star_connectivity):
    connectivity = star_connectivity()
    circ = generate_random_circuit(nqubits=nqubits, ngates=ngates)
    custom_passes = []
    custom_passes.append(Preprocessing())
    if placer == ReverseTraversal:
        custom_passes.append(
            placer(
                routing_algorithm=router(),
            )
        )
    else:
        custom_passes.append(placer())
    custom_passes.append(router())
    custom_passes.append(Unroller(native_gates=NativeGates.default(), backend=backend))
    custom_pipeline = Passes(
        passes=custom_passes,
        connectivity=connectivity,
        native_gates=NativeGates.default(),
    )
    transpiled_circ, final_layout = custom_pipeline(circ, backend=backend)
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=connectivity,
        final_layout=final_layout,
    )


@pytest.mark.parametrize("ngates", [5, 20])
@pytest.mark.parametrize("placer", [Random, ReverseTraversal])
@pytest.mark.parametrize("routing", [ShortestPaths, Sabre])
@pytest.mark.parametrize("restrict_names", [[1, 2, 3], [0, 2, 4], [4, 2, 3]])
def test_custom_passes_restrict(
    backend, ngates, placer, routing, restrict_names, star_connectivity
):
    connectivity = star_connectivity()
    circ = generate_random_circuit(nqubits=3, ngates=ngates, names=restrict_names)
    custom_passes = []
    custom_passes.append(Preprocessing())
    if placer == ReverseTraversal:
        custom_passes.append(
            placer(
                routing_algorithm=routing(),
            )
        )
    else:
        custom_passes.append(placer())
    custom_passes.append(routing())
    custom_passes.append(Unroller(native_gates=NativeGates.default(), backend=backend))
    custom_pipeline = Passes(
        passes=custom_passes,
        connectivity=connectivity,
        native_gates=NativeGates.default(),
        on_qubits=restrict_names,
    )
    transpiled_circ, final_layout = custom_pipeline(circ, backend=backend)
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=restrict_connectivity_qubits(star_connectivity(), restrict_names),
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )
    assert set(transpiled_circ.wire_names) == set(restrict_names)


def test_custom_passes_wrong_pass(backend):
    custom_passes = [0]
    custom_pipeline = Passes(passes=custom_passes, connectivity=None)
    circ = generate_random_circuit(nqubits=5, ngates=5)
    with pytest.raises(TranspilerPipelineError):
        custom_pipeline(circ, backend=backend)


def test_custom_passes_optimizer_backend(backend):
    class Recorder(Optimizer):
        def __call__(self, circuit, backend=None):
            self.backend = backend
            return circuit

    optimizer = Recorder()
    Passes([optimizer], connectivity=None)(Circuit(1), backend=backend)
    assert optimizer.backend is backend


def test_custom_passes_optimizer_without_backend(backend):
    class NoBackend(Optimizer):
        def __call__(self, circuit):
            self.called = True
            return circuit

    optimizer = NoBackend()
    Passes([optimizer], connectivity=None)(Circuit(1), backend=backend)
    assert optimizer.called


def test_int_qubit_names(backend, star_connectivity):
    names = [980, 123, 45, 9, 210464]
    connectivity = star_connectivity(names)
    transpiler = Passes(
        connectivity=connectivity,
        passes=[
            Preprocessing(),
            Random(seed=0),
            Sabre(),
            Unroller(NativeGates.default(), backend=backend),
        ],
    )
    circuit = Circuit(1, wire_names=[123])
    circuit.add(gates.I(0))
    circuit.add(gates.H(0))
    circuit.add(gates.M(0))
    transpiled_circuit, final_layout = transpiler(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )


def test_qiskit_passes(backend, star_connectivity):
    nqubits = 5
    ngates = 10
    connectivity = star_connectivity()
    circ = generate_random_circuit(nqubits=nqubits, ngates=ngates)
    custom_passes = []
    custom_passes.append(Preprocessing())
    custom_passes.append(
        ReverseTraversal(
            routing_algorithm=Sabre(),
        )
    )
    custom_passes.append(Sabre())
    qiskit_passes = [passes.Optimize1qGates(), passes.InverseCancellation()]
    custom_passes.append(QiskitPasses(PassManager(qiskit_passes)))
    custom_passes.append(Unroller(native_gates=NativeGates.default(), backend=backend))
    custom_pipeline = Passes(
        passes=custom_passes,
        connectivity=connectivity,
        native_gates=NativeGates.default(),
    )
    transpiled_circ, final_layout = custom_pipeline(circ, backend=backend)
    assert_transpiling(
        original_circuit=circ,
        transpiled_circuit=transpiled_circ,
        connectivity=star_connectivity(),
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )


def test_generate_preset_passes_structure():
    connectivity = nx.path_graph(5)
    natives = NativeGates.CZ | NativeGates.GPI2 | NativeGates.RZ | NativeGates.M
    pipeline = generate_preset_passes(connectivity, natives, seed=3)
    assert pipeline.connectivity is connectivity
    assert pipeline.native_gates == natives
    validation, unroller, preprocessing, barrier, router, restoration, translator = (
        pipeline.passes
    )
    assert isinstance(validation, Validation)
    assert isinstance(unroller, Unroller) and unroller.min_qubits == 3
    assert isinstance(preprocessing, Preprocessing)
    # the final measurements are kept after the SWAP gates of the router
    assert isinstance(barrier, BarrierBeforeFinalMeasurements)
    assert isinstance(restoration, FilterGates)
    assert isinstance(router, Sabre) and router.trials == 5
    # the router is seeded, which makes the transpilation reproducible
    assert router._rng.getstate() == random.Random(3).getstate()
    assert isinstance(translator, Unroller) and translator.min_qubits == 1
    assert unroller.native_gates == translator.native_gates == natives


def test_generate_preset_passes_default_natives():
    pipeline = generate_preset_passes(nx.path_graph(3))
    assert pipeline.native_gates == NativeGates.default()
    assert pipeline.passes[-1].native_gates == NativeGates.default()


@pytest.mark.parametrize(
    "natives",
    [
        NativeGates.default(),
        NativeGates.U3
        | NativeGates.iSWAP
        | NativeGates.Z
        | NativeGates.RZ
        | NativeGates.M
        | NativeGates.I,
        NativeGates.RZ
        | NativeGates.SX
        | NativeGates.X
        | NativeGates.CNOT
        | NativeGates.M
        | NativeGates.I,
    ],
)
@pytest.mark.parametrize("ngates", [5, 20])
@pytest.mark.parametrize("nqubits", [2, 3, 5])
def test_generate_preset_passes_random_circuits(backend, nqubits, ngates, natives):
    connectivity = nx.path_graph(5)
    circuit = generate_random_circuit(nqubits=nqubits, ngates=ngates)
    circuit.add(gates.M(*range(nqubits)))
    pipeline = generate_preset_passes(connectivity, natives, seed=0, backend=backend)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled_circuit)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=natives,
    )


@pytest.mark.parametrize("ncontrols", [2, 3, 4])
def test_generate_preset_passes_multi_qubit_gates(backend, ncontrols):
    connectivity = nx.path_graph(7)
    circuit = Circuit(ncontrols + 1)
    circuit.add(gates.H(0))
    circuit.add(gates.X(ncontrols).controlled_by(*range(ncontrols)))
    circuit.add(gates.CNOT(ncontrols, 0))
    pipeline = generate_preset_passes(connectivity, seed=1, backend=backend)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )
    # the pipeline that unrolls after the routing cannot route these gates
    unrolling_last = Passes(
        [Preprocessing(), Sabre(), Unroller(NativeGates.default())],
        connectivity=connectivity,
    )
    with pytest.raises(BlockingError):
        unrolling_last(circuit)


@pytest.mark.parametrize(
    "initial_layout, needs_swaps",
    [([1, 2, 3], False), ([4, 3, 2], False), ([3, 1, 4], True)],
)
def test_generate_preset_passes_initial_layout(backend, initial_layout, needs_swaps):
    connectivity = nx.path_graph(5)
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.M(0, 1, 2))
    pipeline = generate_preset_passes(
        connectivity, initial_layout=initial_layout, seed=0, backend=backend
    )
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert transpiled_circuit.wire_names[:3] == initial_layout
    assert sorted(transpiled_circuit.wire_names) == list(connectivity.nodes)
    # the input circuit is not modified
    assert circuit.wire_names == [0, 1, 2]
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )
    # each CNOT on adjacent qubits needs one CZ, and every SWAP needs three more
    ncz = sum(isinstance(gate, gates.CZ) for gate in transpiled_circuit.queue)
    assert (ncz > 2) == needs_swaps
    assert needs_swaps or ncz == 2


@pytest.mark.parametrize(
    "initial_layout, layout_list",
    [
        ({0: 1, 1: 2, 2: 3}, [1, 2, 3]),
        ({0: 3, 2: 4}, [3, 0, 4]),
        ({1: 4}, [0, 4, 1]),
    ],
)
def test_generate_preset_passes_initial_layout_dict(
    backend, initial_layout, layout_list
):
    connectivity = nx.path_graph(5)
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.M(0, 1, 2))
    transpiled_circuit, final_layout = transpile(
        circuit, connectivity, initial_layout=initial_layout, seed=0, backend=backend
    )
    expected_circuit, expected_layout = transpile(
        circuit, connectivity, initial_layout=layout_list, seed=0, backend=backend
    )
    assert transpiled_circuit.wire_names[:3] == layout_list
    assert [(gate.name, gate.qubits) for gate in transpiled_circuit.queue] == [
        (gate.name, gate.qubits) for gate in expected_circuit.queue
    ]
    assert final_layout == expected_layout
    assert circuit.wire_names == [0, 1, 2]
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


@pytest.mark.parametrize(
    "initial_layout", [[0, 1], [0, 0, 1], [0, 1, 9], {3: 0}, {0: 1, 1: 1}, {1: 9}]
)
def test_generate_preset_passes_initial_layout_error(initial_layout):
    pipeline = generate_preset_passes(nx.path_graph(5), initial_layout=initial_layout)
    with pytest.raises(PlacementError):
        pipeline(Circuit(3))


def test_generate_preset_passes_initial_layout_without_connectivity_error():
    with pytest.raises(ConnectivityError):
        generate_preset_passes(initial_layout=[0, 1, 2])


def test_generate_preset_passes_seed(backend):
    connectivity = nx.path_graph(6)
    circuit = generate_random_circuit(nqubits=6, ngates=40, seed=5)

    def transpile(seed):
        pipeline = generate_preset_passes(connectivity, seed=seed, backend=backend)
        transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
        gate_list = [(gate.name, gate.qubits) for gate in transpiled_circuit.queue]
        return gate_list, final_layout

    assert transpile(seed=11) == transpile(seed=11)


@pytest.mark.parametrize("ngates", [5, 20])
def test_generate_preset_passes_without_connectivity(backend, ngates):
    natives = NativeGates.default()
    circuit = generate_random_circuit(nqubits=3, ngates=ngates)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    pipeline = generate_preset_passes(native_gates=natives, backend=backend)
    assert len(pipeline.passes) == 2
    assert isinstance(pipeline.passes[0], Validation)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert final_layout == {0: 0, 1: 1, 2: 2}
    assert final_layout.initial_layout == [0, 1, 2]
    assert final_layout.nqubits_input == 3
    assert transpiled_circuit.nqubits == circuit.nqubits
    assert_decomposition(transpiled_circuit, natives)
    assert_circuit_equivalence(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        final_layout={qubit: qubit for qubit in range(circuit.nqubits)},
    )


@pytest.mark.parametrize("restrict_names", [[1, 2, 3], [0, 2, 4], [4, 2, 3]])
def test_generate_preset_passes_on_qubits(backend, restrict_names, star_connectivity):
    connectivity = star_connectivity()
    circuit = generate_random_circuit(nqubits=3, ngates=10, names=restrict_names)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    pipeline = generate_preset_passes(
        connectivity, on_qubits=restrict_names, seed=0, backend=backend
    )
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=pipeline.connectivity,
        final_layout=final_layout,
    )
    assert set(transpiled_circuit.wire_names) == set(restrict_names)


@pytest.fixture
def recorded_backends(monkeypatch):
    """Record the backends that the unrollers give to ``translate_gate``."""
    backends = []
    translate = unroller_module.translate_gate

    def recorder(gate, native_gates, backend=None, **kwargs):
        backends.append(backend)
        return translate(gate, native_gates, backend=backend, **kwargs)

    monkeypatch.setattr(unroller_module, "translate_gate", recorder)
    return backends


def test_passes_call_backend_reaches_unrollers(recorded_backends):
    own_backend, call_backend = NumpyBackend(), NumpyBackend()
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 2))
    pipeline = Passes(
        [Unroller(NativeGates.default(), backend=own_backend)],
        connectivity=nx.path_graph(3),
    )

    pipeline(circuit, backend=call_backend)
    assert recorded_backends and all(b is call_backend for b in recorded_backends)

    # without a backend in the call the unroller keeps its own, and not the global one
    recorded_backends.clear()
    pipeline(circuit)
    assert recorded_backends and all(b is own_backend for b in recorded_backends)


def test_generate_preset_passes_call_backend(recorded_backends):
    own_backend, call_backend = NumpyBackend(), NumpyBackend()
    circuit = Circuit(3)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    pipeline = generate_preset_passes(nx.path_graph(5), seed=0, backend=own_backend)

    # both unrollers of the preset use the backend of the call
    pipeline(circuit, backend=call_backend)
    assert recorded_backends and all(b is call_backend for b in recorded_backends)

    recorded_backends.clear()
    pipeline(circuit)
    assert recorded_backends and all(b is own_backend for b in recorded_backends)


def _controlled_circuit(nqubits, ncontrols, touched=()):
    """Multi-controlled X on the first ``ncontrols + 1`` qubits, with the other
    qubits idle, unless they are acted on by a gate before the multi-controlled gate."""
    circuit = Circuit(nqubits)
    for qubit in range(ncontrols + 1):
        circuit.add(gates.RY(qubit, 0.3 + 0.2 * qubit))
    for qubit in touched:
        circuit.add(gates.RY(qubit, 0.9 + 0.1 * qubit))
    circuit.add(gates.X(ncontrols).controlled_by(*range(ncontrols)))
    return circuit


def _assert_equivalent_from_zero(circuit, transpiled_circuit, connectivity, layout):
    """Equivalence for the initial state :math:`\\ket{0}^{\\otimes n}` only."""
    original = Preprocessing(connectivity=connectivity)(circuit.copy(deep=True))
    zero_state = np.zeros(2**original.nqubits, dtype=complex)
    zero_state[0] = 1
    assert_circuit_equivalence(
        original_circuit=original,
        transpiled_circuit=transpiled_circuit,
        final_layout=layout,
        test_states=[zero_state],
    )


def _count_two_qubit_gates(circuit):
    return sum(len(gate.qubits) == 2 for gate in circuit.queue)


def test_generate_preset_passes_auxiliary_qubit_options():
    default = generate_preset_passes(nx.path_graph(5)).passes
    # the qubits that multi-controlled gates do not act on are dirty auxiliary qubits
    assert default[1].use_dirty_ancillas
    assert not default[1].qubits_initially_zero and not default[1].minimize_toffolis
    assert not default[-1].use_dirty_ancillas

    zero = generate_preset_passes(nx.path_graph(5), qubits_initially_zero=True).passes
    assert zero[1].use_dirty_ancillas
    assert zero[1].qubits_initially_zero and zero[1].minimize_toffolis
    assert not zero[-1].qubits_initially_zero

    # without connectivity the only unroller decomposes the multi-controlled gates
    _, unroller = generate_preset_passes(qubits_initially_zero=True).passes
    assert unroller.use_dirty_ancillas and unroller.qubits_initially_zero


def test_generate_preset_passes_auxiliary_qubits_lower_the_cost(backend):
    connectivity = nx.complete_graph(12)
    circuit = _controlled_circuit(nqubits=12, ncontrols=7)

    def transpile(**options):
        pipeline = generate_preset_passes(
            connectivity, seed=0, backend=backend, **options
        )
        return pipeline(circuit, backend=backend)

    without_auxiliary = Passes(
        [
            Unroller(NativeGates.default(), backend=backend, min_qubits=3),
            Preprocessing(),
            Sabre(seed=0),
            Unroller(NativeGates.default(), backend=backend),
        ],
        connectivity=connectivity,
    )(circuit, backend=backend)[0]
    dirty, dirty_layout = transpile()
    clean, clean_layout = transpile(qubits_initially_zero=True)

    assert (
        _count_two_qubit_gates(clean)
        < _count_two_qubit_gates(dirty)
        < _count_two_qubit_gates(without_auxiliary)
    )
    # the dirty auxiliary qubits do not need any initial state
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=dirty,
        connectivity=connectivity,
        final_layout=dirty_layout,
    )
    # while the clean ones are valid for the state 0
    _assert_equivalent_from_zero(circuit, clean, connectivity, clean_layout)


@pytest.mark.parametrize("qubits_initially_zero", [False, True])
def test_generate_preset_passes_touched_qubits_are_not_clean(
    backend, qubits_initially_zero
):
    connectivity = nx.complete_graph(10)
    circuit = _controlled_circuit(nqubits=10, ncontrols=5, touched=(6, 7, 8, 9))
    pipeline = generate_preset_passes(
        connectivity,
        seed=0,
        qubits_initially_zero=qubits_initially_zero,
        backend=backend,
    )
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    _assert_equivalent_from_zero(
        circuit, transpiled_circuit, connectivity, final_layout
    )
    if not qubits_initially_zero:
        assert_transpiling(
            original_circuit=circuit,
            transpiled_circuit=transpiled_circuit,
            connectivity=connectivity,
            final_layout=final_layout,
        )


def test_generate_preset_passes_auxiliary_qubits_without_connectivity(backend):
    natives = NativeGates.default()
    circuit = _controlled_circuit(nqubits=10, ncontrols=6)
    pipeline = generate_preset_passes(
        native_gates=natives, qubits_initially_zero=True, backend=backend
    )
    transpiled_circuit, _ = pipeline(circuit, backend=backend)
    assert_decomposition(transpiled_circuit, natives)
    plain = Passes([Unroller(natives, backend=backend)])(circuit, backend=backend)[0]
    assert _count_two_qubit_gates(transpiled_circuit) < _count_two_qubit_gates(plain)
    state = backend.to_numpy(backend.execute_circuit(circuit).state())
    final_state = backend.to_numpy(backend.execute_circuit(transpiled_circuit).state())
    phase = np.vdot(final_state, state)
    assert np.isclose(abs(phase), 1.0)
    np.testing.assert_allclose(final_state * phase / abs(phase), state, atol=1e-8)


@pytest.mark.parametrize(
    "layout_method, placer_class",
    [
        (None, None),
        ("trivial", None),
        ("random", Random),
        ("subgraph", Subgraph),
        ("reverse_traversal", ReverseTraversal),
    ],
)
def test_generate_preset_passes_layout_method(
    backend, layout_method, placer_class, grid_connectivity
):
    connectivity = grid_connectivity()
    circuit = generate_random_circuit(nqubits=4, ngates=20, seed=2)
    circuit.add(gates.M(*range(4)))
    pipeline = generate_preset_passes(
        connectivity, seed=5, layout_method=layout_method, backend=backend
    )
    if placer_class is None:
        assert len(pipeline.passes) == 7
    else:
        assert len(pipeline.passes) == 8
        assert isinstance(pipeline.passes[2], Preprocessing)
        assert isinstance(pipeline.passes[3], placer_class)
        assert isinstance(pipeline.passes[5], Sabre)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


def test_generate_preset_passes_layout_method_instance(backend, grid_connectivity):
    connectivity = grid_connectivity()
    placer = Random(samples=10, seed=3)
    circuit = generate_random_circuit(nqubits=4, ngates=20, seed=2)
    pipeline = generate_preset_passes(
        connectivity, layout_method=placer, backend=backend
    )
    assert pipeline.passes[3] is placer
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


def test_generate_preset_passes_methods_are_seeded(grid_connectivity):
    connectivity = grid_connectivity()
    kwargs = {"seed": 7, "layout_method": "reverse_traversal"}
    pipeline = generate_preset_passes(connectivity, **kwargs)
    placer = pipeline.passes[3]
    assert placer.routing_algorithm._rng.getstate() == random.Random(7).getstate()

    placer = generate_preset_passes(
        connectivity, seed=7, layout_method="random"
    ).passes[3]
    assert placer.seed == 7
    router = generate_preset_passes(
        connectivity, seed=7, routing_method="shortest_paths"
    ).passes[-3]
    assert router._rng.getstate() == random.Random(7).getstate()


def test_generate_preset_passes_random_layout_is_reproducible(grid_connectivity):
    connectivity = grid_connectivity()
    circuit = generate_random_circuit(nqubits=4, ngates=20, seed=2)

    def transpile():
        pipeline = generate_preset_passes(connectivity, seed=9, layout_method="random")
        transpiled_circuit, final_layout = pipeline(circuit)
        gate_list = [(gate.name, gate.qubits) for gate in transpiled_circuit.queue]
        return gate_list, transpiled_circuit.wire_names, final_layout

    assert transpile() == transpile()


@pytest.mark.parametrize("routing_method", [None, "sabre"])
def test_generate_preset_passes_routing_method_sabre(
    backend, routing_method, grid_connectivity
):
    connectivity = grid_connectivity()
    circuit = generate_random_circuit(nqubits=4, ngates=20, seed=2)
    pipeline = generate_preset_passes(
        connectivity, seed=1, routing_method=routing_method, backend=backend
    )
    router = pipeline.passes[4]
    assert isinstance(router, Sabre) and router.trials == 5
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


def test_generate_preset_passes_routing_method_instance(backend, grid_connectivity):
    connectivity = grid_connectivity()
    router = Sabre(lookahead=0, delta=0.0, seed=2, trials=2)
    circuit = generate_random_circuit(nqubits=4, ngates=20, seed=2)
    pipeline = generate_preset_passes(
        connectivity, routing_method=router, backend=backend
    )
    assert pipeline.passes[4] is router
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


def test_generate_preset_passes_routing_method_shortest_paths(backend):
    connectivity = nx.path_graph(5)
    # a single gate between qubits at distance two
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.M(0, 1, 2))
    pipeline = generate_preset_passes(
        connectivity, seed=1, routing_method="shortest_paths", backend=backend
    )
    assert isinstance(pipeline.passes[4], ShortestPaths)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


def test_generate_preset_passes_star_methods(backend, star_connectivity):
    connectivity = star_connectivity()
    circuit = generate_random_circuit(nqubits=5, ngates=15, seed=4)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    pipeline = generate_preset_passes(
        connectivity,
        layout_method="star",
        routing_method="star",
        backend=backend,
    )
    assert isinstance(pipeline.passes[3], StarConnectivityPlacer)
    assert isinstance(pipeline.passes[5], StarConnectivityRouter)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
    )


@pytest.mark.parametrize("layout_method", [None, "trivial"])
def test_generate_preset_passes_initial_layout_with_trivial_method(layout_method):
    pipeline = generate_preset_passes(
        nx.path_graph(5), initial_layout=[1, 2, 3], layout_method=layout_method
    )
    transpiled_circuit, _ = pipeline(Circuit(3))
    assert transpiled_circuit.wire_names[:3] == [1, 2, 3]


@pytest.mark.parametrize("layout_method", ["random", "subgraph", Random()])
def test_generate_preset_passes_initial_layout_with_placer_error(layout_method):
    with pytest.raises(PlacementError):
        generate_preset_passes(
            nx.path_graph(5), initial_layout=[1, 2, 3], layout_method=layout_method
        )


@pytest.mark.parametrize("layout_method", ["grid", "Random", 3])
def test_generate_preset_passes_unknown_layout_method(layout_method):
    with pytest.raises(ValueError, match="layout method"):
        generate_preset_passes(nx.path_graph(5), layout_method=layout_method)


@pytest.mark.parametrize("routing_method", ["basic", "Sabre", 3])
def test_generate_preset_passes_unknown_routing_method(routing_method):
    with pytest.raises(ValueError, match="routing method"):
        generate_preset_passes(nx.path_graph(5), routing_method=routing_method)


@pytest.mark.parametrize(
    "options", [{"layout_method": "random"}, {"routing_method": "sabre"}]
)
def test_generate_preset_passes_methods_without_connectivity_error(options):
    with pytest.raises(ConnectivityError):
        generate_preset_passes(**options)


@pytest.mark.parametrize("routing_method", ["sabre", "shortest_paths"])
def test_generate_preset_passes_disconnected_connectivity(backend, routing_method):
    connectivity = nx.Graph([(0, 1), (1, 2), (3, 4)])
    pipeline = generate_preset_passes(
        connectivity, seed=1, routing_method=routing_method, backend=backend
    )
    circuit = Circuit(5)
    circuit.add([gates.H(0), gates.CNOT(0, 2), gates.CNOT(3, 4), gates.M(0, 2, 3, 4)])
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled_circuit)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=NativeGates.default(),
    )
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 3))
    with pytest.raises(ConnectivityError, match="different connected components"):
        pipeline(circuit, backend=backend)


@pytest.mark.parametrize("directed", [False, True], ids=["undirected", "directed"])
def test_generate_preset_passes_unitary_on_many_qubits(backend, directed):
    connectivity = nx.DiGraph if directed else nx.Graph
    connectivity = connectivity([(0, 1), (1, 2), (2, 3)])
    natives = NativeGates.default() | NativeGates.CNOT
    circuit = Circuit(4)
    circuit.add(gates.H(0))
    circuit.add(gates.Unitary(random_unitary(8, seed=1, backend=backend), 2, 0, 1))
    circuit.add(gates.CNOT(1, 3))
    circuit.add(gates.M(0, 2))
    pipeline = generate_preset_passes(connectivity, natives, seed=1, backend=backend)
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled_circuit)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=natives,
    )


def test_transpile_circuit(backend):
    connectivity = nx.path_graph(5)
    natives = NativeGates.default()
    circuit = generate_random_circuit(nqubits=4, ngates=10)
    circuit.add(gates.M(*range(4)))
    transpiled_circuit, final_layout = transpile(
        circuit, connectivity, natives, seed=0, backend=backend
    )
    pipeline = generate_preset_passes(connectivity, natives, seed=0, backend=backend)
    expected_circuit, expected_layout = pipeline(circuit, backend=backend)
    assert final_layout == expected_layout
    assert [(gate.name, gate.qubits) for gate in transpiled_circuit.queue] == [
        (gate.name, gate.qubits) for gate in expected_circuit.queue
    ]
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=natives,
    )


@pytest.mark.parametrize("routing_method", [None, "shortest_paths"])
@pytest.mark.parametrize(
    "natives",
    [
        NativeGates.default(),
        NativeGates.CZ | NativeGates.U3,
        NativeGates.CZ | NativeGates.iSWAP | NativeGates.GPI2,
        NativeGates["ECR"] | NativeGates["RZ"] | NativeGates["SX"],
    ],
)
def test_transpile_global_phase(backend, natives, routing_method):
    """The transpiled circuit is equal to the original one, including the global phase,
    which is carried by a single global phase gate at the end of the circuit."""
    nqubits = 5
    circuit = Circuit(nqubits)
    circuit.add([gates.H(0), gates.SX(1), gates.T(2), gates.U3(3, 0.1, 0.2, 0.3)])
    circuit.add([gates.CU1(0, 4, 0.4), gates.iSWAP(1, 3), gates.TOFFOLI(0, 2, 4)])
    circuit.add([gates.ECR(2, 3), gates.CNOT(0, 3), gates.SXDG(4), gates.SWAP(1, 4)])

    connectivity = nx.path_graph(nqubits)
    transpiled_circuit, final_layout = transpile(
        circuit,
        connectivity,
        natives,
        seed=0,
        routing_method=routing_method,
        backend=backend,
    )

    global_phases = [
        gate for gate in transpiled_circuit.queue if isinstance(gate, gates.GlobalPhase)
    ]
    assert len(global_phases) == 1
    assert transpiled_circuit.queue[-1] is global_phases[0]
    assert transpiled_circuit.global_phase != 0

    state = random_statevector(2**nqubits, seed=1, backend=backend)
    # the state is copied because some backends modify it in place
    target_state = backend.execute_circuit(
        circuit, initial_state=backend.cast(state, copy=True)
    ).state()
    final_state = backend.execute_circuit(
        transpiled_circuit, initial_state=backend.cast(state, copy=True)
    ).state()
    final_state = _transpose_qubits(final_state, list(final_layout.values()))
    backend.assert_allclose(final_state, target_state, atol=1e-6)


def test_transpile_list_of_circuits(backend):
    connectivity = nx.path_graph(5)
    natives = NativeGates.default()
    circuits = [
        generate_random_circuit(nqubits=nqubits, ngates=10, seed=nqubits)
        for nqubits in (2, 5)
    ]
    results = transpile(circuits, connectivity, natives, seed=0, backend=backend)
    assert len(results) == len(circuits)
    for circuit, (transpiled_circuit, final_layout) in zip(circuits, results):
        assert_transpiling(
            original_circuit=circuit,
            transpiled_circuit=transpiled_circuit,
            connectivity=connectivity,
            final_layout=final_layout,
            native_gates=natives,
        )


def test_transpile_without_connectivity(backend):
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    transpiled_circuit, final_layout = transpile(circuit, backend=backend)
    assert final_layout.routing_permutation == [0, 1]
    assert_decomposition(transpiled_circuit, NativeGates.default())


def test_transpile_forwards_backend(backend, monkeypatch):
    """The backend reaches all the passes, not only the unrollers."""
    received = []
    call = Passes.__call__

    def spy(self, circuit, backend=None):
        received.append(backend)
        return call(self, circuit, backend=backend)

    monkeypatch.setattr(Passes, "__call__", spy)
    circuit = Circuit(2)
    circuit.add(gates.CNOT(0, 1))
    transpile(circuit, backend=backend)
    transpile([circuit, circuit], backend=backend)
    assert received == [backend] * 3


def test_passes_router_without_connectivity_error(backend):
    with pytest.raises(ValueError):
        Passes([Sabre()])(Circuit(2), backend=backend)


def test_generate_preset_passes_callback_and_transpile_list(backend):
    counts = []
    circuits = []
    for qubits in [(0, 1), (1, 2)]:
        circuit = Circuit(3)
        circuit.add(gates.CNOT(*qubits))
        circuits.append(circuit)
    results = transpile(
        circuits,
        nx.path_graph(3),
        seed=1,
        callback=lambda **kwargs: counts.append(kwargs["count"]),
        backend=backend,
    )
    assert len(results) == 2
    # validation, unroller, preprocessing, barrier, unroller; the router is skipped, and the
    # barrier is removed
    assert counts == [0, 1, 2, 3, 5, 6, 0, 1, 2, 3, 5, 6]


@pytest.mark.parametrize("connectivity", [None, nx.path_graph(2)])
def test_transpile_approximation_degree(backend, connectivity):
    circuit = Circuit(2)
    circuit.add(gates.fSim(0, 1, 0.05, 0.02))

    exact, _ = transpile(circuit, connectivity, backend=backend)
    approximate, _ = transpile(
        circuit, connectivity, approximation_degree=0.9, backend=backend
    )
    assert _count_two_qubit_gates(exact) > 0
    assert _count_two_qubit_gates(approximate) == 0

    # the degree is shared by the circuits of a list
    results = transpile(
        [circuit, circuit], connectivity, approximation_degree=0.9, backend=backend
    )
    assert [_count_two_qubit_gates(transpiled) for transpiled, _ in results] == [0, 0]


@pytest.mark.parametrize("degree", [-0.5, 2.0])
def test_transpile_approximation_degree_error(degree):
    with pytest.raises(ValueError, match="approximation_degree"):
        transpile(Circuit(2), approximation_degree=degree)


def test_transpile_in_parallel(backend):
    connectivity = nx.path_graph(5)
    circuits = [generate_random_circuit(5, 30, seed=seed) for seed in (1, 2, 3)]
    kwargs = {"connectivity": connectivity, "seed": 7, "backend": backend}
    results = transpile(circuits, num_processes=2, **kwargs)
    assert len(results) == 3
    for circuit, (transpiled_circuit, final_layout) in zip(circuits, results):
        assert_transpiling(
            original_circuit=circuit,
            transpiled_circuit=transpiled_circuit,
            connectivity=connectivity,
            final_layout=final_layout,
        )
        # each circuit is transpiled as if it was alone, whatever the other circuits are
        alone, alone_layout = transpile(circuit, **kwargs)
        assert [(gate.name, gate.qubits) for gate in transpiled_circuit.queue] == [
            (gate.name, gate.qubits) for gate in alone.queue
        ]
        assert final_layout == alone_layout


def test_transpile_in_parallel_error():
    circuits = [Circuit(2), Circuit(2)]
    circuits[1].add(gates.PauliNoiseChannel([0], [("X", 0.1)]))
    with pytest.raises(DecompositionError, match="PauliNoiseChannel"):
        transpile(circuits, nx.path_graph(2), num_processes=2)


@pytest.mark.parametrize(
    "ncircuits, num_processes", [(0, 4), (1, 4), (3, None), (3, 1)]
)
def test_transpile_without_extra_processes(backend, ncircuits, num_processes):
    """Without several processes, or with less than two circuits, nothing is parallel."""
    connectivity = nx.path_graph(4)
    circuits = [generate_random_circuit(4, 10, seed=seed) for seed in range(ncircuits)]
    kwargs = {"connectivity": connectivity, "seed": 3, "backend": backend}
    results = transpile(circuits, num_processes=num_processes, **kwargs)
    expected = transpile(circuits, **kwargs)
    assert len(results) == ncircuits
    assert [
        [(gate.name, gate.qubits) for gate in circuit.queue] for circuit, _ in results
    ] == [
        [(gate.name, gate.qubits) for gate in circuit.queue] for circuit, _ in expected
    ]
    assert [layout for _, layout in results] == [layout for _, layout in expected]


def test_transpile_single_circuit_with_processes(backend):
    circuit = generate_random_circuit(4, 10)
    result = transpile(circuit, nx.path_graph(4), num_processes=4, backend=backend)
    assert isinstance(result, tuple)
    assert isinstance(result[0], Circuit)


@pytest.mark.parametrize("num_processes", [0, -2, 2.0, "2"])
def test_transpile_num_processes_invalid(num_processes):
    with pytest.raises(ValueError, match="must be a positive integer"):
        transpile([Circuit(2)], num_processes=num_processes)


def test_transpile_in_parallel_broken_pool_hint(monkeypatch):
    """A worker that ends abruptly, as when the ``__main__`` guard is missing, is explained."""

    class BrokenExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, *args):
            raise BrokenProcessPool("A process in the process pool was terminated.")

    monkeypatch.setattr(pipeline_module, "ProcessPoolExecutor", BrokenExecutor)
    with pytest.raises(BrokenProcessPool, match="__main__"):
        transpile([Circuit(2), Circuit(2)], nx.path_graph(2), num_processes=2)


def test_transpile_callback_with_processes_error():
    with pytest.raises(ValueError, match="cannot be used with more than one process"):
        transpile([Circuit(2)], num_processes=2, callback=lambda **kwargs: None)
    # one process runs the callback in the current process
    counts = []
    transpile(
        [Circuit(2)],
        num_processes=1,
        callback=lambda **kwargs: counts.append(kwargs["count"]),
    )
    assert counts == [0, 1]


def test_generate_preset_passes_dense_layout(backend, grid_connectivity):
    connectivity = grid_connectivity()
    natives = NativeGates.default()
    circuit = generate_random_circuit(5, 10)
    transpiled_circuit, layout = transpile(
        circuit,
        connectivity,
        natives,
        seed=1,
        layout_method="dense",
        backend=backend,
    )
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        connectivity=connectivity,
        final_layout=layout,
        native_gates=natives,
    )


def test_generate_preset_passes_approximation_degree_structure():
    connectivity = nx.path_graph(4)
    passes = generate_preset_passes(connectivity, approximation_degree=0.5).passes
    # only the unroller to the native gates approximates the two-qubit gates
    assert passes[1].approximation_degree == 1.0
    assert passes[-1].approximation_degree == 0.5
    assert generate_preset_passes(connectivity).passes[-1].approximation_degree == 1.0

    # without connectivity the only unroller decomposes the circuit to the native gates
    _, unroller = generate_preset_passes(approximation_degree=0.5).passes
    assert unroller.approximation_degree == 0.5


@pytest.mark.parametrize("connectivity", [None, nx.path_graph(2)])
def test_generate_preset_passes_approximation_degree(backend, connectivity):
    circuit = Circuit(2)
    circuit.add(gates.fSim(0, 1, 0.05, 0.02))

    exact = generate_preset_passes(connectivity, backend=backend)
    approximate = generate_preset_passes(
        connectivity, approximation_degree=0.9, backend=backend
    )

    assert _count_two_qubit_gates(exact(circuit, backend=backend)[0]) > 0
    assert _count_two_qubit_gates(approximate(circuit, backend=backend)[0]) == 0


@pytest.mark.parametrize("connectivity", [None, nx.path_graph(3)])
@pytest.mark.parametrize("degree", [-0.5, 2.0])
def test_generate_preset_passes_approximation_degree_error(connectivity, degree):
    with pytest.raises(ValueError, match="approximation_degree"):
        generate_preset_passes(connectivity, approximation_degree=degree)


def test_generate_preset_passes_synthesis_options_structure():
    connectivity = nx.path_graph(4)
    passes = generate_preset_passes(connectivity, threshold=0.5, weight=1.0).passes
    # only the unroller to the native gates uses the options of the two-qubit synthesis
    assert (passes[1].threshold, passes[1].weight) == (None, None)
    assert (passes[-1].threshold, passes[-1].weight) == (0.5, 1.0)

    # without connectivity the only unroller decomposes the circuit to the native gates
    _, unroller = generate_preset_passes(threshold=0.5, weight=1.0).passes
    assert (unroller.threshold, unroller.weight) == (0.5, 1.0)


def test_generate_preset_passes_multi_controlled_options():
    default = generate_preset_passes(nx.path_graph(4)).passes[1]
    assert default.use_dirty_ancillas
    assert not default.minimize_toffolis
    assert not default.minimize_depth

    # minimizing the Toffolis follows the clean qubits, unless the depth is minimized
    zero = generate_preset_passes(nx.path_graph(4), qubits_initially_zero=True).passes[
        1
    ]
    assert zero.minimize_toffolis
    depth = generate_preset_passes(
        nx.path_graph(4), qubits_initially_zero=True, minimize_depth=True
    ).passes[1]
    assert depth.minimize_depth
    assert not depth.minimize_toffolis

    explicit = generate_preset_passes(
        nx.path_graph(4), use_dirty_ancillas=False, minimize_toffolis=True
    ).passes[1]
    assert explicit.minimize_toffolis
    assert not explicit.use_dirty_ancillas

    # without connectivity the only unroller also decomposes the multi-controlled gates
    _, unroller = generate_preset_passes(minimize_depth=True).passes
    assert unroller.minimize_depth


@pytest.mark.parametrize("connectivity", [None, nx.path_graph(3)])
def test_generate_preset_passes_minimize_toffolis_and_depth_error(connectivity):
    with pytest.raises(ValueError, match="cannot be used together"):
        generate_preset_passes(
            connectivity, minimize_toffolis=True, minimize_depth=True
        )


def test_transpile_synthesis_options(backend, monkeypatch):
    circuit = Circuit(2)
    circuit.add(gates.fSim(0, 1, 0.05, 0.02))
    connectivity = nx.path_graph(2)

    exact = transpile(circuit, connectivity, backend=backend)
    neglected = transpile(circuit, connectivity, threshold=1.0, backend=backend)
    assert _count_two_qubit_gates(exact[0]) > 0
    assert _count_two_qubit_gates(neglected[0]) == 0

    calls = []
    generate = pipeline_module.generate_preset_passes

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return generate(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "generate_preset_passes", spy)
    transpile(
        circuit,
        connectivity,
        weight=1.0,
        use_dirty_ancillas=False,
        minimize_toffolis=False,
        minimize_depth=True,
        backend=backend,
    )
    options = calls[-1]
    assert options["threshold"] is None
    assert options["weight"] == 1.0
    assert not options["use_dirty_ancillas"]
    assert options["minimize_toffolis"] is False
    assert options["minimize_depth"]


def test_generate_preset_passes_routing_trials():
    connectivity = nx.path_graph(4)
    passes = generate_preset_passes(connectivity, routing_trials=3).passes
    (router,) = [pass_ for pass_ in passes if isinstance(pass_, Sabre)]
    assert router.trials == 3

    # the other routers do not use the trials
    passes = generate_preset_passes(
        connectivity, routing_method="shortest_paths", routing_trials=0
    ).passes
    assert any(isinstance(pass_, ShortestPaths) for pass_ in passes)

    with pytest.raises(ValueError, match="trials"):
        generate_preset_passes(connectivity, routing_trials=0)


def test_transpile_routing_trials(backend, monkeypatch):
    calls = []
    generate = pipeline_module.generate_preset_passes

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return generate(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "generate_preset_passes", spy)
    circuit = Circuit(3)
    circuit.add(gates.CZ(0, 2))
    transpile(circuit, nx.path_graph(3), routing_trials=2, seed=0, backend=backend)
    assert calls[-1]["routing_trials"] == 2


def _names(passes):
    return [type(pass_).__name__ for pass_ in passes]


def test_generate_preset_passes_optimization_level_0_is_the_default():
    connectivity = nx.path_graph(4)
    default = generate_preset_passes(connectivity).passes
    level_0 = generate_preset_passes(connectivity, optimization_level=0).passes
    assert _names(level_0) == _names(default)
    assert _names(default) == [
        "Validation",
        "Unroller",
        "Preprocessing",
        "BarrierBeforeFinalMeasurements",
        "Sabre",
        "FilterGates",
        "Unroller",
    ]

    default = generate_preset_passes().passes
    assert _names(generate_preset_passes(optimization_level=0).passes) == _names(
        default
    )
    assert _names(default) == ["Validation", "Unroller"]


def test_transpile_optimization_level_0_is_the_default(backend):
    connectivity = nx.path_graph(4)
    circuit = generate_random_circuit(nqubits=4, ngates=20, seed=3)
    circuit.add(gates.M(*range(4)))

    default, default_layout = transpile(circuit, connectivity, seed=1, backend=backend)
    level_0, level_0_layout = transpile(
        circuit, connectivity, seed=1, optimization_level=0, backend=backend
    )

    assert [(gate.name, gate.qubits) for gate in level_0.queue] == [
        (gate.name, gate.qubits) for gate in default.queue
    ]
    backend.assert_allclose(level_0.unitary(backend), default.unitary(backend))
    assert level_0_layout == default_layout
    assert vars(level_0_layout) == vars(default_layout)


@pytest.mark.parametrize("connectivity", [None, nx.path_graph(3)])
@pytest.mark.parametrize("optimization_level", [3, -1, "0", None])
def test_optimization_level_error(connectivity, optimization_level):
    with pytest.raises(ValueError, match="optimization_level"):
        generate_preset_passes(connectivity, optimization_level=optimization_level)
    with pytest.raises(ValueError, match="optimization_level"):
        transpile(Circuit(2), connectivity, optimization_level=optimization_level)


def test_transpile_forwards_the_optimization_level(monkeypatch):
    calls = []
    generate = pipeline_module.generate_preset_passes

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return generate(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "generate_preset_passes", spy)
    transpile(Circuit(2), nx.path_graph(2), optimization_level=0)
    assert calls[-1]["optimization_level"] == 0


def test_generate_preset_passes_optimization_level_1_structure():
    connectivity = nx.path_graph(4)
    passes = generate_preset_passes(connectivity, optimization_level=1).passes
    assert _names(passes) == [
        "Validation",
        "Unroller",
        "InverseCancellation",
        "CommutativeCancellation",
        "Preprocessing",
        "NoiseAwarePlacer",
        "ReverseTraversal",
        "BarrierBeforeFinalMeasurements",
        "Sabre",
        "FilterGates",
        "ConsolidateBlocks",
        "Unroller",
        "FixedPoint",
    ]
    assert _names(passes[-1].optimizers) == [
        "Optimize1qGatesDecomposition",
        "InverseCancellation",
    ]
    assert passes[-1].optimizers[0].basis == set(NativeGates.default().gate_names)


def test_generate_preset_passes_optimization_level_1_directed_and_unconnected():
    # the last optimizers also act on the gates that fix the direction
    connectivity = nx.DiGraph([(0, 1), (1, 2)])
    passes = generate_preset_passes(connectivity, optimization_level=1).passes
    assert _names(passes)[-3:] == ["GateDirection", "Unroller", "FixedPoint"]

    # without connectivity there are no layout and routing, and the unroller is split
    passes = generate_preset_passes(optimization_level=1).passes
    assert _names(passes) == [
        "Validation",
        "Unroller",
        "InverseCancellation",
        "CommutativeCancellation",
        "ConsolidateBlocks",
        "Unroller",
        "FixedPoint",
    ]
    assert passes[1].min_qubits == 3
    assert passes[-2].min_qubits == 1


def test_generate_preset_passes_optimization_level_1_without_euler_basis():
    """The single-qubit optimizer is not used if the native gates have no Euler basis."""
    natives = NativeGates.PRX | NativeGates.CZ | NativeGates.M
    passes = generate_preset_passes(
        nx.path_graph(3), natives, optimization_level=1
    ).passes
    assert _names(passes[-1].optimizers) == ["InverseCancellation"]


@pytest.mark.parametrize(
    "kwargs, trials, expected",
    [
        ({}, 5, [NoiseAwarePlacer, ReverseTraversal]),
        ({"layout_trials": 2}, 2, [NoiseAwarePlacer, ReverseTraversal]),
        ({"layout_method": "reverse_traversal"}, 5, [ReverseTraversal]),
        ({"layout_method": "trivial"}, None, []),
        ({"layout_method": "dense"}, None, [Dense]),
        ({"initial_layout": [2, 1, 0]}, None, []),
        ({"routing_method": "shortest_paths"}, None, []),
        ({"routing_method": ShortestPaths()}, None, []),
        ({"routing_method": "sabre"}, 5, [NoiseAwarePlacer, ReverseTraversal]),
    ],
)
def test_generate_preset_passes_optimization_level_1_layout(kwargs, trials, expected):
    passes = generate_preset_passes(
        nx.path_graph(3), optimization_level=1, **kwargs
    ).passes
    placers = [pass_ for pass_ in passes if isinstance(pass_, Placer)]
    assert [type(pass_) for pass_ in placers] == expected
    if trials is not None:
        assert placers[-1].trials == trials


@pytest.mark.parametrize(
    "optimization_level, call_limit", [(1, (50_000, 1_000)), (2, 1_000_000)]
)
def test_generate_preset_passes_layout_search(optimization_level, call_limit):
    """The layout without SWAP gates is searched first, and kept if it already exists."""
    connectivity = nx.path_graph(4)
    passes = generate_preset_passes(
        connectivity, optimization_level=optimization_level
    ).passes
    search, placer = [pass_ for pass_ in passes if isinstance(pass_, Placer)]
    assert search.keep_layout
    assert search.call_limit == call_limit
    assert placer.keep_layout

    # without ``hardware``, the model has the connectivity and no errors
    assert nx.utils.graphs_equal(search.hardware.connectivity, connectivity)
    assert search.hardware.native_gates == NativeGates.default()
    assert not search.hardware.errors

    # a layout that the user asks for is always searched
    passes = generate_preset_passes(
        connectivity, layout_method="reverse_traversal", optimization_level=1
    ).passes
    (placer,) = [pass_ for pass_ in passes if isinstance(pass_, Placer)]
    assert not placer.keep_layout


def test_generate_preset_passes_layout_search_hardware():
    connectivity = nx.path_graph(4)
    hardware = HardwareModel((0, 1, 2, 3), connectivity)
    passes = generate_preset_passes(
        connectivity, hardware=hardware, optimization_level=1
    ).passes
    search = next(pass_ for pass_ in passes if isinstance(pass_, NoiseAwarePlacer))
    assert search.hardware is hardware

    # the hardware is only used by the layout search of the levels that have it
    passes = generate_preset_passes(connectivity, hardware=hardware).passes
    assert not any(isinstance(pass_, Placer) for pass_ in passes)

    with pytest.raises(ValueError):
        generate_preset_passes(hardware=hardware, optimization_level=1)


def test_generate_preset_passes_hardware_connectivity():
    # the couplings of the hardware can be listed in any direction, and can be more
    hardware = HardwareModel((0, 1, 2, 3, 4), [(1, 0), (2, 1), (3, 2), (4, 3)])
    for connectivity in (nx.path_graph(4), nx.DiGraph([(0, 1), (1, 2)])):
        generate_preset_passes(connectivity, hardware=hardware, optimization_level=1)

    for qubits, couplings in (
        ((0, 1, 2, 3), [(0, 1), (1, 2)]),
        ((0, 1, 2), [(0, 1), (1, 2)]),
    ):
        with pytest.raises(ConnectivityError, match="subgraph"):
            generate_preset_passes(
                nx.path_graph(4),
                hardware=HardwareModel(qubits, couplings),
                optimization_level=1,
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimization_level": 0},
        {"optimization_level": 0, "layout_method": "dense"},
        # the native gates have no Euler basis, so there are no single-qubit optimizers
        {
            "native_gates": NativeGates.PRX | NativeGates.CZ | NativeGates.M,
            "layout_method": "trivial",
            "optimization_level": 1,
        },
    ],
)
def test_generate_preset_passes_hardware_unused_warns(caplog, kwargs):
    hardware = HardwareModel((0, 1, 2), nx.path_graph(3))
    with caplog.at_level("WARNING"):
        generate_preset_passes(nx.path_graph(3), hardware=hardware, **kwargs)
    assert caplog.text.count("``hardware`` has no effect") == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimization_level": 1},
        {"optimization_level": 2},
        # the single-qubit optimizers use it even if the layout is not searched
        {"layout_method": "trivial", "optimization_level": 1},
        {"initial_layout": [2, 1, 0], "optimization_level": 2},
        {"routing_method": "shortest_paths", "optimization_level": 1},
    ],
)
def test_generate_preset_passes_hardware_used_does_not_warn(caplog, kwargs):
    hardware = HardwareModel((0, 1, 2), nx.path_graph(3))
    with caplog.at_level("WARNING"):
        generate_preset_passes(nx.path_graph(3), hardware=hardware, **kwargs)
    assert "has no effect" not in caplog.text


def test_transpile_layout_search_hardware():
    """The layout without SWAP gates of the quietest qubits is chosen."""
    couplings = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    hardware = HardwareModel(
        tuple(range(6)),
        couplings,
        errors={"2q": dict(zip(couplings, [0.05, 0.05, 0.05, 0.001, 0.001]))},
    )
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(2, 1))
    for optimization_level in (1, 2):
        transpiled, _ = transpile(
            circuit,
            hardware.connectivity,
            optimization_level=optimization_level,
            hardware=hardware,
        )
        assert not any(isinstance(gate, gates.SWAP) for gate in transpiled.queue)
        used = {
            transpiled.wire_names[qubit]
            for gate in transpiled.queue
            if len(gate.qubits) == 2
            for qubit in gate.qubits
        }
        assert used == {3, 4, 5}


@pytest.mark.parametrize("trivial", [True, False])
def test_transpile_layout_search_skips(trivial):
    """The trivial layout is kept if it needs no SWAP gates, and the search is skipped if
    a layout without them exists. The reverse traversal only runs if there is none."""
    connectivity = nx.path_graph(5)
    circuit = Circuit(5)
    for pair in [(0, 1), (1, 2)] if trivial else [(0, 2), (2, 3)]:
        circuit.add(gates.CNOT(*pair))
    ran = []
    transpiled, _ = transpile(
        circuit,
        connectivity,
        optimization_level=1,
        callback=lambda transpiler_pass, **_: ran.append(type(transpiler_pass)),
    )
    assert NoiseAwarePlacer in ran
    assert Sabre not in ran
    assert not any(isinstance(gate, gates.SWAP) for gate in transpiled.queue)
    if trivial:
        assert transpiled.wire_names == list(range(5))


def test_transpile_layout_search_falls_back(backend):
    """A circuit that needs SWAP gates has no layout without them."""
    connectivity = nx.path_graph(5)
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(0, 2))
    ran = []
    transpiled, layout = transpile(
        circuit,
        connectivity,
        optimization_level=1,
        seed=1,
        callback=lambda transpiler_pass, **_: ran.append(type(transpiler_pass)),
        backend=backend,
    )
    assert ran.index(NoiseAwarePlacer) < ran.index(ReverseTraversal)
    assert Sabre in ran
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=layout,
    )


def test_transpile_layout_search_directed():
    connectivity = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    circuit = Circuit(4)
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.CNOT(2, 3))
    transpiled, layout = transpile(
        circuit,
        connectivity,
        NativeGates.default() | NativeGates.CNOT,
        optimization_level=1,
    )
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=layout,
        native_gates=NativeGates.default() | NativeGates.CNOT,
    )


@pytest.mark.parametrize("optimization_level", [1, 2])
def test_generate_preset_passes_hardware_optimizers(optimization_level):
    """The single-qubit optimizers choose the gates by the errors of the hardware."""
    connectivity = nx.path_graph(3)
    hardware = HardwareModel((0, 1, 2), connectivity, errors={"1q": {0: 0.01}})
    for given, expected in ((None, None), (hardware, hardware)):
        passes = generate_preset_passes(
            connectivity, hardware=given, optimization_level=optimization_level
        ).passes
        optimizers = [
            optimizer
            for pass_ in passes
            if isinstance(pass_, FixedPoint)
            for optimizer in pass_.optimizers
            if isinstance(optimizer, Optimize1qGatesDecomposition)
        ]
        assert len(optimizers) == optimization_level
        assert all(optimizer.hardware is expected for optimizer in optimizers)


@pytest.mark.parametrize("optimization_level", [1, 2])
def test_transpile_hardware_single_qubit_gates(backend, optimization_level):
    natives = (
        NativeGates.RZ
        | NativeGates.RX
        | NativeGates.RY
        | NativeGates.CZ
        | NativeGates.M
    )
    connectivity = nx.path_graph(2)
    hardware = HardwareModel((0, 1), connectivity, natives, errors={"1q": {1: 0.01}})
    circuit = Circuit(2)
    for qubit in range(2):
        circuit.add(gates.H(qubit))
        circuit.add(gates.S(qubit))
        circuit.add(gates.T(qubit))
    circuit.add(gates.CZ(0, 1))

    transpiled, layout = transpile(
        circuit,
        connectivity,
        natives,
        layout_method="trivial",
        optimization_level=optimization_level,
        hardware=hardware,
        backend=backend,
    )
    assert [
        (gate.name, gate.qubits) for gate in transpiled.queue if len(gate.qubits) == 1
    ] == [
        ("rx", (0,)),
        ("ry", (0,)),
        ("rz", (1,)),
        ("ry", (1,)),
        ("rz", (1,)),
    ]
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=layout,
        native_gates=natives,
    )


@pytest.mark.parametrize(
    "routing_method, router_class",
    [
        ("sabre", Sabre),
        ("shortest_paths", ShortestPaths),
        ("none", NoRouting),
        ("star", StarConnectivityRouter),
    ],
)
def test_generate_preset_passes_barrier_around_router(
    routing_method, router_class, star_connectivity
):
    """The routers of the library keep the final measurements after the SWAP gates."""
    star = routing_method == "star"
    passes = generate_preset_passes(
        star_connectivity() if star else nx.path_graph(4),
        routing_method=routing_method,
        layout_method="star" if star else None,
    ).passes
    (position,) = [i for i, item in enumerate(passes) if isinstance(item, Router)]
    assert isinstance(passes[position], router_class)
    assert isinstance(passes[position - 1], BarrierBeforeFinalMeasurements)
    assert isinstance(passes[position + 1], FilterGates)


def test_generate_preset_passes_no_barrier_for_other_routers():
    class Custom(Router):
        def __init__(self, connectivity=None):
            self.connectivity = connectivity

        def __call__(self, circuit):
            return circuit, dict(zip(circuit.wire_names, range(circuit.nqubits)))

    passes = generate_preset_passes(nx.path_graph(3), routing_method=Custom()).passes
    assert not any(
        isinstance(item, (BarrierBeforeFinalMeasurements, FilterGates))
        for item in passes
    )


@pytest.mark.parametrize("optimization_level", [0, 1, 2])
def test_transpile_keeps_the_barriers_on_some_qubits(backend, optimization_level):
    connectivity = nx.path_graph(4)
    circuit = Circuit(4)
    circuit.add(gates.H(0))
    # its decomposition has a global phase, which the router moves to the end
    circuit.add(gates.TOFFOLI(0, 1, 2))
    circuit.add(gates.CNOT(0, 3))
    circuit.add(gates.Barrier(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.M(0, 1, 2, 3))
    transpiled, layout = transpile(
        circuit,
        connectivity,
        optimization_level=optimization_level,
        seed=1,
        backend=backend,
    )
    # the barrier that protects the final measurements is not left in the circuit, and the
    # one on some qubits is
    barriers = [gate for gate in transpiled.queue if isinstance(gate, gates.Barrier)]
    assert [len(gate.qubits) for gate in barriers] == [2]
    assert isinstance([gate for gate in transpiled.queue if gate.qubits][-1], gates.M)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=layout,
    )


@pytest.mark.parametrize("on_qubits, barriers", [(None, 1), ([1, 2, 3], 0)])
def test_transpile_removes_the_barriers_on_all_the_qubits(on_qubits, barriers):
    """The barrier of the circuit is removed too if it is on all the qubits of the hardware,
    because it is not told apart from the one that protects the measurements."""
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.Barrier(0, 1, 2))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.M(0, 1, 2))
    transpiled, _ = transpile(circuit, nx.path_graph(5), on_qubits=on_qubits, seed=1)
    found = [gate for gate in transpiled.queue if isinstance(gate, gates.Barrier)]
    # the circuit has the qubits of the hardware, so the barrier of the circuit is only on
    # all of them if the hardware has three
    assert len(found) == barriers
    assert isinstance(transpiled.queue[-1], gates.M)


@pytest.mark.parametrize("optimization_level", [0, 1, 2])
def test_generate_preset_passes_can_be_pickled(optimization_level):
    """The worker processes of ``transpile`` get the pipeline, so it cannot have lambdas."""
    passes = generate_preset_passes(
        nx.path_graph(4), optimization_level=optimization_level, seed=1
    )
    circuit = Circuit(4)
    circuit.add(gates.CNOT(0, 3))
    circuit.add(gates.M(0, 3))
    copy = pickle.loads(pickle.dumps(passes))
    first, _ = passes(circuit)
    second, _ = copy(circuit)
    assert [(gate.name, gate.qubits) for gate in first.queue] == [
        (gate.name, gate.qubits) for gate in second.queue
    ]


@pytest.mark.parametrize("optimization_level", [1, 2])
def test_generate_preset_passes_default_post_layout(optimization_level):
    hardware = HardwareModel((0, 1, 2, 3), nx.path_graph(4))
    passes = generate_preset_passes(
        nx.path_graph(4), optimization_level=optimization_level, hardware=hardware
    ).passes
    post_layout = passes[-1]
    assert isinstance(post_layout, NoiseAwarePostLayout)
    assert post_layout.hardware is hardware
    assert post_layout.trust_placer
    search = next(item for item in passes if isinstance(item, NoiseAwarePlacer))
    assert search.hardware is hardware

    # before the pass that sets the direction of the gates
    passes = generate_preset_passes(
        nx.DiGraph([(0, 1), (1, 2)]),
        optimization_level=optimization_level,
        hardware=HardwareModel((0, 1, 2), nx.DiGraph([(0, 1), (1, 2)])),
    ).passes
    position = next(
        i for i, item in enumerate(passes) if isinstance(item, NoiseAwarePostLayout)
    )
    assert isinstance(passes[position + 1], GateDirection)

    # the one of the user is the one that runs
    custom = NoiseAwarePostLayout(hardware)
    passes = generate_preset_passes(
        nx.path_graph(4),
        optimization_level=optimization_level,
        hardware=hardware,
        post_layout=custom,
    ).passes
    assert passes[-1] is custom
    assert sum(isinstance(item, NoiseAwarePostLayout) for item in passes) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"optimization_level": 0},
        {"layout_method": "trivial", "optimization_level": 1},
        {"initial_layout": [0, 1, 2], "optimization_level": 2},
        {"routing_method": "shortest_paths", "optimization_level": 1},
    ],
)
def test_generate_preset_passes_no_default_post_layout(kwargs):
    connectivity = nx.path_graph(3)
    hardware = HardwareModel((0, 1, 2), connectivity)
    passes = generate_preset_passes(connectivity, hardware=hardware, **kwargs).passes
    assert not any(isinstance(item, NoiseAwarePostLayout) for item in passes)

    # it needs a model, which the layout search makes up if there is none
    passes = generate_preset_passes(connectivity, optimization_level=2).passes
    assert not any(isinstance(item, NoiseAwarePostLayout) for item in passes)


@pytest.mark.parametrize(
    "optimization_level, hardware, keep_layout",
    [(1, True, True), (2, True, False), (2, False, True)],
)
def test_generate_preset_passes_keep_layout(optimization_level, hardware, keep_layout):
    connectivity = nx.path_graph(3)
    passes = generate_preset_passes(
        connectivity,
        optimization_level=optimization_level,
        hardware=HardwareModel((0, 1, 2), connectivity) if hardware else None,
    ).passes
    search = next(item for item in passes if isinstance(item, NoiseAwarePlacer))
    assert search.keep_layout is keep_layout


def test_transpile_default_noise_aware_layout():
    """The layout of the second level is the one with the least noise, and the post layout
    only runs if there is no layout without SWAP gates."""
    couplings = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    hardware = HardwareModel(
        tuple(range(6)),
        couplings,
        errors={"2q": dict(zip(couplings, [0.05, 0.04, 0.02, 0.001, 0.003]))},
    )

    def transpile_pairs(pairs, optimization_level):
        circuit = Circuit(6)
        for pair in pairs:
            circuit.add(gates.CNOT(*pair))
        circuit.add(gates.M(0, 1, 2))
        ran = []
        transpiled, _ = transpile(
            circuit,
            hardware.connectivity,
            optimization_level=optimization_level,
            hardware=hardware,
            seed=1,
            callback=lambda transpiler_pass, **_: ran.append(transpiler_pass),
        )
        used = {
            transpiled.wire_names[qubit]
            for gate in transpiled.queue
            if len(gate.qubits) == 2
            for qubit in gate.qubits
        }
        return used, any(isinstance(item, NoiseAwarePostLayout) for item in ran)

    # the first level keeps a layout without SWAP gates, and the second one moves it
    assert transpile_pairs([(0, 1), (1, 2)], 1) == ({0, 1, 2}, False)
    assert transpile_pairs([(0, 1), (1, 2)], 2) == ({3, 4, 5}, False)
    for optimization_level in (1, 2):
        assert transpile_pairs([(0, 2), (2, 1)], optimization_level) == (
            {3, 4, 5},
            False,
        )
        # a circuit that needs a SWAP gate is moved by the post layout
        used, post_layout_ran = transpile_pairs(
            [(0, 1), (1, 2), (0, 2)], optimization_level
        )
        assert used == {3, 4, 5} and post_layout_ran


@pytest.mark.parametrize(
    "optimization_level, layout_trials, routing_trials",
    [(0, 1, 5), (1, 5, 5), (2, 20, 20)],
)
def test_generate_preset_passes_trials(
    optimization_level, layout_trials, routing_trials
):
    passes = generate_preset_passes(
        nx.path_graph(3),
        layout_method="reverse_traversal",
        optimization_level=optimization_level,
    ).passes
    placer = next(item for item in passes if isinstance(item, ReverseTraversal))
    router = next(item for item in passes if isinstance(item, Sabre))
    assert placer.trials == layout_trials
    assert router.trials == routing_trials

    # the trials that are given are the ones used
    passes = generate_preset_passes(
        nx.path_graph(3),
        layout_method="reverse_traversal",
        optimization_level=optimization_level,
        layout_trials=2,
        routing_trials=3,
    ).passes
    placer = next(item for item in passes if isinstance(item, ReverseTraversal))
    router = next(item for item in passes if isinstance(item, Sabre))
    assert (placer.trials, router.trials) == (2, 3)


@pytest.mark.parametrize(
    "optimization_level, approximation_degree, hardware, in_loop",
    [
        (2, None, True, True),
        (2, None, False, False),
        (2, 0.99, True, False),
        (2, 1.0, True, False),
        (1, None, True, False),
    ],
)
def test_generate_preset_passes_error_aware_identity_removal(
    optimization_level, approximation_degree, hardware, in_loop
):
    """The gates are removed with the errors of their qubits, which are known after the
    layout, so the loop at the end has the pass if there are errors to use."""
    connectivity = nx.path_graph(3)
    model = HardwareModel((0, 1, 2), connectivity) if hardware else None
    passes = generate_preset_passes(
        connectivity,
        optimization_level=optimization_level,
        approximation_degree=approximation_degree,
        hardware=model,
    ).passes
    loops = [item.optimizers for item in passes if isinstance(item, FixedPoint)]
    found = [
        optimizer
        for loop in loops
        for optimizer in loop
        if isinstance(optimizer, RemoveIdentityEquivalent)
    ]
    assert len(found) == int(in_loop)
    if in_loop:
        assert found[0] is loops[0][0]
        assert found[0].hardware is model
        assert found[0].approximation_degree is None

    # the one before the layout does not know the qubits
    before = [item for item in passes if isinstance(item, RemoveIdentityEquivalent)]
    assert len(before) == int(optimization_level == 2)
    assert all(item.hardware is None for item in before)

    # the unroller does not approximate with the errors, and it is not given ``None``
    if approximation_degree is None:
        unrollers = [item for item in passes if isinstance(item, Unroller)]
        assert all(item.approximation_degree == 1.0 for item in unrollers)


@pytest.mark.parametrize(
    "optimization_level, approximation_degree, remaining",
    [(2, None, [1]), (2, 1.0, [0, 1]), (1, None, [0, 1])],
)
def test_transpile_error_aware_identity_removal(
    backend, optimization_level, approximation_degree, remaining
):
    natives = (
        NativeGates.RZ
        | NativeGates.RX
        | NativeGates.RY
        | NativeGates.CZ
        | NativeGates.M
    )
    connectivity = nx.path_graph(2)
    # the rotation is worth executing on the second qubit, where the error is lower
    hardware = HardwareModel(
        (0, 1), connectivity, natives, errors={"1q": {0: 1e-3, 1: 1e-9}}
    )
    circuit = Circuit(2)
    circuit.add(gates.RX(0, 1e-3))
    circuit.add(gates.RX(1, 1e-3))
    circuit.add(gates.M(0, 1))
    transpiled, _ = transpile(
        circuit,
        connectivity,
        natives,
        layout_method="trivial",
        optimization_level=optimization_level,
        approximation_degree=approximation_degree,
        hardware=hardware,
        backend=backend,
    )
    found = [
        transpiled.wire_names[gate.qubits[0]]
        for gate in transpiled.queue
        if gate.name == "rx"
    ]
    assert found == remaining


def test_passes_elide_permutations_without_router():
    """The permutation of the removed SWAP gates is the one of the final layout."""
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.CNOT(1, 2))
    circuit.add(gates.M(0, 1, 2))
    transpiled, layout = Passes([ElidePermutations()])(circuit)
    assert not any(isinstance(gate, gates.SWAP) for gate in transpiled.queue)
    assert dict(layout) == {0: 1, 1: 0, 2: 2}
    assert layout.routing_permutation == [1, 0, 2]
    assert layout.initial_layout == [0, 1, 2]
    assert_circuit_equivalence(
        original_circuit=circuit, transpiled_circuit=transpiled, final_layout=layout
    )


@pytest.mark.parametrize("router", [ShortestPaths, Sabre])
def test_passes_elide_permutations_with_router(backend, router):
    """The permutation is composed with the one of the router, and with the auxiliary
    qubits that the preprocessing adds."""
    connectivity = nx.path_graph(5)
    circuit = Circuit(4)
    circuit.add(gates.SWAP(0, 3))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.CNOT(2, 3))
    circuit.add(gates.SWAP(1, 2))
    circuit.add(gates.CNOT(0, 3))
    circuit.add(gates.H(1))
    circuit.add(gates.M(0, 1, 2, 3))
    pipeline = Passes(
        [ElidePermutations(), Preprocessing(), router(seed=1)],
        connectivity=connectivity,
    )
    transpiled, layout = pipeline(circuit, backend=backend)
    assert layout.nqubits_input == 4
    assert layout.auxiliary_qubits == [4]
    assert layout.initial_layout == list(range(5))
    # there is no unroller, so the gates of the circuit and the SWAP gates of the router
    # are native
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=layout,
        native_gates=NativeGates.default()
        | NativeGates.SWAP
        | NativeGates.CNOT
        | NativeGates.H,
    )


def test_passes_elide_permutations_are_composed():
    """The permutations of several passes are composed, in the order that they run."""

    class AppendSwap(Optimizer):
        def __call__(self, circuit):
            new = Circuit(**circuit.init_kwargs)
            new.add(circuit.queue)
            new.add(gates.SWAP(0, 2))
            return new

    circuit = Circuit(3)
    circuit.add(gates.SWAP(0, 1))
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(1, 2))
    # The SWAP gate that the other pass adds is on the first and third qubits of the circuit
    # that the first pass returns, whose first qubit corresponds to the second qubit of the
    # original circuit. So it is a SWAP gate on the second and third qubits of the original.
    expected = Circuit(3)
    expected.add(circuit.queue)
    expected.add(gates.SWAP(1, 2))

    pipeline = Passes([ElidePermutations(), AppendSwap(), ElidePermutations()])
    transpiled, layout = pipeline(circuit)
    assert not any(isinstance(gate, gates.SWAP) for gate in transpiled.queue)
    assert layout.routing_permutation != [1, 0, 2]
    assert_circuit_equivalence(
        original_circuit=expected, transpiled_circuit=transpiled, final_layout=layout
    )


def test_passes_elide_permutations_after_router():
    pipeline = Passes(
        [Preprocessing(), ShortestPaths(seed=1), ElidePermutations()],
        connectivity=nx.path_graph(3),
    )
    circuit = Circuit(3)
    circuit.add(gates.CNOT(0, 2))
    with pytest.raises(TranspilerPipelineError, match="before it"):
        pipeline(circuit)


@pytest.mark.parametrize(
    "routing_method", [None, "sabre", "shortest_paths", Sabre(seed=1)]
)
def test_generate_preset_passes_elide_permutations(routing_method):
    passes = generate_preset_passes(
        nx.path_graph(4), routing_method=routing_method, optimization_level=2
    ).passes
    # first of the optimizers, because the others work with the gates that remain
    assert _names(passes)[:3] == ["Validation", "Unroller", "ElidePermutations"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routing_method": "none"},
        {"routing_method": NoRouting()},
        {"optimization_level": 1},
        {"optimization_level": 0},
        {"connectivity": None},
    ],
)
def test_generate_preset_passes_no_elide_permutations(kwargs):
    kwargs = {"connectivity": nx.path_graph(4), "optimization_level": 2, **kwargs}
    passes = generate_preset_passes(**kwargs).passes
    assert "ElidePermutations" not in _names(passes)


def test_generate_preset_passes_consolidate_before_the_layout():
    for optimization_level, connectivity, routing_method, expected in (
        (1, nx.path_graph(4), None, 1),
        (2, nx.path_graph(4), None, 2),
        (2, nx.path_graph(4), "none", 2),
        # there is nothing to place and route, so the other one is enough
        (2, None, None, 1),
    ):
        passes = generate_preset_passes(
            connectivity,
            routing_method=routing_method,
            optimization_level=optimization_level,
        ).passes
        positions = [
            position
            for position, item in enumerate(passes)
            if isinstance(item, ConsolidateBlocks)
        ]
        assert len(positions) == expected
        if expected == 2:
            # the blocks that are a SWAP gate are removed after the consolidation, unless
            # the circuit cannot be routed
            consolidate = passes[positions[0]]
            if routing_method == "none":
                assert not consolidate.split_swaps
                assert isinstance(passes[positions[0] + 1], Preprocessing)
            else:
                assert consolidate.split_swaps
                assert isinstance(passes[positions[0] + 1], ElidePermutations)
                assert isinstance(passes[positions[0] + 2], Preprocessing)


def test_generate_preset_passes_loop_commutative_cancellation():
    natives = NativeGates.I | NativeGates.RZ | NativeGates.SX | NativeGates.CNOT
    for given, basis in (
        (None, set(NativeGates.default().gate_names)),
        (natives, set(natives.gate_names)),
    ):
        for optimization_level in (1, 2):
            passes = generate_preset_passes(
                nx.path_graph(3), given, optimization_level=optimization_level
            ).passes
            loops = [item.optimizers for item in passes if isinstance(item, FixedPoint)]
            found = [
                [isinstance(optimizer, CommutativeCancellation) for optimizer in loop]
                for loop in loops
            ]
            if optimization_level == 1:
                assert not any(any(loop) for loop in found)
            else:
                # only the first loop, which has the native gates as the allowed ones
                assert sum(any(loop) for loop in found) == 1 and any(found[0])
                assert loops[0][found[0].index(True)].basis == basis


def test_transpile_chain_on_a_grid():
    """A chain of qubits has a layout where it needs no SWAP gates, even on a large grid, so
    the first level, with its small search, finds it, and it has the least entangling gates.
    """
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5))
    circuit = Circuit(25)
    circuit.add(gates.H(0))
    for qubit in range(24):
        circuit.add(gates.CNOT(qubit, qubit + 1))
    transpiled, layout = transpile(circuit, grid, optimization_level=1, seed=0)
    assert sum(isinstance(gate, gates.CZ) for gate in transpiled.queue) == 24
    assert layout.routing_permutation == list(range(25))


def test_transpile_elides_the_swap_gates(backend):
    connectivity = nx.path_graph(4)
    circuit = Circuit(4)
    for qubits in ((0, 1), (1, 2), (2, 3)):
        circuit.add(gates.SWAP(*qubits))
    for qubit in range(4):
        circuit.add(gates.H(qubit))
    circuit.add(gates.M(*range(4)))

    results = {}
    for optimization_level in (1, 2):
        transpiled, layout = transpile(
            circuit,
            connectivity,
            optimization_level=optimization_level,
            seed=1,
            backend=backend,
        )
        assert_transpiling(
            original_circuit=circuit,
            transpiled_circuit=transpiled,
            connectivity=connectivity,
            final_layout=layout,
        )
        results[optimization_level] = (
            sum(isinstance(gate, gates.CZ) for gate in transpiled.queue),
            layout.routing_permutation,
        )
    # the SWAP gates of the first level are three CZ gates each, and the second level
    # has none, because the qubits that they exchange are the ones of the final layout
    assert results[1] == (9, [0, 1, 2, 3])
    assert results[2] == (0, [1, 2, 3, 0])


@pytest.mark.parametrize("seed", range(4))
def test_transpile_elide_permutations_is_equivalent(backend, seed):
    """The final layout stays right with the router, the post layout and the auxiliary
    qubits of a hardware that is larger than the circuit."""
    couplings = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    hardware = HardwareModel(
        tuple(range(6)),
        couplings,
        errors={"2q": dict(zip(couplings, [0.05, 0.04, 0.02, 0.001, 0.003]))},
    )
    rng = random.Random(seed)
    circuit = Circuit(4)
    for _ in range(20):
        kind = rng.random()
        first, second = rng.sample(range(4), 2)
        if kind < 0.4:
            circuit.add(gates.SWAP(first, second))
        elif kind < 0.8:
            circuit.add(gates.CNOT(first, second))
        else:
            circuit.add(gates.H(first))
    circuit.add(gates.M(*range(4)))
    transpiled, layout = transpile(
        circuit,
        hardware.connectivity,
        optimization_level=2,
        hardware=hardware,
        seed=seed,
        backend=backend,
    )
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=hardware.connectivity,
        final_layout=layout,
    )


def test_transpile_elides_swap_equivalent_blocks(backend):
    """Three CNOT gates that are a SWAP gate cost no CZ gate at the second level."""
    connectivity = nx.path_graph(3)
    circuit = Circuit(3)
    circuit.add(
        [gates.CNOT(0, 1), gates.CNOT(1, 0), gates.CNOT(0, 1), gates.CNOT(1, 2)]
    )
    circuit.add(gates.M(0, 1, 2))
    cz_gates = {}
    for optimization_level in (1, 2):
        transpiled, layout = transpile(
            circuit,
            connectivity,
            optimization_level=optimization_level,
            seed=1,
            backend=backend,
        )
        assert_transpiling(
            original_circuit=circuit,
            transpiled_circuit=transpiled,
            connectivity=connectivity,
            final_layout=layout,
        )
        cz_gates[optimization_level] = sum(
            isinstance(gate, gates.CZ) for gate in transpiled.queue
        )
    assert cz_gates[2] == 1
    assert cz_gates[1] > cz_gates[2]


def test_generate_preset_passes_noise_router():
    """The routers weigh the SWAP gates with the errors from the second level."""
    connectivity = nx.path_graph(4)
    couplings = list(connectivity.edges)
    hardware = HardwareModel(
        tuple(range(4)),
        couplings,
        errors={"2q": dict(zip(couplings, [0.05, 0.01, 0.02]))},
    )

    def routers(**kwargs):
        passes = generate_preset_passes(connectivity, **kwargs).passes
        (router,) = [item for item in passes if isinstance(item, Sabre)]
        (placer,) = [item for item in passes if isinstance(item, ReverseTraversal)]
        return router, placer.routing_algorithm

    for optimization_level, heuristic in ((1, "decay"), (2, "noise")):
        for router in routers(optimization_level=optimization_level, hardware=hardware):
            assert router.heuristic == heuristic
            assert (router.hardware is hardware) == (heuristic == "noise")

    # there are no errors to weigh without the hardware model
    for router in routers(optimization_level=2):
        assert router.heuristic == "decay"
        assert router.hardware is None

    # a router that is given is not changed, nor the one of the placer that is given
    own = Sabre(heuristic="basic")
    placer = ReverseTraversal(Sabre(heuristic="basic"))
    passes = generate_preset_passes(
        connectivity,
        optimization_level=2,
        hardware=hardware,
        routing_method=own,
        layout_method=placer,
    ).passes
    assert any(item is own for item in passes)
    assert any(item is placer for item in passes)
    assert own.heuristic == placer.routing_algorithm.heuristic == "basic"


@pytest.mark.parametrize("seed", range(3))
def test_transpile_noise_router_is_equivalent(backend, seed):
    """The circuit stays equivalent with the noise router in the whole pipeline."""
    connectivity = nx.cycle_graph(6)
    couplings = list(connectivity.edges)
    noisy = {(0, 1), (1, 2), (2, 3)}
    hardware = HardwareModel(
        tuple(range(6)),
        couplings,
        errors={"2q": {edge: 0.2 if edge in noisy else 0.001 for edge in couplings}},
    )
    circuit = Circuit(6)
    circuit.add([gates.CNOT(0, 3), gates.CNOT(1, 4), gates.CNOT(2, 5)])
    circuit.add(gates.M(*range(6)))
    transpiled, layout = transpile(
        circuit,
        connectivity,
        optimization_level=2,
        hardware=hardware,
        seed=seed,
        backend=backend,
    )
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=layout,
    )


def test_generate_preset_passes_level_0_layout_trials():
    passes = generate_preset_passes(
        nx.path_graph(3), layout_method="reverse_traversal", optimization_level=0
    ).passes
    (placer,) = [pass_ for pass_ in passes if isinstance(pass_, ReverseTraversal)]
    assert placer.trials == 1


@pytest.mark.parametrize("optimization_level", [0, 1, 2])
def test_generate_preset_passes_sampling_only(optimization_level):
    default = generate_preset_passes(
        nx.path_graph(3), optimization_level=optimization_level
    )
    sampling = generate_preset_passes(
        nx.path_graph(3), optimization_level=optimization_level, sampling_only=True
    )
    if optimization_level == 0:
        # there are no optimizers
        assert _names(sampling.passes) == _names(default.passes)
    else:
        # the SWAP gates are removed first at the second level
        start = 3 if optimization_level == 2 else 2
        assert _names(sampling.passes)[start : start + 3] == [
            "OptimizeSwapBeforeMeasure",
            "RemoveDiagonalGatesBeforeMeasurement",
            "RemoveFinalReset",
        ]
        assert _names(sampling.passes)[start + 3 :] == _names(default.passes)[start:]


@pytest.mark.parametrize(
    "natives",
    [
        NativeGates.default(),
        NativeGates.I
        | NativeGates.RZ
        | NativeGates.SX
        | NativeGates.X
        | NativeGates.CNOT,
        NativeGates.U3 | NativeGates.CZ | NativeGates.M,
    ],
    ids=["default", "SX-CNOT", "U3"],
)
@pytest.mark.parametrize("directed", [False, True])
def test_generate_preset_passes_optimization_level_1(backend, natives, directed):
    connectivity = (
        nx.DiGraph([(0, 1), (2, 1), (2, 3), (4, 3), (4, 0)])
        if directed
        else nx.path_graph(5)
    )
    circuit = generate_random_circuit(nqubits=5, ngates=30, seed=4)
    circuit.add(gates.M(*range(5)))

    pipeline = generate_preset_passes(
        connectivity, natives, seed=1, optimization_level=1, backend=backend
    )
    transpiled, final_layout = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=natives,
    )

    baseline, _ = generate_preset_passes(
        connectivity, natives, seed=1, backend=backend
    )(circuit, backend=backend)
    assert _count_two_qubit_gates(transpiled) <= _count_two_qubit_gates(baseline)
    assert transpiled.ngates < baseline.ngates


def test_generate_preset_passes_optimization_level_1_without_connectivity(backend):
    circuit = Circuit(3)
    circuit.add(
        [gates.H(0), gates.RZ(0, 0.3), gates.CNOT(0, 1), gates.RZ(0, 0.4), gates.H(0)]
    )
    circuit.add(gates.TOFFOLI(0, 1, 2))
    baseline = generate_preset_passes(backend=backend)(circuit, backend=backend)[0]
    optimized = generate_preset_passes(optimization_level=1, backend=backend)(
        circuit, backend=backend
    )[0]

    assert optimized.ngates < baseline.ngates
    first = baseline.unitary(backend)
    second = optimized.unitary(backend)
    overlap = backend.abs(backend.trace(backend.dagger(first) @ second)) / len(first)
    backend.assert_allclose(overlap, 1.0, atol=1e-8)


def test_generate_preset_passes_sampling_only_removes_swaps(backend):
    circuit = Circuit(3)
    circuit.add([gates.RY(0, 0.8), gates.RY(1, 1.3), gates.SWAP(0, 1), gates.M(0, 1)])
    connectivity = nx.path_graph(3)
    kwargs = {"optimization_level": 1, "seed": 0, "backend": backend}

    exact, _ = generate_preset_passes(connectivity, **kwargs)(circuit, backend=backend)
    sampling, _ = generate_preset_passes(connectivity, sampling_only=True, **kwargs)(
        circuit, backend=backend
    )

    assert sum(isinstance(gate, gates.CZ) for gate in exact.queue) == 3
    assert sum(isinstance(gate, gates.CZ) for gate in sampling.queue) == 0


def test_transpile_optimization_level_1(backend, monkeypatch):
    calls = []
    generate = pipeline_module.generate_preset_passes

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return generate(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "generate_preset_passes", spy)
    circuit = Circuit(3)
    circuit.add(
        [gates.H(0), gates.RZ(0, 0.3), gates.CNOT(0, 1), gates.RZ(0, 0.4), gates.M(0)]
    )
    exact, _ = transpile(circuit, nx.path_graph(3), seed=0, backend=backend)
    optimized, _ = transpile(
        [circuit],
        nx.path_graph(3),
        seed=0,
        optimization_level=1,
        sampling_only=True,
        backend=backend,
    )[0]

    assert calls[-1]["optimization_level"] == 1
    assert calls[-1]["sampling_only"]
    assert optimized.ngates < exact.ngates


def test_transpile_sampling_only_warns_without_measurements(caplog):
    with_measurement = Circuit(2)
    with_measurement.add([gates.H(0), gates.M(0)])
    without_measurement = Circuit(2)
    without_measurement.add(gates.H(0))
    connectivity = nx.path_graph(2)

    with caplog.at_level("WARNING"):
        transpile(
            with_measurement, connectivity, optimization_level=1, sampling_only=True
        )
    assert "sampling_only" not in caplog.text

    with caplog.at_level("WARNING"):
        transpile(
            [with_measurement, without_measurement],
            connectivity,
            optimization_level=1,
            sampling_only=True,
        )
    assert caplog.text.count("sampling_only") == 1

    caplog.clear()
    with caplog.at_level("WARNING"):
        transpile(without_measurement, connectivity, sampling_only=True)
        transpile(without_measurement, connectivity, optimization_level=1)
    assert "sampling_only" not in caplog.text


def test_generate_preset_passes_optimization_level_2_structure():
    connectivity = nx.path_graph(4)
    passes = generate_preset_passes(connectivity, optimization_level=2).passes
    assert _names(passes) == [
        "Validation",
        "Unroller",
        "ElidePermutations",
        "InverseCancellation",
        "CommutativeCancellation",
        "RemoveIdentityEquivalent",
        "ConsolidateBlocks",
        "ElidePermutations",
        "Preprocessing",
        "NoiseAwarePlacer",
        "ReverseTraversal",
        "BarrierBeforeFinalMeasurements",
        "Sabre",
        "FilterGates",
        "ConsolidateBlocks",
        "CommutativeCancellation",
        "Unroller",
        "FixedPoint",
        "FixedPoint",
    ]
    assert _names(passes[-2].optimizers) == [
        "Optimize1qGatesDecomposition",
        "CommutativeCancellation",
        "InverseCancellation",
    ]
    assert _names(passes[-1].optimizers) == [
        "Optimize1qGatesDecomposition",
        "Optimize1qGatesSimpleCommutation",
        "InverseCancellation",
    ]
    assert not passes[-2].optimizers[-1].commutation
    assert passes[-1].optimizers[-1].commutation
    basis = set(NativeGates.default().gate_names)
    assert passes[-1].optimizers[0].basis == basis
    assert passes[-1].optimizers[1].basis == basis
    # the loop only merges the rotations into native gates, while the one before the last
    # unroller can merge them into any gate
    assert passes[-2].optimizers[1].basis == basis
    assert passes[-4].basis is None


@pytest.mark.parametrize("approximation_degree", [1.0, 0.9])
def test_generate_preset_passes_optimization_level_2_approximation_degree(
    approximation_degree,
):
    passes = generate_preset_passes(
        nx.path_graph(3),
        optimization_level=2,
        approximation_degree=approximation_degree,
    ).passes
    (identity,) = [
        pass_ for pass_ in passes if isinstance(pass_, RemoveIdentityEquivalent)
    ]
    assert identity.approximation_degree == approximation_degree


def test_generate_preset_passes_optimization_level_2_directed_and_unconnected():
    # the last loops also act on the gates that fix the direction
    connectivity = nx.DiGraph([(0, 1), (1, 2)])
    passes = generate_preset_passes(connectivity, optimization_level=2).passes
    assert _names(passes)[-4:] == [
        "GateDirection",
        "Unroller",
        "FixedPoint",
        "FixedPoint",
    ]

    # without connectivity there are no layout and routing, and the unroller is split
    passes = generate_preset_passes(optimization_level=2).passes
    assert _names(passes) == [
        "Validation",
        "Unroller",
        "InverseCancellation",
        "CommutativeCancellation",
        "RemoveIdentityEquivalent",
        "ConsolidateBlocks",
        "CommutativeCancellation",
        "Unroller",
        "FixedPoint",
        "FixedPoint",
    ]
    assert passes[1].min_qubits == 3
    assert passes[-3].min_qubits == 1


def test_generate_preset_passes_optimization_level_2_without_euler_basis():
    """The single-qubit optimizers are not used if the native gates have no Euler basis."""
    natives = NativeGates.PRX | NativeGates.CZ | NativeGates.M
    passes = generate_preset_passes(
        nx.path_graph(3), natives, optimization_level=2
    ).passes
    assert _names(passes).count("FixedPoint") == 1
    assert _names(passes[-1].optimizers) == [
        "CommutativeCancellation",
        "InverseCancellation",
    ]
    assert "Optimize1qGatesSimpleCommutation" not in _names(passes)


def test_generate_preset_passes_optimization_level_2_layout():
    passes = generate_preset_passes(nx.path_graph(3), optimization_level=2).passes
    search, placer = [pass_ for pass_ in passes if isinstance(pass_, Placer)]
    assert isinstance(search, NoiseAwarePlacer)
    assert isinstance(placer, ReverseTraversal)
    assert placer.trials == 20

    passes = generate_preset_passes(
        nx.path_graph(3), optimization_level=2, layout_method="trivial"
    ).passes
    assert not any(isinstance(pass_, Placer) for pass_ in passes)


@pytest.mark.parametrize(
    "natives",
    [
        NativeGates.default(),
        NativeGates.I
        | NativeGates.RZ
        | NativeGates.SX
        | NativeGates.X
        | NativeGates.CNOT,
        NativeGates.U3 | NativeGates.CZ | NativeGates.M,
    ],
    ids=["default", "SX-CNOT", "U3"],
)
@pytest.mark.parametrize("directed", [False, True])
def test_generate_preset_passes_optimization_level_2(backend, natives, directed):
    connectivity = (
        nx.DiGraph([(0, 1), (2, 1), (2, 3), (4, 3), (4, 0)])
        if directed
        else nx.path_graph(5)
    )
    circuit = generate_random_circuit(nqubits=5, ngates=30, seed=4)
    circuit.add(gates.M(*range(5)))

    pipeline = generate_preset_passes(
        connectivity, natives, seed=1, optimization_level=2, backend=backend
    )
    transpiled, final_layout = pipeline(circuit, backend=backend)
    assert pipeline.is_satisfied(transpiled)
    assert_transpiling(
        original_circuit=circuit,
        transpiled_circuit=transpiled,
        connectivity=connectivity,
        final_layout=final_layout,
        native_gates=natives,
    )

    baseline, _ = generate_preset_passes(
        connectivity, natives, seed=1, backend=backend
    )(circuit, backend=backend)
    assert _count_two_qubit_gates(transpiled) <= _count_two_qubit_gates(baseline)
    assert transpiled.ngates < baseline.ngates


def test_generate_preset_passes_optimization_level_2_commutes_single_qubit_gates(
    backend,
):
    """The second loop merges gates that the first one cannot, as they are not adjacent."""
    circuit = Circuit(2)
    circuit.add(
        [
            gates.RX(0, 0.4),
            gates.H(0),
            gates.RZ(0, 0.3),
            gates.CZ(0, 1),
            gates.RZ(0, 0.4),
            gates.H(0),
            gates.RX(0, 0.2),
        ]
    )
    kwargs = {"seed": 0, "layout_method": "trivial", "backend": backend}
    connectivity = nx.path_graph(2)
    level_1, _ = generate_preset_passes(connectivity, optimization_level=1, **kwargs)(
        circuit, backend=backend
    )
    level_2, _ = generate_preset_passes(connectivity, optimization_level=2, **kwargs)(
        circuit, backend=backend
    )

    assert level_2.ngates < level_1.ngates
    first = level_1.unitary(backend)
    second = level_2.unitary(backend)
    overlap = backend.abs(backend.trace(backend.dagger(first) @ second)) / len(first)
    backend.assert_allclose(overlap, 1.0, atol=1e-8)


def test_generate_preset_passes_optimization_level_2_without_connectivity(backend):
    circuit = Circuit(3)
    circuit.add(
        [gates.H(0), gates.RZ(0, 0.3), gates.CNOT(0, 1), gates.RZ(0, 0.4), gates.H(0)]
    )
    circuit.add(gates.TOFFOLI(0, 1, 2))
    level_1 = generate_preset_passes(optimization_level=1, backend=backend)(
        circuit, backend=backend
    )[0]
    level_2 = generate_preset_passes(optimization_level=2, backend=backend)(
        circuit, backend=backend
    )[0]

    assert level_2.ngates <= level_1.ngates
    first = level_1.unitary(backend)
    second = level_2.unitary(backend)
    overlap = backend.abs(backend.trace(backend.dagger(first) @ second)) / len(first)
    backend.assert_allclose(overlap, 1.0, atol=1e-8)


def test_transpile_optimization_level_2(backend, monkeypatch):
    calls = []
    generate = pipeline_module.generate_preset_passes

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return generate(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "generate_preset_passes", spy)
    circuit = Circuit(3)
    circuit.add(
        [gates.H(0), gates.RZ(0, 0.3), gates.CNOT(0, 1), gates.RZ(0, 0.4), gates.M(0)]
    )
    exact, _ = transpile(circuit, nx.path_graph(3), seed=0, backend=backend)
    optimized, _ = transpile(
        circuit,
        nx.path_graph(3),
        seed=0,
        optimization_level=2,
        approximation_degree=0.95,
    )

    assert calls[-1]["optimization_level"] == 2
    assert calls[-1]["approximation_degree"] == 0.95
    assert optimized.ngates < exact.ngates
