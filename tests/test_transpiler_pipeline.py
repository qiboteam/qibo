import random

import networkx as nx
import numpy as np
import pytest
from qiskit.transpiler import PassManager, passes

from qibo import Circuit, gates
from qibo.backends import NumpyBackend
from qibo.quantum_info import random_unitary
from qibo.transpiler import unroller as unroller_module
from qibo.transpiler._exceptions import (
    BlockingError,
    ConnectivityError,
    PlacementError,
    TranspilerPipelineError,
)
from qibo.transpiler.abstract import Optimizer
from qibo.transpiler.asserts import (
    assert_circuit_equivalence,
    assert_decomposition,
    assert_transpiling,
)
from qibo.transpiler.optimizer import GateDirection, Preprocessing
from qibo.transpiler.pipeline import (
    Passes,
    generate_preset_passes,
    restrict_connectivity_qubits,
)
from qibo.transpiler.placer import (
    Random,
    ReverseTraversal,
    StarConnectivityPlacer,
    Subgraph,
)
from qibo.transpiler.qiskit import QiskitPasses
from qibo.transpiler.router import Sabre, ShortestPaths, StarConnectivityRouter
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
        Unroller,
        Preprocessing,
        Sabre,
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
    unroller, preprocessing, router, translator = pipeline.passes
    assert isinstance(unroller, Unroller) and unroller.min_qubits == 3
    assert isinstance(preprocessing, Preprocessing)
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


@pytest.mark.parametrize("initial_layout", [[0, 1], [0, 0, 1], [0, 1, 9]])
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
    assert len(pipeline.passes) == 1
    transpiled_circuit, final_layout = pipeline(circuit, backend=backend)
    assert final_layout is None
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
    assert default[0].use_dirty_ancillas
    assert not default[0].qubits_initially_zero and not default[0].minimize_toffolis
    assert not default[-1].use_dirty_ancillas

    zero = generate_preset_passes(nx.path_graph(5), qubits_initially_zero=True).passes
    assert zero[0].use_dirty_ancillas
    assert zero[0].qubits_initially_zero and zero[0].minimize_toffolis
    assert not zero[-1].qubits_initially_zero

    # without connectivity the only unroller decomposes the multi-controlled gates
    (unroller,) = generate_preset_passes(qubits_initially_zero=True).passes
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
        assert len(pipeline.passes) == 4
    else:
        assert len(pipeline.passes) == 5
        assert isinstance(pipeline.passes[1], Preprocessing)
        assert isinstance(pipeline.passes[2], placer_class)
        assert isinstance(pipeline.passes[3], Sabre)
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
    assert pipeline.passes[2] is placer
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
    placer = pipeline.passes[2]
    assert placer.routing_algorithm._rng.getstate() == random.Random(7).getstate()

    placer = generate_preset_passes(
        connectivity, seed=7, layout_method="random"
    ).passes[2]
    assert placer.seed == 7
    router = generate_preset_passes(
        connectivity, seed=7, routing_method="shortest_paths"
    ).passes[-2]
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
    router = pipeline.passes[2]
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
    assert pipeline.passes[2] is router
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
    assert isinstance(pipeline.passes[2], ShortestPaths)
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
    assert isinstance(pipeline.passes[2], StarConnectivityPlacer)
    assert isinstance(pipeline.passes[3], StarConnectivityRouter)
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


@pytest.mark.parametrize("layout_method", ["dense", "Random", 3])
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
