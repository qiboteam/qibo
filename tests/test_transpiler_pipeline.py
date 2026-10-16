import random

import networkx as nx
import numpy as np
import pytest
from qiskit.transpiler import PassManager, passes

from qibo import Circuit, gates
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
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import (
    Passes,
    generate_preset_passes,
    restrict_connectivity_qubits,
)
from qibo.transpiler.placer import Random, ReverseTraversal
from qibo.transpiler.qiskit import QiskitPasses
from qibo.transpiler.router import Sabre, ShortestPaths
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
