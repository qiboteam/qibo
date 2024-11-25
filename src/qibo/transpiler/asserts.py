from typing import Optional

import networkx as nx
import numpy as np

from qibo import gates
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.quantum_info.random_ensembles import random_statevector
from qibo.transpiler._exceptions import (
    ConnectivityError,
    DecompositionError,
    PlacementError,
    TranspilerPipelineError,
)
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.unroller import NativeGates


def assert_transpiling(
    original_circuit: Circuit,
    transpiled_circuit: Circuit,
    connectivity: nx.Graph,
    final_layout: dict,
    native_gates: NativeGates = NativeGates.default(),
    check_circuit_equivalence=True,
):
    """Check that all transpiler passes have been executed correctly.

    Args:
        original_circuit (qibo.models.Circuit): circuit before transpiling.
        transpiled_circuit (qibo.models.Circuit): circuit after transpiling.
        connectivity (networkx.Graph): chip qubits connectivity.
        final_layout (dict): final physical-logical qubit mapping.
        native_gates (NativeGates): native gates supported by the hardware.
        check_circuit_equivalence (Bool): use simulations to check if the transpiled circuit is the same as the original.
    """
    assert_connectivity(circuit=transpiled_circuit, connectivity=connectivity)
    assert_decomposition(
        circuit=transpiled_circuit,
        native_gates=native_gates,
    )
    if original_circuit.nqubits != transpiled_circuit.nqubits:
        qubit_matcher = Preprocessing(connectivity=connectivity)
        original_circuit = qubit_matcher(circuit=original_circuit)
    assert_placement(circuit=original_circuit, connectivity=connectivity)
    assert_placement(circuit=transpiled_circuit, connectivity=connectivity)
    if check_circuit_equivalence:
        assert_circuit_equivalence(
            original_circuit=original_circuit,
            transpiled_circuit=transpiled_circuit,
            final_map=final_layout,
        )


def assert_circuit_equivalence(
    original_circuit: Circuit,
    transpiled_circuit: Circuit,
    final_map: dict,
    test_states: Optional[list] = None,
    ntests: int = 3,
):
    """Checks that the transpiled circuit agrees with the original using simulation.

    Args:
        original_circuit (:class:`qibo.models.circuit.Circuit`): Original circuit.
        transpiled_circuit (:class:`qibo.models.circuit.Circuit`): Transpiled circuit.
        final_map (dict): logical-physical qubit mapping after routing.
        test_states (list, optional): states on which the test is performed.
            If ``None``, ``ntests`` random states will be tested. Defauts to ``None``.
        ntests (int, optional): number of random states tested. Defauts to :math:`3`.
    """
    backend = NumpyBackend()
    if transpiled_circuit.nqubits != original_circuit.nqubits:
        raise_error(
            ValueError,
            "Transpiled and original circuit do not have the same number of qubits.",
        )

    if test_states is None:
        test_states = [
            random_statevector(dims=2**original_circuit.nqubits, backend=backend)
            for _ in range(ntests)
        ]

    ordering = list(final_map.values())

    for i, state in enumerate(test_states):
        target_state = backend.execute_circuit(
            original_circuit, initial_state=state
        ).state()
        final_state = backend.execute_circuit(
            transpiled_circuit,
            initial_state=state,
        ).state()
        final_state = _transpose_qubits(final_state, ordering)
        fidelity = np.abs(np.dot(np.conj(target_state), final_state))
        try:
            np.testing.assert_allclose(fidelity, 1.0)
        except AssertionError:
            raise_error(TranspilerPipelineError, "Circuit equivalence not satisfied.")


def _transpose_qubits(state: np.ndarray, qubits_ordering: np.ndarray):
    """Reorders qubits of a given state vector.

    Args:
        state (np.ndarray): final state of the circuit.
        qubits_ordering (np.ndarray): final qubit ordering.
    """
    original_shape = state.shape
    state = np.reshape(state, len(qubits_ordering) * (2,))
    state = np.transpose(state, qubits_ordering)
    return np.reshape(state, original_shape)


def assert_placement(circuit: Circuit, connectivity: nx.Graph):
    """Check if the layout of the circuit is consistent with the circuit and connectivity graph.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit model to check.
        connectivity (:class:`networkx.Graph`, optional): Chip connectivity.
    """
    if connectivity is None:
        raise_error(
            ValueError,
            "Connectivity graph is missing.",
        )

    if circuit.nqubits != len(circuit.wire_names) or circuit.nqubits != len(
        connectivity.nodes
    ):
        raise_error(
            PlacementError,
            f"Number of qubits in the circuit ({circuit.nqubits}) "
            + f"does not match the number of qubits in the layout ({len(circuit.wire_names)}) "
            + f"or the connectivity graph ({len(connectivity.nodes)}).",
        )
    if set(circuit.wire_names) != set(connectivity.nodes):
        raise_error(
            PlacementError,
            "Some physical qubits in the layout may be missing or duplicated.",
        )


def assert_connectivity(connectivity: nx.Graph, circuit: Circuit):
    """Assert if a circuit can be executed on Hardware.

    No gates acting on more than two qubits.
    All two-qubit operations can be performed on hardware.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to check.
        connectivity (:class:`networkx.Graph`): chip connectivity.
    """
    layout = circuit.wire_names
    for gate in circuit.queue:
        if len(gate.qubits) > 2 and not isinstance(gate, gates.M):
            raise_error(ConnectivityError, f"{gate.name} acts on more than two qubits.")
        if len(gate.qubits) == 2:
            physical_qubits = (layout[gate.qubits[0]], layout[gate.qubits[1]])
            if physical_qubits not in connectivity.edges:
                raise_error(
                    ConnectivityError,
                    f"The circuit does not respect the connectivity. {gate.name} acts on {physical_qubits} but only the following qubits are directly connected: {connectivity.edges}.",
                )


def assert_decomposition(
    circuit: Circuit,
    native_gates: NativeGates,
):
    """Checks if a circuit has been correctly decomposed into native gates.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit model to check.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`):
            native gates in the transpiled circuit.
    """
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue
        if len(gate.qubits) <= 2:
            try:
                native_type_gate = NativeGates.from_gate(gate)
                if not native_type_gate & native_gates:
                    raise_error(
                        DecompositionError,
                        f"{gate.name} is not a native gate.",
                    )
            except ValueError:
                raise_error(
                    DecompositionError,
                    f"{gate.name} is not a native gate.",
                )
        else:
            raise_error(
                DecompositionError, f"{gate.name} acts on more than two qubits."
            )
