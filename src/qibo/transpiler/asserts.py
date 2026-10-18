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
    native_gates: NativeGates | None = None,
    check_circuit_equivalence: bool = True,
) -> None:
    """Check that all transpiler passes have been executed correctly.

    Args:
        original_circuit (:class:`qibo.models.circuit.Circuit`): Circuit before transpiling.
        transpiled_circuit (:class:`qibo.models.circuit.Circuit`): Circuit after transpiling.
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
        final_layout (dict): Final {logical: physical} qubit mapping.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native gates supported by the hardware.
            Defaults to :meth:`qibo.transpiler.unroller.NativeGates.default`.
        check_circuit_equivalence (bool, optional): Check if the transpiled circuit is equivalent to the original one.
            Defaults to ``True``.
    """
    if native_gates is None:
        native_gates = NativeGates.default()
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
            final_layout=final_layout,
        )


def assert_circuit_equivalence(
    original_circuit: Circuit,
    transpiled_circuit: Circuit,
    final_layout: dict,
    test_states: list | None = None,
    ntests: int = 3,
) -> None:
    """Checks that the transpiled circuit is equivalent to the original one.

    Args:
        original_circuit (:class:`qibo.models.circuit.Circuit`): Circuit before transpiling.
        transpiled_circuit (:class:`qibo.models.circuit.Circuit`): Circuit after transpiling.
        final_layout (dict): Final {logical: physical} qubit mapping.
        test_states (list, optional): List of states to test the equivalence.
            If ``None``, ``ntests`` random states will be tested. Defaults to ``None``.
        ntests (int, optional): Number of random states to test the equivalence. Defaults to :math:`3`.
    """
    backend = NumpyBackend()
    if transpiled_circuit.nqubits != original_circuit.nqubits:
        raise_error(
            ValueError,
            f"Transpiled circuit ({transpiled_circuit.nqubits}) and original circuit "
            + f"({original_circuit.nqubits}) do not have the same number of qubits.",
        )

    if test_states is None:
        test_states = [
            random_statevector(dims=2**original_circuit.nqubits, backend=backend)
            for _ in range(ntests)
        ]

    ordering = list(final_layout.values())

    for state in test_states:
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


def _transpose_qubits(state: np.ndarray, qubits_ordering: list[int]) -> np.ndarray:
    """Reorders qubits of a given state vector.

    Args:
        state (np.ndarray): State vector to reorder.
        qubits_ordering (list[int]): Final qubit ordering.

    Returns:
        np.ndarray: State vector with the qubits reordered.
    """
    original_shape = state.shape
    state = np.reshape(state, len(qubits_ordering) * (2,))
    state = np.transpose(state, qubits_ordering)
    return np.reshape(state, original_shape)


def assert_placement(circuit: Circuit, connectivity: nx.Graph | None) -> None:
    """Check if the layout of the circuit is consistent with the circuit and connectivity graph.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit to check.
        connectivity (:class:`networkx.Graph`): Hardware connectivity. A
            ``ValueError`` is raised if it is ``None``.
    """
    if connectivity is None:
        raise_error(
            ValueError,
            "Connectivity graph is not provided.",
        )

    if circuit.nqubits != len(circuit.wire_names) or circuit.nqubits != len(
        connectivity.nodes
    ):
        raise_error(
            PlacementError,
            f"Number of qubits in the circuit ({circuit.nqubits}) "
            + f"does not match either the number of qubits in the layout ({len(circuit.wire_names)}) "
            + f"or the connectivity graph ({len(connectivity.nodes)}).",
        )
    if set(circuit.wire_names) != set(connectivity.nodes):
        raise_error(
            PlacementError,
            "Some physical qubits in the layout may be missing or duplicated.",
        )


def assert_connectivity(connectivity: nx.Graph, circuit: Circuit) -> None:
    """Assert if a circuit can be executed on Hardware.

    No gates acting on more than two qubits, except for measurements and barriers.
    All two-qubit operations can be performed on hardware. Measurements and barriers
    do not require the qubits to be connected.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit to check.
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
    """
    layout = circuit.wire_names
    for gate in circuit.queue:
        if isinstance(gate, (gates.Barrier, gates.M)):
            continue
        if len(gate.qubits) > 2:
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
) -> None:
    """Checks if a circuit has been correctly decomposed into native gates.

    The identity gate, the measurements, the resets and the directives are not checked,
    and :class:`qibo.gates.gates.Z` and :class:`qibo.gates.gates.RZ` are also accepted
    if :class:`qibo.gates.gates.U3` or :class:`qibo.gates.gates.GPI2` are native,
    because they are virtual gates that are assumed to be available together with them,
    as in :class:`qibo.transpiler.unroller.Unroller`.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit to check.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates supported by the hardware.
    """
    if native_gates & (NativeGates.U3 | NativeGates.GPI2):
        native_gates |= NativeGates.Z | NativeGates.RZ
    for gate in circuit.queue:
        if isinstance(
            gate, (gates.Align, gates.Barrier, gates.I, gates.M, gates.ResetChannel)
        ):
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
