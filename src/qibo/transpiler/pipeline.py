from typing import Optional

import networkx as nx
import numpy as np

from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.quantum_info.random_ensembles import random_statevector
from qibo.transpiler._exceptions import TranspilerPipelineError
from qibo.transpiler.abstract import Optimizer, Placer, Router
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.placer import StarConnectivityPlacer, Trivial, assert_placement
from qibo.transpiler.router import (
    ConnectivityError,
    StarConnectivityRouter,
    assert_connectivity,
)
from qibo.transpiler.unroller import (
    DecompositionError,
    NativeGates,
    Unroller,
    assert_decomposition,
)


def assert_circuit_equivalence(
    original_circuit: Circuit,
    transpiled_circuit: Circuit,
    final_map: dict,
    initial_map: Optional[dict] = None,
    test_states: Optional[list] = None,
    ntests: int = 3,
):
    """Checks that the transpiled circuit agrees with the original using simulation.

    Args:
        original_circuit (:class:`qibo.models.circuit.Circuit`): Original circuit.
        transpiled_circuit (:class:`qibo.models.circuit.Circuit`): Transpiled circuit.
        final_map (dict): logical-physical qubit mapping after routing.
        initial_map (dict, optional): logical_physical qubit mapping before routing.
            If ``None``, trivial initial map is used. Defauts to ``None``.
        test_states (list, optional): states on which the test is performed.
            If ``None``, ``ntests`` random states will be tested. Defauts to ``None``.
        ntests (int, optional): number of random states tested. Defauts to :math:`3`.
    """
    backend = NumpyBackend()
    ordering = np.argsort(np.array(list(final_map.values())))
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
    if initial_map is not None:
        reordered_test_states = []
        initial_map = np.array(list(initial_map.values()))
        reordered_test_states = [
            _transpose_qubits(initial_state, initial_map)
            for initial_state in test_states
        ]
    else:
        reordered_test_states = test_states

    for i in range(len(test_states)):
        target_state = backend.execute_circuit(
            original_circuit, initial_state=test_states[i]
        ).state()
        final_state = backend.execute_circuit(
            transpiled_circuit, initial_state=reordered_test_states[i]
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


def assert_transpiling(
    original_circuit: Circuit,
    transpiled_circuit: Circuit,
    connectivity: nx.Graph,
    initial_layout: dict,
    final_layout: dict,
    native_gates: NativeGates = NativeGates.default(),
    check_circuit_equivalence=True,
):
    """Check that all transpiler passes have been executed correctly.

    Args:
        original_circuit (qibo.models.Circuit): circuit before transpiling.
        transpiled_circuit (qibo.models.Circuit): circuit after transpiling.
        connectivity (networkx.Graph): chip qubits connectivity.
        initial_layout (dict): initial physical-logical qubit mapping.
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
    assert_placement(
        circuit=original_circuit, layout=initial_layout, connectivity=connectivity
    )
    assert_placement(
        circuit=transpiled_circuit, layout=final_layout, connectivity=connectivity
    )
    if check_circuit_equivalence:
        assert_circuit_equivalence(
            original_circuit=original_circuit,
            transpiled_circuit=transpiled_circuit,
            initial_map=initial_layout,
            final_map=final_layout,
        )


def restrict_connectivity_qubits(connectivity: nx.Graph, qubits: list):
    """Restrict the connectivity to selected qubits.

    Args:
        connectivity (:class:`networkx.Graph`): chip connectivity.
        qubits (list): list of physical qubits to be used.

    Returns:
        (:class:`networkx.Graph`): restricted connectivity.
    """
    if not set(qubits).issubset(set(connectivity.nodes)):
        raise_error(
            ConnectivityError, "Some qubits are not in the original connectivity."
        )

    new_connectivity = nx.Graph()
    new_connectivity.add_nodes_from(qubits)
    new_edges = [
        edge for edge in connectivity.edges if edge[0] in qubits and edge[1] in qubits
    ]
    new_connectivity.add_edges_from(new_edges)

    if not nx.is_connected(new_connectivity):
        raise_error(ConnectivityError, "New connectivity graph is not connected.")

    return new_connectivity


class Passes:
    """Define a transpiler pipeline consisting of smaller transpiler steps that are applied sequentially:

    Args:
        passes (list, optional): list of passes to be applied sequentially.
            If ``None``, default transpiler will be used.
            Defaults to ``None``.
        connectivity (:class:`networkx.Graph`, optional): physical qubits connectivity.
            If ``None``, :class:`` is used.
            Defaults to ``None``.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): native gates.
            Defaults to :math:`qibo.transpiler.unroller.NativeGates.default`.
        on_qubits (list, optional): list of physical qubits to be used.
            If "None" all qubits are used. Defaults to ``None``.
        int_qubit_name (bool, optional): if `True` the `final_layout` keys are
            cast to integers.
    """

    def __init__(
        self,
        passes: list = None,
        connectivity: nx.Graph = None,
        native_gates: NativeGates = NativeGates.default(),
        on_qubits: list = None,
        int_qubit_names: bool = False,
    ):
        if on_qubits is not None:
            connectivity = restrict_connectivity_qubits(connectivity, on_qubits)
        self.connectivity = connectivity
        self.native_gates = native_gates
        self.passes = self.default() if passes is None else passes
        self.initial_layout = None
        self.int_qubit_names = int_qubit_names

    def default(self):
        """Return the default transpiler pipeline for the required hardware connectivity."""
        if not isinstance(self.connectivity, nx.Graph):
            raise_error(
                TranspilerPipelineError,
                "Define the hardware chip connectivity to use default transpiler",
            )
        default_passes = []
        # preprocessing
        default_passes.append(Preprocessing(connectivity=self.connectivity))
        # default placer pass
        default_passes.append(StarConnectivityPlacer())
        # default router pass
        default_passes.append(StarConnectivityRouter())
        # default unroller pass
        default_passes.append(Unroller(native_gates=self.native_gates))

        return default_passes

    def __call__(self, circuit):
        """
        This function returns the compiled circuits and the dictionary mapping
        physical (keys) to logical (values) qubit. If `int_qubit_name` is `True`
        each key `i` correspond to the `i-th` qubit in the graph.
        """
        final_layout = self.initial_layout = None
        for transpiler_pass in self.passes:
            if isinstance(transpiler_pass, Optimizer):
                transpiler_pass.connectivity = self.connectivity
                circuit = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Placer):
                transpiler_pass.connectivity = self.connectivity
                if self.initial_layout is None:
                    self.initial_layout = transpiler_pass(circuit)
                    final_layout = (
                        self.initial_layout
                    )  # This way the final layout will be the same as the initial layout if no router is used
                else:
                    raise_error(
                        TranspilerPipelineError,
                        "You are defining more than one placer pass.",
                    )
            elif isinstance(transpiler_pass, Router):
                transpiler_pass.connectivity = self.connectivity
                if self.initial_layout is not None:
                    circuit, final_layout = transpiler_pass(
                        circuit, self.initial_layout
                    )
                else:
                    raise_error(
                        TranspilerPipelineError, "Use a placement pass before routing."
                    )
            elif isinstance(transpiler_pass, Unroller):
                circuit = transpiler_pass(circuit)
            else:
                raise_error(
                    TranspilerPipelineError,
                    f"Unrecognised transpiler pass: {transpiler_pass}",
                )
        if self.int_qubit_names and final_layout is not None:
            final_layout = {int(key[1:]): value for key, value in final_layout.items()}
        return circuit, final_layout

    def is_satisfied(self, circuit: Circuit):
        """Returns ``True`` if the circuit respects the hardware connectivity and native gates, ``False`` otherwise.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be checked.

        Returns:
            (bool): satisfiability condition.
        """
        try:
            assert_connectivity(circuit=circuit, connectivity=self.connectivity)
            assert_decomposition(circuit=circuit, native_gates=self.native_gates)
            return True
        except ConnectivityError:
            return False
        except DecompositionError:
            return False

    def get_initial_layout(self):
        """Return initial qubit layout"""
        return self.initial_layout
