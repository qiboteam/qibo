from inspect import signature

import networkx as nx

from qibo import Circuit
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.transpiler._exceptions import (
    ConnectivityError,
    DecompositionError,
    PlacementError,
    TranspilerPipelineError,
)
from qibo.transpiler.abstract import Optimizer, Placer, Router
from qibo.transpiler.asserts import (
    assert_connectivity,
    assert_decomposition,
    assert_placement,
)
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.router import Sabre
from qibo.transpiler.unroller import NativeGates, Unroller


class Passes:
    """Define a transpiler pipeline consisting of smaller transpiler steps that are applied sequentially.

    Args:
        passes (list, optional): List of transpiler passes to be applied sequentially.
            If ``None``, no pass is applied. See
            :func:`qibo.transpiler.pipeline.generate_preset_passes` for a ready-to-use list.
            Defaults to ``None``.
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            It is required if ``on_qubits`` is given. Defaults to ``None``.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native gates supported by the hardware.
            Defaults to :meth:`qibo.transpiler.unroller.NativeGates.default`.
        on_qubits (list, optional): List of qubits to be used in the transpiler.
            If ``None``, all qubits in the connectivity will be used. Defaults to ``None``.
    """

    def __init__(
        self,
        passes: list | None = None,
        connectivity: nx.Graph | None = None,
        native_gates: NativeGates | None = None,
        on_qubits: list | None = None,
    ):
        if on_qubits is not None:
            if connectivity is None:
                raise_error(
                    ConnectivityError,
                    "``connectivity`` must be provided when ``on_qubits`` is given.",
                )
            connectivity = restrict_connectivity_qubits(connectivity, on_qubits)
        if native_gates is None:
            native_gates = NativeGates.default()
        self.connectivity = connectivity
        self.native_gates = native_gates
        self.passes = [] if passes is None else passes

    def __call__(
        self, circuit: Circuit, backend: Backend | None = None
    ) -> tuple[Circuit, dict | None]:
        """Apply the transpiler pipeline to the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
            backend (:class:`qibo.backends.abstract.Backend`, optional): Backend passed
                to the transpiler passes that take one. If ``None``, defaults to the
                global backend, except for the unrollers, which keep their own backend.
                Defaults to ``None``.

        Returns:
            (:class:`qibo.models.circuit.Circuit`, dict): Transpiled circuit and final
            {logical: physical} qubit mapping. The mapping is ``None`` if the pipeline
            has no router.
        """
        # an unroller uses its own backend unless one is explicitly given
        unroller_backend = backend
        backend = _check_backend(backend)

        final_layout = None
        for transpiler_pass in self.passes:
            if isinstance(transpiler_pass, Optimizer):
                transpiler_pass.connectivity = self.connectivity
                if "backend" in signature(transpiler_pass.__call__).parameters:
                    circuit = transpiler_pass(circuit, backend=backend)
                else:
                    circuit = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Placer):
                transpiler_pass.connectivity = self.connectivity
                final_layout = transpiler_pass(circuit, backend=backend)
            elif isinstance(transpiler_pass, Router):
                transpiler_pass.connectivity = self.connectivity
                circuit, final_layout = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Unroller):
                circuit = transpiler_pass(circuit, backend=unroller_backend)
            elif transpiler_pass.__class__.__name__ == "QiskitPasses":
                circuit = transpiler_pass(circuit)
            else:
                raise_error(
                    TranspilerPipelineError,
                    f"Unrecognised transpiler pass: {transpiler_pass}",
                )
        return circuit, final_layout

    def is_satisfied(self, circuit: Circuit) -> bool:
        """Check if the circuit respects the hardware connectivity and native gates.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be checked.

        Returns:
            (bool): ``True`` if the circuit respects the hardware connectivity and native gates, ``False`` otherwise.
        """
        try:
            assert_placement(circuit=circuit, connectivity=self.connectivity)
            assert_connectivity(circuit=circuit, connectivity=self.connectivity)
            assert_decomposition(circuit=circuit, native_gates=self.native_gates)
            return True
        except (ConnectivityError, DecompositionError, PlacementError, ValueError):
            return False


def generate_preset_passes(
    connectivity: nx.Graph | None = None,
    native_gates: NativeGates | None = None,
    initial_layout: list[int | str] | None = None,
    seed: int | None = None,
    on_qubits: list[int | str] | None = None,
    backend: Backend | None = None,
) -> Passes:
    """Generate a ready-to-use transpiler pipeline that does not optimize the circuit.

    The pipeline only makes the circuit executable on the hardware, with the following
    passes:

    1. :class:`qibo.transpiler.unroller.Unroller` decomposes the gates acting on three or
       more qubits, which the router does not support.
    2. :class:`qibo.transpiler.optimizer.Preprocessing` places the qubits on
       ``initial_layout``, or on the first nodes of the connectivity if there is none,
       and pads the circuit with the unused qubits of the hardware.
    3. :class:`qibo.transpiler.router.Sabre` adds the SWAP gates that make all the gates
       executable, keeping the best of five trials.
    4. :class:`qibo.transpiler.unroller.Unroller` decomposes the circuit to the native gates.

    Without ``connectivity`` there is nothing to place and route, so the pipeline only
    decomposes the circuit to the native gates.

    Args:
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            Defaults to ``None``.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native gates
            supported by the hardware. Defaults to
            :meth:`qibo.transpiler.unroller.NativeGates.default`.
        initial_layout (list, optional): Physical qubits on which the logical qubits of the
            circuit are placed, see :class:`qibo.transpiler.optimizer.Preprocessing`.
            It requires ``connectivity``. Defaults to ``None``.
        seed (int, optional): Seed of the random choices of the router, which makes the
            transpilation reproducible. Defaults to ``None``.
        on_qubits (list, optional): List of qubits to be used in the transpiler, see
            :class:`qibo.transpiler.pipeline.Passes`. It requires ``connectivity``.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used by the
            unrollers to compute the gate matrices, unless another one is given when the
            pipeline is called. If ``None``, it uses the global backend. Defaults to ``None``.

    Returns:
        :class:`qibo.transpiler.pipeline.Passes`: Transpiler pipeline.
    """
    if connectivity is None and initial_layout is not None:
        raise_error(
            ConnectivityError,
            "``connectivity`` must be provided when ``initial_layout`` is given.",
        )

    if native_gates is None:
        native_gates = NativeGates.default()

    if connectivity is None:
        passes = [Unroller(native_gates, backend=backend)]
    else:
        passes = [
            Unroller(native_gates, backend=backend, min_qubits=3),
            Preprocessing(initial_layout=initial_layout),
            Sabre(seed=seed, trials=5),
            Unroller(native_gates, backend=backend),
        ]

    return Passes(
        passes=passes,
        connectivity=connectivity,
        native_gates=native_gates,
        on_qubits=on_qubits,
    )


def restrict_connectivity_qubits(
    connectivity: nx.Graph, qubits: list[int | str]
) -> nx.Graph:
    """Restrict the connectivity to selected qubits.

    Args:
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
        qubits (list): List of qubits to select.

    Returns:
        (:class:`networkx.Graph`): New restricted connectivity.
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
