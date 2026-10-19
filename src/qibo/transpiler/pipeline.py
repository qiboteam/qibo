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
from qibo.transpiler.optimizer import GateDirection, Preprocessing
from qibo.transpiler.placer import (
    Random,
    ReverseTraversal,
    StarConnectivityPlacer,
    Subgraph,
)
from qibo.transpiler.router import Sabre, ShortestPaths, StarConnectivityRouter
from qibo.transpiler.unroller import NativeGates, Unroller


class Passes:
    """Define a transpiler pipeline consisting of smaller transpiler steps that are applied sequentially.

    Args:
        passes (list, optional): List of transpiler passes to be applied sequentially.
            If ``None``, no pass is applied. See
            :func:`qibo.transpiler.pipeline.generate_preset_passes` for a ready-to-use list.
            Defaults to ``None``.
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity. It can be
            a directed graph, in which case an edge ``(a, b)`` means that a two-qubit gate
            is supported with its first qubit on ``a`` and its second on ``b``, see
            :class:`qibo.transpiler.optimizer.GateDirection`. It is required if
            ``on_qubits`` is given. Defaults to ``None``.
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

        If there is no connectivity, only the native gates are checked, because there is
        nothing to place and route.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be checked.

        Returns:
            (bool): ``True`` if the circuit respects the hardware connectivity and native gates, ``False`` otherwise.
        """
        try:
            if self.connectivity is not None:
                assert_placement(circuit=circuit, connectivity=self.connectivity)
                assert_connectivity(circuit=circuit, connectivity=self.connectivity)
            assert_decomposition(circuit=circuit, native_gates=self.native_gates)
            return True
        except (ConnectivityError, DecompositionError, PlacementError):
            return False


def generate_preset_passes(
    connectivity: nx.Graph | None = None,
    native_gates: NativeGates | None = None,
    initial_layout: list[int | str] | None = None,
    seed: int | None = None,
    on_qubits: list[int | str] | None = None,
    qubits_initially_zero: bool = False,
    layout_method: str | Placer | None = None,
    routing_method: str | Router | None = None,
    backend: Backend | None = None,
) -> Passes:
    """Generate a ready-to-use transpiler pipeline that does not optimize the circuit.

    The pipeline only makes the circuit executable on the hardware, with the following
    passes:

    1. :class:`qibo.transpiler.unroller.Unroller` decomposes the gates acting on three or
       more qubits, which the router does not support. The qubits of the circuit that
       a multi-controlled gate does not act on are used as auxiliary qubits, see
       ``qubits_initially_zero``.
    2. :class:`qibo.transpiler.optimizer.Preprocessing` places the qubits on
       ``initial_layout``, or on the first nodes of the connectivity if there is none,
       and pads the circuit with the unused qubits of the hardware.
    3. The placer selected by ``layout_method``, if it is not the trivial one.
    4. The router selected by ``routing_method``, which adds the SWAP gates that make all
       the gates executable. By default, it is :class:`qibo.transpiler.router.Sabre`
       with the best of five trials.
    5. :class:`qibo.transpiler.unroller.Unroller` decomposes the circuit to the native gates.

    If ``connectivity`` is directed, the pipeline ends with a
    :class:`qibo.transpiler.optimizer.GateDirection` pass and an unroller, so that the
    two-qubit gates act in the direction of the edges.

    Without ``connectivity`` there is nothing to place and route, so the pipeline only
    decomposes the circuit to the native gates.

    Args:
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity. It can be
            directed, see above.
            Defaults to ``None``.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native gates
            supported by the hardware. Defaults to
            :meth:`qibo.transpiler.unroller.NativeGates.default`.
        initial_layout (list, optional): Physical qubits on which the logical qubits of the
            circuit are placed, see :class:`qibo.transpiler.optimizer.Preprocessing`.
            It requires ``connectivity``, and it cannot be used with a non-trivial
            ``layout_method``. Defaults to ``None``.
        seed (int, optional): Seed of the random choices of the placer and the router, which
            makes the transpilation reproducible. Defaults to ``None``.
        on_qubits (list, optional): List of qubits to be used in the transpiler, see
            :class:`qibo.transpiler.pipeline.Passes`. It requires ``connectivity``.
            Defaults to ``None``.
        qubits_initially_zero (bool, optional): If ``True``, the circuit is assumed to start
            in the state :math:`\\ket{0}^{\\otimes n}`. The qubits on which no previous gate
            acts are then used as clean auxiliary qubits to decompose multi-controlled gates,
            with the construction that minimizes the number of Toffoli gates, see
            :class:`qibo.transpiler.unroller.Unroller`. This makes the decomposition
            shallower and, for many controls, also shorter. If there are no clean qubits
            for a gate, its dirty auxiliary qubits are used with the same construction,
            which has more CNOT gates than the default one. Without ``qubits_initially_zero``
            the other qubits are used as dirty auxiliary qubits, which can be in any state.
            It must be ``False`` if the circuit is executed with another initial state.
            Defaults to ``False``.
        layout_method (str or :class:`qibo.transpiler.abstract.Placer`, optional): Placer
            that finds the initial layout. It can be ``"trivial"``, which keeps the
            ``wire_names`` of the circuit if they are nodes of the connectivity and uses
            the first nodes otherwise, ``"random"``
            (:class:`qibo.transpiler.placer.Random`), ``"subgraph"``
            (:class:`qibo.transpiler.placer.Subgraph`), ``"reverse_traversal"``
            (:class:`qibo.transpiler.placer.ReverseTraversal` with
            :class:`qibo.transpiler.router.Sabre`), ``"star"``
            (:class:`qibo.transpiler.placer.StarConnectivityPlacer`) or a placer. It
            requires ``connectivity``. Defaults to ``None``, which is ``"trivial"``.
        routing_method (str or :class:`qibo.transpiler.abstract.Router`, optional): Router
            that makes the circuit executable. It can be ``"sabre"``
            (:class:`qibo.transpiler.router.Sabre` with five trials), ``"shortest_paths"``
            (:class:`qibo.transpiler.router.ShortestPaths`), ``"star"``
            (:class:`qibo.transpiler.router.StarConnectivityRouter`) or a router. It
            requires ``connectivity``. Defaults to ``None``, which is ``"sabre"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used by the
            unrollers to compute the gate matrices, unless another one is given when the
            pipeline is called. If ``None``, it uses the global backend. Defaults to ``None``.

    Returns:
        :class:`qibo.transpiler.pipeline.Passes`: Transpiler pipeline.

    Raises:
        ConnectivityError: If ``initial_layout``, ``layout_method`` or ``routing_method``
            is given without ``connectivity``.
        PlacementError: If ``initial_layout`` is given with a non-trivial ``layout_method``.
        ValueError: If ``layout_method`` or ``routing_method`` is not a valid method.
    """
    if connectivity is None and not (
        initial_layout is None and layout_method is None and routing_method is None
    ):
        raise_error(
            ConnectivityError,
            "``connectivity`` must be provided when ``initial_layout``, "
            + "``layout_method`` or ``routing_method`` is given.",
        )

    if native_gates is None:
        native_gates = NativeGates.default()

    # multi-controlled gates use the other qubits as auxiliary qubits, which is exact
    # for any state if they are dirty, and it is cheaper if they are clean
    unroller_kwargs = {
        "backend": backend,
        "use_dirty_ancillas": True,
        "qubits_initially_zero": qubits_initially_zero,
        "minimize_toffolis": qubits_initially_zero,
    }

    if connectivity is None:
        return Passes(
            passes=[Unroller(native_gates, **unroller_kwargs)],
            native_gates=native_gates,
            on_qubits=on_qubits,
        )

    placers = {
        "random": lambda: Random(seed=seed),
        "reverse_traversal": lambda: ReverseTraversal(
            routing_algorithm=Sabre(seed=seed)
        ),
        "star": StarConnectivityPlacer,
        "subgraph": Subgraph,
    }
    routers = {
        "sabre": lambda: Sabre(seed=seed, trials=5),
        "shortest_paths": lambda: ShortestPaths(seed=seed),
        "star": StarConnectivityRouter,
    }

    if layout_method is None or layout_method == "trivial":
        placer = None
    elif isinstance(layout_method, Placer):
        placer = layout_method
    elif isinstance(layout_method, str) and layout_method in placers:
        placer = placers[layout_method]()
    else:
        raise_error(
            ValueError,
            f"Unknown layout method {layout_method!r}. It must be a placer or one of "
            + f"{['trivial', *placers]}.",
        )
    if placer is not None and initial_layout is not None:
        raise_error(
            PlacementError,
            "``initial_layout`` cannot be used with a non-trivial ``layout_method``.",
        )

    if routing_method is None:
        routing_method = "sabre"
    if isinstance(routing_method, Router):
        router = routing_method
    elif isinstance(routing_method, str) and routing_method in routers:
        router = routers[routing_method]()
    else:
        raise_error(
            ValueError,
            f"Unknown routing method {routing_method!r}. It must be a router or one of "
            + f"{list(routers)}.",
        )

    passes = [
        Unroller(native_gates, min_qubits=3, **unroller_kwargs),
        Preprocessing(initial_layout=initial_layout),
        *([] if placer is None else [placer]),
        router,
        Unroller(native_gates, backend=backend),
    ]
    if connectivity.is_directed():
        # the unroller does not choose the direction of the two-qubit gates, and the
        # gates that reverse them are not necessarily native
        passes += [GateDirection(), Unroller(native_gates, backend=backend)]

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

    new_connectivity = nx.DiGraph() if connectivity.is_directed() else nx.Graph()
    new_connectivity.add_nodes_from(qubits)
    new_edges = [
        edge for edge in connectivity.edges if edge[0] in qubits and edge[1] in qubits
    ]
    new_connectivity.add_edges_from(new_edges)

    if not nx.is_connected(new_connectivity.to_undirected()):
        raise_error(ConnectivityError, "New connectivity graph is not connected.")

    return new_connectivity


def transpile(
    circuits: Circuit | list[Circuit] | tuple[Circuit],
    connectivity: nx.Graph | None = None,
    native_gates: NativeGates | None = None,
    initial_layout: list[int | str] | None = None,
    seed: int | None = None,
    on_qubits: list[int | str] | None = None,
    qubits_initially_zero: bool = False,
    layout_method: str | Placer | None = None,
    routing_method: str | Router | None = None,
    backend: Backend | None = None,
) -> tuple[Circuit, dict | None] | list[tuple[Circuit, dict | None]]:
    """Transpile one or more circuits in a single call, without optimizing them.

    It builds the pipeline of :func:`qibo.transpiler.pipeline.generate_preset_passes` once
    and applies it to the circuits, which makes them executable on the hardware.
    All the arguments except ``circuits`` are the ones of
    :func:`qibo.transpiler.pipeline.generate_preset_passes`, and they are shared by all the
    circuits.

    Example:
        A random unitary on three qubits and a CNOT gate between qubits that are not neighbors
        are transpiled to a line of four qubits, where :class:`qibo.gates.CNOT` and
        :class:`qibo.gates.U3` are the native gates. The unitary is decomposed, the SWAP gates
        that the routing adds are decomposed too, and the transpiled circuit is drawn. If
        ``circuits`` is a list, the result is a list with a transpiled circuit and a final
        layout for each circuit.

        .. testcode::

            import networkx as nx

            from qibo import Circuit, gates
            from qibo.quantum_info import random_unitary
            from qibo.transpiler import NativeGates, transpile
            from qibo.transpiler.asserts import assert_transpiling

            circuit = Circuit(4)
            circuit.add(gates.H(0))
            circuit.add(gates.Unitary(random_unitary(8, seed=42), 0, 1, 2))
            circuit.add(gates.CNOT(0, 3))
            circuit.add(gates.M(*range(4)))

            connectivity = nx.path_graph(4)
            natives = NativeGates.from_gatelist([gates.U3, gates.CNOT, gates.M])
            transpiled_circuit, final_layout = transpile(circuit, connectivity, natives, seed=42)

            # Raises an error if the circuits are not equivalent, or if the transpiled circuit
            # does not respect the connectivity and the native gates
            assert_transpiling(
                original_circuit=circuit,
                transpiled_circuit=transpiled_circuit,
                connectivity=connectivity,
                final_layout=final_layout,
                native_gates=natives,
            )

            print(final_layout)
            transpiled_circuit.draw(line_wrap=1000, parallel=True)

        .. testoutput::

            {0: 2, 1: 0, 2: 1, 3: 3}
            0: ──────────────────────────────────o──X─o────────────o─────────o──U3───────o─U3───────o─U3───────o─U3───────────────o─────────o──U3───────o─U3───────o─U3───────o─U3───────────────o─────────o──U3─────o─U3───────o─U3───────o─U3────M─
            1: ─U3────o─U3───────o─U3───────o─U3─X──o─X─U3─RZ─X─RZ─X─RZ─X─RZ─X──U3─o─X─o─X─U3─U3─U3─X─U3─U3─U3─X─U3─U3─o────X─o─X─X─U3─X─U3─X──RZ─o─X─o─X─U3─U3─U3─X─U3─U3─U3─X─U3─U3─o────X─o─X─X─RZ─X─RZ─X──o──X─o─X─U3─U3─U3─X─U3─U3─U3─X─U3─U3─M─
            2: ─U3─U3─X─U3─U3─U3─X─U3─U3─U3─X─U3─U3───────────o─────────o─U3─U3────X─o─X───────────────────────────────X─U3─o─X─o──────o─U3─U3────X─o─X───────────────────────────────X─RZ─o─X─o──────o─U3─U3─X──o─X─o─────────────────────────────M─
            3: ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────X─────────────────────────────M─

    Args:
        circuits (:class:`qibo.models.circuit.Circuit` or list or tuple): Circuit, or list of
            circuits, to be transpiled.
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            Defaults to ``None``.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native gates
            supported by the hardware. Defaults to
            :meth:`qibo.transpiler.unroller.NativeGates.default`.
        initial_layout (list, optional): Physical qubits on which the logical qubits of the
            circuits are placed. Defaults to ``None``.
        seed (int, optional): Seed of the random choices of the placer and the router.
            Defaults to ``None``.
        on_qubits (list, optional): List of qubits to be used in the transpiler.
            Defaults to ``None``.
        qubits_initially_zero (bool, optional): If ``True``, the circuits are assumed to
            start in the state :math:`\\ket{0}^{\\otimes n}`. Defaults to ``False``.
        layout_method (str or :class:`qibo.transpiler.abstract.Placer`, optional): Placer
            that finds the initial layout. Defaults to ``None``.
        routing_method (str or :class:`qibo.transpiler.abstract.Router`, optional): Router
            that makes the circuits executable. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used by the
            unrollers to compute the gate matrices. Defaults to ``None``.

    Returns:
        (:class:`qibo.models.circuit.Circuit`, dict) or list: Transpiled circuit and final
        {logical: physical} qubit mapping, see :class:`qibo.transpiler.pipeline.Passes`.
        If ``circuits`` is a list, the list of these pairs, in the same order.

    """
    passes = generate_preset_passes(
        connectivity=connectivity,
        native_gates=native_gates,
        initial_layout=initial_layout,
        seed=seed,
        on_qubits=on_qubits,
        qubits_initially_zero=qubits_initially_zero,
        layout_method=layout_method,
        routing_method=routing_method,
        backend=backend,
    )
    if isinstance(circuits, Circuit):
        return passes(circuits)

    return [passes(circuit) for circuit in circuits]
