import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from inspect import signature
from itertools import repeat

import networkx as nx

from qibo import Circuit, gates
from qibo.backends import Backend, _check_backend
from qibo.config import log, raise_error
from qibo.gates.abstract import Gate
from qibo.transpiler._exceptions import (
    ConnectivityError,
    DecompositionError,
    PlacementError,
    TranspilerPipelineError,
)
from qibo.transpiler.abstract import Optimizer, Placer, PostLayout, Router
from qibo.transpiler.asserts import (
    _find_unconnected_gate,
    assert_connectivity,
    assert_decomposition,
    assert_placement,
)
from qibo.transpiler.hardware import HardwareModel
from qibo.transpiler.layout import TranspileLayout
from qibo.transpiler.optimizer import (
    BarrierBeforeFinalMeasurements,
    CommutativeCancellation,
    ConsolidateBlocks,
    FilterGates,
    FixedPoint,
    GateDirection,
    InverseCancellation,
    Optimize1qGatesDecomposition,
    Optimize1qGatesSimpleCommutation,
    OptimizeSwapBeforeMeasure,
    Preprocessing,
    RemoveDiagonalGatesBeforeMeasurement,
    RemoveFinalReset,
    RemoveIdentityEquivalent,
    Validation,
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
from qibo.transpiler.router import (
    NoRouting,
    Sabre,
    ShortestPaths,
    StarConnectivityRouter,
)
from qibo.transpiler.unroller import NativeGates, Unroller

# Optimization levels of :func:`qibo.transpiler.pipeline.generate_preset_passes`.
_OPTIMIZATION_LEVELS = (0, 1, 2)


class Passes:
    """Define a transpiler pipeline consisting of smaller transpiler steps that
    are applied sequentially.

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
        callback (Callable, optional): Function called after each pass that runs, with the
            keyword arguments ``transpiler_pass``, the pass that has run, ``circuit``, the
            circuit after the pass, ``time``, the seconds that the pass has taken, and
            ``count``, the position of the pass in ``passes``. A router is skipped, and the
            callback is not called for it, if the circuit already respects the
            connectivity, since no qubit has to be moved. Defaults to ``None``.

    Example:
        The callback prints the passes of the pipeline as they run. The circuit already
        respects the connectivity, so the router is skipped and it is not printed.

        .. testcode::

            import networkx as nx

            from qibo import Circuit, gates
            from qibo.transpiler import Passes, Preprocessing, Sabre

            circuit = Circuit(2)
            circuit.add(gates.CNOT(0, 1))

            def callback(transpiler_pass, circuit, time, count):
                print(count, type(transpiler_pass).__name__, circuit.nqubits)

            pipeline = Passes(
                [Preprocessing(), Sabre()], connectivity=nx.path_graph(3), callback=callback
            )
            transpiled_circuit, final_layout = pipeline(circuit)

        .. testoutput::

            0 Preprocessing 3
    """

    def __init__(
        self,
        passes: list | None = None,
        connectivity: nx.Graph | None = None,
        native_gates: NativeGates | None = None,
        on_qubits: list | None = None,
        callback: Callable | None = None,
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
        self.callback = callback

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
            (:class:`qibo.models.circuit.Circuit`,
            :class:`qibo.transpiler.layout.TranspileLayout`): Transpiled circuit and final
            {logical: physical} qubit mapping, which also stores the layout before the
            routing. If the pipeline has no router, the qubits are not moved, so the
            mapping is the identity.
        """
        # an unroller uses its own backend unless one is explicitly given
        unroller_backend = backend
        backend = _check_backend(backend)

        final_layout = None
        nqubits_input = circuit.nqubits
        placer = None
        for count, transpiler_pass in enumerate(self.passes):
            start = time.perf_counter()
            skipped = False
            if isinstance(transpiler_pass, Optimizer):
                transpiler_pass.connectivity = self.connectivity
                if "backend" in signature(transpiler_pass.__call__).parameters:
                    circuit = transpiler_pass(circuit, backend=backend)
                else:
                    circuit = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Placer):
                transpiler_pass.connectivity = self.connectivity
                wire_names = circuit.wire_names.copy()
                final_layout = transpiler_pass(circuit, backend=backend)
                # a placer that leaves the layout, *e.g.* because it needs no SWAP gates,
                # does not take the place of the one that decided it
                if placer is None or circuit.wire_names != wire_names:
                    placer = transpiler_pass
            elif isinstance(transpiler_pass, Router):
                transpiler_pass.connectivity = self.connectivity
                assert_placement(circuit, self.connectivity)
                initial_layout = circuit.wire_names
                # no qubit has to be moved if the circuit respects the connectivity
                skipped = (
                    _find_unconnected_gate(transpiler_pass.connectivity, circuit)
                    is None
                )
                if skipped:
                    final_layout = {
                        name: index for index, name in enumerate(initial_layout)
                    }
                else:
                    circuit, final_layout = transpiler_pass(circuit)
                final_layout = TranspileLayout(
                    final_layout, initial_layout, nqubits_input
                )
            elif isinstance(transpiler_pass, PostLayout):
                transpiler_pass.connectivity = self.connectivity
                if final_layout is None:
                    raise_error(
                        TranspilerPipelineError,
                        "A post layout relabels a routed circuit, so it needs a router "
                        + "before it.",
                    )
                skipped = transpiler_pass.skip(placer)
                if not skipped:
                    final_layout = transpiler_pass(circuit, final_layout)
            elif isinstance(transpiler_pass, Unroller):
                circuit = transpiler_pass(circuit, backend=unroller_backend)
            elif transpiler_pass.__class__.__name__ == "QiskitPasses":
                circuit = transpiler_pass(circuit)
            else:
                raise_error(
                    TranspilerPipelineError,
                    f"Unrecognised transpiler pass: {transpiler_pass}",
                )

            if self.callback is not None and not skipped:
                self.callback(
                    transpiler_pass=transpiler_pass,
                    circuit=circuit,
                    time=time.perf_counter() - start,
                    count=count,
                )

        if final_layout is None:
            final_layout = TranspileLayout(
                {name: index for index, name in enumerate(circuit.wire_names)},
                circuit.wire_names,
                nqubits_input,
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
    initial_layout: list[int | str] | dict[int, int | str] | None = None,
    seed: int | None = None,
    on_qubits: list[int | str] | None = None,
    qubits_initially_zero: bool = False,
    layout_method: str | Placer | None = None,
    routing_method: str | Router | None = None,
    layout_trials: int | None = None,
    callback: Callable | None = None,
    approximation_degree: float = 1.0,
    threshold: float | None = None,
    weight: float | None = None,
    use_dirty_ancillas: bool = True,
    minimize_toffolis: bool | None = None,
    minimize_depth: bool = False,
    routing_trials: int | None = None,
    optimization_level: int = 0,
    sampling_only: bool = False,
    post_layout: PostLayout | None = None,
    hardware: HardwareModel | None = None,
    backend: Backend | None = None,
) -> Passes:
    """Generate a ready-to-use transpiler pipeline.

    With the default ``optimization_level=0``, the pipeline only makes the circuit
    executable on the hardware, with the following passes:

    1. :class:`qibo.transpiler.optimizer.Validation` raises an error at the start if the
       circuit cannot be transpiled, *e.g.* if it has a noise channel, more qubits than
       the hardware, or an invalid ``initial_layout``.
    2. :class:`qibo.transpiler.unroller.Unroller` decomposes the gates acting on three or
       more qubits, which the router does not support. The qubits of the circuit that
       a multi-controlled gate does not act on are used as auxiliary qubits, see
       ``qubits_initially_zero``.
    3. :class:`qibo.transpiler.optimizer.Preprocessing` places the qubits on
       ``initial_layout``, or on the first nodes of the connectivity if there is none,
       and pads the circuit with the unused qubits of the hardware.
    4. The placer selected by ``layout_method``, if it is not the trivial one.
    5. The router selected by ``routing_method``, which adds the SWAP gates that make all
       the gates executable. By default, it is :class:`qibo.transpiler.router.Sabre`
       with the best of ``routing_trials`` trials. If it is one of the routers of
       :mod:`qibo.transpiler.router`, a barrier on all the qubits is inserted before it by
       :class:`qibo.transpiler.optimizer.BarrierBeforeFinalMeasurements`, so that the final
       measurements stay after the SWAP gates, and the barriers on all the qubits are
       removed after it by :class:`qibo.transpiler.optimizer.FilterGates`, so the barriers
       of the circuit on all the qubits are removed too, and the ones on some of them stay.
    6. :class:`qibo.transpiler.unroller.Unroller` decomposes the circuit to the native gates,
       approximating some two-qubit gates with fewer entangling gates if
       ``approximation_degree`` is lower than :math:`1`.
    7. ``post_layout``, if it is given, which chooses other physical qubits for the
       routed circuit without changing its gates. It runs on the native circuit, where the
       entangling gates of each coupling are exact. If ``connectivity`` is directed, it runs
       before the pass that sets the direction of the gates, which depends on the qubits.
       Otherwise, it runs after the optimizers, so that it sees the final gates. At levels
       :math:`1` and :math:`2`, it is by default
       :class:`qibo.transpiler.placer.NoiseAwarePostLayout` if ``hardware`` is given and the
       layout is searched as described below. It is skipped if the layout was kept, or if
       it is the best one that the search found.

    If ``connectivity`` is directed, the pipeline ends with a
    :class:`qibo.transpiler.optimizer.GateDirection` pass and an unroller, so that the
    two-qubit gates act in the direction of the edges.

    The optimization levels :math:`1` and :math:`2` add optimizers to the pipeline. The ones
    that can write gates that are not native run before the last unroller. Level :math:`1`
    adds:

    * :class:`qibo.transpiler.optimizer.InverseCancellation` and
      :class:`qibo.transpiler.optimizer.CommutativeCancellation` before the layout.
    * :class:`qibo.transpiler.optimizer.ConsolidateBlocks` before the last unroller, which
      rewrites the blocks of two-qubit gates that have fewer :class:`qibo.gates.CZ` gates.
    * A loop of :class:`qibo.transpiler.optimizer.Optimize1qGatesDecomposition`, with the
      native gates as ``basis`` and ``hardware``, and
      :class:`qibo.transpiler.optimizer.InverseCancellation` at the end. The first one is left
      out if the native gates have no Euler basis.
    * A search for the layout, unless ``layout_method`` or ``initial_layout`` is given.
      First, :class:`qibo.transpiler.placer.NoiseAwarePlacer` looks for a layout where no
      SWAP gate is needed, and it chooses the one with the least noise if ``hardware`` is
      given. If there is none, the ``"reverse_traversal"`` placer with five trials looks
      for a layout that needs few SWAP gates. Neither of them moves the circuit if the layout
      that :class:`qibo.transpiler.optimizer.Preprocessing` set already needs no SWAP gates.

    Level :math:`2` adds, to the optimizers of level :math:`1`:

    * :class:`qibo.transpiler.optimizer.RemoveIdentityEquivalent` before the layout, which
      also removes the gates close to the identity if ``approximation_degree`` is lower
      than :math:`1`.
    * :class:`qibo.transpiler.optimizer.CommutativeCancellation` before the last unroller.
    * The layout search does not keep a layout without SWAP gates if ``hardware`` is given,
      so that the one with the least noise is chosen. The ``"reverse_traversal"`` placer and
      the ``"sabre"`` router have :math:`20` trials.
    * A second loop at the end, of
      :class:`qibo.transpiler.optimizer.Optimize1qGatesDecomposition`,
      :class:`qibo.transpiler.optimizer.Optimize1qGatesSimpleCommutation` and
      :class:`qibo.transpiler.optimizer.InverseCancellation` with commutation. It is also
      left out if the native gates have no Euler basis.

    The optimizers can change the global phase of the circuit, and the gates that they
    merge do not keep their trainable parameters.

    Without ``connectivity`` there is nothing to place and route, so the pipeline only
    validates the circuit and decomposes it to the native gates.

    Args:
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity. It can be
            directed, see above.
            Defaults to ``None``.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native gates
            supported by the hardware. Defaults to
            :meth:`qibo.transpiler.unroller.NativeGates.default`.
        initial_layout (list or dict, optional): Physical qubits on which the logical qubits
            of the circuit are placed, as a list or as a dictionary that may leave qubits
            out, see :class:`qibo.transpiler.optimizer.Preprocessing`.
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
            the first nodes otherwise, ``"dense"``
            (:class:`qibo.transpiler.placer.Dense`), ``"random"``
            (:class:`qibo.transpiler.placer.Random`), ``"subgraph"``
            (:class:`qibo.transpiler.placer.Subgraph`), ``"reverse_traversal"``
            (:class:`qibo.transpiler.placer.ReverseTraversal` with
            :class:`qibo.transpiler.router.Sabre`), ``"star"``
            (:class:`qibo.transpiler.placer.StarConnectivityPlacer`) or a placer. It
            requires ``connectivity``. Defaults to ``None``, which is ``"trivial"``, or
            the search of the layout described above if ``optimization_level`` is at
            least :math:`1`, ``initial_layout`` is ``None`` and ``routing_method`` is ``None``
            or ``"sabre"``.
        routing_method (str or :class:`qibo.transpiler.abstract.Router`, optional): Router
            that makes the circuit executable. It can be ``"sabre"``
            (:class:`qibo.transpiler.router.Sabre` with ``routing_trials`` trials),
            ``"shortest_paths"``
            (:class:`qibo.transpiler.router.ShortestPaths`), ``"star"``
            (:class:`qibo.transpiler.router.StarConnectivityRouter`), ``"none"``
            (:class:`qibo.transpiler.router.NoRouting`, which raises an error if the
            circuit needs routing) or a router. A router is skipped if the circuit
            already respects the connectivity. It requires ``connectivity``. Defaults
            to ``None``, which is ``"sabre"``.
        layout_trials (int, optional): Number of layout attempts of the
            ``"reverse_traversal"`` placer, see
            :class:`qibo.transpiler.placer.ReverseTraversal`. It is not used by the other
            placers. Defaults to ``None``, which is :math:`1`, :math:`5` or :math:`20` if
            ``optimization_level`` is :math:`0`, :math:`1` or :math:`2`.
        callback (Callable, optional): Function called after each pass that runs, see
            :class:`qibo.transpiler.pipeline.Passes`. Defaults to ``None``.
        approximation_degree (float, optional): Fidelity of each entangling gate, between
            :math:`0` and :math:`1`. If it is lower than :math:`1`, the unroller that
            decomposes the circuit to the native gates approximates the two-qubit gates that
            it decomposes from their matrix with fewer :class:`qibo.gates.CZ` gates, if it
            increases the expected fidelity, see
            :class:`qibo.transpiler.unroller.Unroller`. The multi-qubit gates are decomposed
            exactly. Defaults to :math:`1`, which is the exact decomposition.
        threshold (float, optional): Tolerance of the decomposition of the two-qubit gates
            that are decomposed from their matrix, like for ``approximation_degree``, see
            :class:`qibo.transpiler.unroller.Unroller`. Defaults to ``None``, which uses
            the default value of the decomposition.
        weight (float, optional): Weight of the imaginary part in the matrix that is
            diagonalized to decompose those gates, like for ``approximation_degree``, see
            :class:`qibo.transpiler.unroller.Unroller`. Defaults to ``None``, which uses
            the default value of the decomposition.
        use_dirty_ancillas (bool, optional): If ``True``, the qubits that a multi-controlled
            gate does not act on are used as dirty auxiliary qubits in its decomposition,
            see :class:`qibo.transpiler.unroller.Unroller`. Defaults to ``True``.
        minimize_toffolis (bool, optional): If ``True``, the multi-controlled
            :class:`qibo.gates.X` gates with auxiliary qubits are decomposed with the lowest
            number of Toffoli gates, see :class:`qibo.transpiler.unroller.Unroller`. It
            cannot be used together with ``minimize_depth``. Defaults to ``None``, which is
            ``qubits_initially_zero`` unless ``minimize_depth`` is ``True``.
        minimize_depth (bool, optional): If ``True``, the multi-controlled
            :class:`qibo.gates.X` gates with auxiliary qubits are decomposed with a depth
            that is logarithmic in the number of controls, see
            :class:`qibo.transpiler.unroller.Unroller`. It cannot be used together with
            ``minimize_toffolis``. Defaults to ``False``.
        routing_trials (int, optional): Number of independent routing attempts of the
            ``"sabre"`` router, which keeps the best one, see
            :class:`qibo.transpiler.router.Sabre`. It is not used by the other routers, nor
            by the router of the ``"reverse_traversal"`` placer. Defaults to ``None``, which is
            :math:`20` if ``optimization_level`` is :math:`2` and :math:`5` otherwise.
        optimization_level (int, optional): Optimization level of the pipeline. It can be
            :math:`0`, which is the pipeline described above, that only makes the circuit
            executable on the hardware and does not optimize it, or :math:`1` or :math:`2`,
            which add the optimizers described above. Defaults to :math:`0`.
        sampling_only (bool, optional): If ``True``, the optimizers of levels :math:`1` and
            :math:`2` also use the rewrites that keep the distribution of the measured
            qubits, but can change the state of the circuit before the measurements:
            :class:`qibo.transpiler.optimizer.OptimizeSwapBeforeMeasure`,
            :class:`qibo.transpiler.optimizer.RemoveDiagonalGatesBeforeMeasurement` and
            :class:`qibo.transpiler.optimizer.RemoveFinalReset`, before the layout. It has
            no effect if ``optimization_level`` is :math:`0`. Defaults to ``False``.
        post_layout (:class:`qibo.transpiler.abstract.PostLayout`, optional): Pass that
            chooses other physical qubits for the routed circuit, *e.g.*
            :class:`qibo.transpiler.placer.NoiseAwarePostLayout`, which moves it to the
            least noisy qubits that can run it. It runs at the end of the pipeline, except
            on a directed ``connectivity``, and it updates the final layout. It can skip
            itself if it has nothing to do after the placer, see
            :meth:`qibo.transpiler.abstract.PostLayout.skip`, which is told the placer that
            decided the layout. It requires ``connectivity``. Defaults to ``None``, so the
            qubits are the ones that the placer chose, unless ``hardware`` is given and the
            layout is searched at level :math:`1` or :math:`2`, and then it is
            :class:`qibo.transpiler.placer.NoiseAwarePostLayout` with ``trust_placer``.
        hardware (:class:`qibo.transpiler.hardware.HardwareModel`, optional): Hardware model
            whose errors the :class:`qibo.transpiler.placer.NoiseAwarePlacer` of the layout
            search uses, so that the layout without SWAP gates that it finds is the one with
            the least noise. Its connectivity must be ``connectivity``, or a graph that
            contains it. If ``None``, the placer takes the first layout without SWAP gates
             that it finds, and the optimizer chooses the shortest gates. A warning is logged
            if no pass uses it, *e.g.* if ``optimization_level`` is :math:`0`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used by the
            unrollers to compute the gate matrices, unless another one is given when the
            pipeline is called. If ``None``, it uses the global backend. Defaults to ``None``.

    Returns:
        :class:`qibo.transpiler.pipeline.Passes`: Transpiler pipeline.

    Raises:
        ConnectivityError: If ``initial_layout``, ``layout_method`` or ``routing_method``
            is given without ``connectivity``, or if the qubits and couplings of
            ``connectivity`` are not in ``hardware``.
        PlacementError: If ``initial_layout`` is given with a non-trivial ``layout_method``.
        ValueError: If ``layout_method`` or ``routing_method`` is not a valid method, if
            ``post_layout`` or ``hardware`` is given without ``connectivity``, if
            ``approximation_degree`` is not between :math:`0` and :math:`1`, if
            ``minimize_toffolis`` and ``minimize_depth`` are both ``True``, if
            ``routing_trials`` is not positive and the router is ``"sabre"``, or if
            ``optimization_level`` is not a valid level.
    """
    if optimization_level not in _OPTIMIZATION_LEVELS:
        raise_error(
            ValueError,
            f"``optimization_level`` must be one of {list(_OPTIMIZATION_LEVELS)}, "
            + f"but it is {optimization_level!r}.",
        )

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
        "use_dirty_ancillas": use_dirty_ancillas,
        "qubits_initially_zero": qubits_initially_zero,
        "minimize_toffolis": (
            qubits_initially_zero and not minimize_depth
            if minimize_toffolis is None
            else minimize_toffolis
        ),
        "minimize_depth": minimize_depth,
    }
    synthesis = {
        "approximation_degree": approximation_degree,
        "threshold": threshold,
        "weight": weight,
    }
    before_layout, before_unroller, after_unroller = _optimization_stages(
        optimization_level, native_gates, sampling_only, approximation_degree, hardware
    )

    if post_layout is not None and connectivity is None:
        raise_error(ValueError, "``post_layout`` requires ``connectivity``.")

    if hardware is not None:
        if connectivity is None:
            raise_error(ValueError, "``hardware`` requires ``connectivity``.")

        couplings = hardware.connectivity
        if not set(connectivity.nodes) <= set(couplings.nodes) or any(
            not (couplings.has_edge(*edge) or couplings.has_edge(*edge[::-1]))
            for edge in connectivity.edges
        ):
            raise_error(
                ConnectivityError,
                "The connectivity must be a subgraph of the one of ``hardware``.",
            )

    if connectivity is None:
        passes = [Validation()]
        if optimization_level == 0:
            passes.append(Unroller(native_gates, **synthesis, **unroller_kwargs))
        else:
            passes.append(Unroller(native_gates, min_qubits=3, **unroller_kwargs))
            passes.extend([*before_layout, *before_unroller])
            passes.append(Unroller(native_gates, **synthesis, backend=backend))
            passes.extend(after_unroller)

        return Passes(
            passes=passes,
            native_gates=native_gates,
            on_qubits=on_qubits,
            callback=callback,
        )

    # the layout is searched by default from the first level, unless the user gives a layout
    # or a router that does not work with it
    default_layout = (
        layout_method is None
        and initial_layout is None
        and optimization_level >= 1
        and (routing_method is None or routing_method == "sabre")
    )
    if default_layout:
        layout_method = "reverse_traversal"

    # the model that the user gave, since the layout search makes one if there is none
    noise_aware = hardware is not None

    if layout_trials is None:
        layout_trials = (1, 5, 20)[optimization_level]

    if routing_trials is None:
        routing_trials = 20 if optimization_level == 2 else 5

    placers = {
        "dense": Dense,
        "random": lambda: Random(seed=seed),
        "reverse_traversal": lambda: ReverseTraversal(
            routing_algorithm=Sabre(seed=seed),
            trials=layout_trials,
            seed=seed,
            keep_layout=default_layout,
        ),
        "star": StarConnectivityPlacer,
        "subgraph": Subgraph,
    }

    routers = {
        "none": NoRouting,
        "sabre": lambda: Sabre(seed=seed, trials=routing_trials),
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

    layout_passes = [] if placer is None else [placer]
    if default_layout:
        # a layout without SWAP gates is the best one, so it is searched before the
        # reverse traversal, which only searches if there is none. The trivial layout is
        # kept if it is one, unless the least noisy one is wanted from the second level
        if hardware is None:
            hardware = HardwareModel(
                tuple(connectivity.nodes), connectivity, native_gates
            )
        layout_passes.insert(
            0,
            NoiseAwarePlacer(
                hardware,
                call_limit=(50000, 1000) if optimization_level == 1 else int(1e6),
                keep_layout=bool(not (noise_aware and optimization_level >= 2)),
            ),
        )
        # the layout of the search is only improved if the search did not find the best one
        if post_layout is None and noise_aware:
            post_layout = NoiseAwarePostLayout(hardware, trust_placer=True)

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

    # the final measurements stay after the SWAP gates of the routers that are known to work
    # with a barrier, which is removed afterwards with the others on all the qubits, while
    # the barriers on some of them stay
    nqubits = connectivity.number_of_nodes() if on_qubits is None else len(on_qubits)
    protection = (
        [BarrierBeforeFinalMeasurements()],
        [FilterGates(partial(_is_not_barrier, nqubits=nqubits))],
    )
    if not isinstance(
        router, (NoRouting, Sabre, ShortestPaths, StarConnectivityRouter)
    ):
        protection = ([], [])

    passes = [
        Validation(initial_layout=initial_layout),
        Unroller(native_gates, min_qubits=3, **unroller_kwargs),
        *before_layout,
        Preprocessing(initial_layout=initial_layout),
        *layout_passes,
        *protection[0],
        router,
        *protection[1],
        *before_unroller,
        Unroller(native_gates, **synthesis, backend=backend),
    ]
    # the post layout runs last, so that it sees the final gates, unless the pass that
    # sets the direction of the gates has to run after it, because it depends on the qubits
    last = [] if post_layout is None else [post_layout]
    if connectivity.is_directed():
        # the unroller does not choose the direction of the two-qubit gates, and the
        # gates that reverse them are not necessarily native. They are decomposed exactly,
        # without ``approximation_degree``, since they are simple gates that cannot be
        # approximated with fewer entangling gates.
        passes.extend([*last, GateDirection(), Unroller(native_gates, backend=backend)])
        last = []
    passes.extend([*after_unroller, *last])

    if noise_aware and not any(
        getattr(item, "hardware", None) is hardware
        for transpiler_pass in passes
        for item in [transpiler_pass, *getattr(transpiler_pass, "optimizers", [])]
    ):
        log.warning(
            "``hardware`` has no effect, because no pass of the pipeline uses it with "
            + "this configuration."
        )

    return Passes(
        passes=passes,
        connectivity=connectivity,
        native_gates=native_gates,
        on_qubits=on_qubits,
        callback=callback,
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
    initial_layout: list[int | str] | dict[int, int | str] | None = None,
    seed: int | None = None,
    on_qubits: list[int | str] | None = None,
    qubits_initially_zero: bool = False,
    layout_method: str | Placer | None = None,
    routing_method: str | Router | None = None,
    layout_trials: int | None = None,
    callback: Callable | None = None,
    approximation_degree: float = 1.0,
    threshold: float | None = None,
    weight: float | None = None,
    use_dirty_ancillas: bool = True,
    minimize_toffolis: bool | None = None,
    minimize_depth: bool = False,
    routing_trials: int | None = None,
    optimization_level: int = 0,
    sampling_only: bool = False,
    post_layout: PostLayout | None = None,
    hardware: HardwareModel | None = None,
    num_processes: int | None = None,
    backend: Backend | None = None,
) -> tuple[Circuit, dict | None] | list[tuple[Circuit, dict | None]]:
    """Transpile one or more circuits in a single call.

    It builds the pipeline of :func:`qibo.transpiler.pipeline.generate_preset_passes` once
    and applies it to the circuits, which makes them executable on the hardware, and
    optimizes them if ``optimization_level`` is higher than :math:`0`.
    All the arguments except ``circuits`` and ``num_processes`` are the ones of
    :func:`qibo.transpiler.pipeline.generate_preset_passes`, and they are shared by all the
    circuits.

    The circuits of a list can be transpiled in parallel with ``num_processes``. Each of
    them is then transpiled by a worker process with its own copy of the pipeline, so the
    result for a circuit does not depend on the other circuits nor on ``num_processes``,
    and, with a ``seed``, it is the one of transpiling the circuit alone. This can differ
    from the sequential result, where the random state of the placer and the router
    continues from one circuit to the next. The worker processes are started with the
    ``spawn`` method, so the script that calls :func:`qibo.transpiler.pipeline.transpile`
    must protect the call with ``if __name__ == "__main__":``. They do not share the global
    backend either, so ``backend`` should be given if it is not the default one.

    Example:
        A rotation, a Toffoli gate and a CNOT gate between qubits that are not neighbors are
        transpiled to a line of four qubits, where :class:`qibo.gates.CNOT` and
        :class:`qibo.gates.U3` are the native gates. The Toffoli gate is decomposed, the SWAP
        gates that the routing adds are decomposed too, and the transpiled circuit is drawn.
        The decomposition of the Toffoli gate leaves a global phase, which the drawing
        reports. If ``circuits`` is a list, the result is a list with a transpiled circuit
        and a final layout for each circuit.

        .. testcode::

            import networkx as nx

            from qibo import Circuit, gates
            from qibo.backends import NumpyBackend
            from qibo.transpiler import NativeGates, transpile
            from qibo.transpiler.asserts import assert_transpiling

            backend = NumpyBackend()

            circuit = Circuit(4)
            circuit.add(gates.RY(0, 0.5))
            circuit.add(gates.TOFFOLI(0, 1, 2))
            circuit.add(gates.CNOT(0, 3))
            circuit.add(gates.M(*range(4)))

            connectivity = nx.path_graph(4)
            natives = NativeGates.from_gatelist([gates.U3, gates.CNOT, gates.M])
            transpiled_circuit, final_layout = transpile(
                circuit, connectivity, natives, seed=42, backend=backend
            )

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

            {0: 2, 1: 1, 2: 0, 3: 3}
            Global phase = exp(i * 9π/8)
            0: ─U3────────────o─────────o───────o─X─o────────────────M─
            1: ────o────o─X─o─X─RZ─X─RZ─X─RZ─U3─X─o─X─o─RZ─o─o─X─o───M─
            2: ─U3─X─RZ─X─o─X──────o─RZ───────────────X─RZ─X─X─o─X─o─M─
            3: ────────────────────────────────────────────────────X─M─

    Args:
        circuits (:class:`qibo.models.circuit.Circuit` or list or tuple): Circuit, or list of
            circuits, to be transpiled.
        connectivity (:class:`networkx.Graph`, optional): Hardware connectivity.
            Defaults to ``None``.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`, optional): Native gates
            supported by the hardware. Defaults to
            :meth:`qibo.transpiler.unroller.NativeGates.default`.
        initial_layout (list or dict, optional): Physical qubits on which the logical qubits
            of the circuits are placed, see
            :class:`qibo.transpiler.optimizer.Preprocessing`. Defaults to ``None``.
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
        layout_trials (int, optional): Number of layout attempts of the
            ``"reverse_traversal"`` placer. Defaults to ``None``, which is :math:`1`, :math:`5`
            or :math:`20` if ``optimization_level`` is :math:`0`, :math:`1` or :math:`2`.
        callback (Callable, optional): Function called after each pass that runs, for each
            circuit, see :class:`qibo.transpiler.pipeline.Passes`. It cannot be used with
            more than one process. Defaults to ``None``.
        approximation_degree (float, optional): Fidelity of each entangling gate, between
            :math:`0` and :math:`1`. If it is lower than :math:`1`, the two-qubit gates that
            are decomposed from their matrix are approximated with fewer entangling gates,
            if it increases the expected fidelity, see
            :func:`qibo.transpiler.pipeline.generate_preset_passes`. Defaults to :math:`1`,
            which is the exact decomposition.
        threshold (float, optional): Tolerance of the decomposition of the two-qubit gates
            that are decomposed from their matrix, like for ``approximation_degree``.
            Defaults to ``None``, which uses the default value of the decomposition.
        weight (float, optional): Weight of the imaginary part in the matrix that is
            diagonalized to decompose those gates, like for ``approximation_degree``.
            Defaults to ``None``, which uses the default value of the decomposition.
        use_dirty_ancillas (bool, optional): If ``True``, the qubits that a multi-controlled
            gate does not act on are used as dirty auxiliary qubits in its decomposition.
            Defaults to ``True``.
        minimize_toffolis (bool, optional): If ``True``, the multi-controlled
            :class:`qibo.gates.X` gates with auxiliary qubits are decomposed with the lowest
            number of Toffoli gates. It cannot be used together with ``minimize_depth``.
            Defaults to ``None``, which is ``qubits_initially_zero`` unless
            ``minimize_depth`` is ``True``.
        minimize_depth (bool, optional): If ``True``, the multi-controlled
            :class:`qibo.gates.X` gates with auxiliary qubits are decomposed with a depth
            that is logarithmic in the number of controls. It cannot be used together with
            ``minimize_toffolis``. Defaults to ``False``.
        routing_trials (int, optional): Number of independent routing attempts of the
            ``"sabre"`` router, which keeps the best one. It is not used by the other
            routers. Defaults to ``None``, which is :math:`20` if ``optimization_level`` is
            :math:`2` and :math:`5` otherwise.
        optimization_level (int, optional): Optimization level of the pipeline, see
            :func:`qibo.transpiler.pipeline.generate_preset_passes`. It can be :math:`0`,
            :math:`1` or :math:`2`. Defaults to :math:`0`, which does not optimize the circuits.
        sampling_only (bool, optional): If ``True``, the optimizers also use the rewrites
            that keep the distribution of the measured qubits but can change the state of
            the circuit before the measurements, see
            :func:`qibo.transpiler.pipeline.generate_preset_passes`. It logs a warning
            for the circuits without measurements, and it has no effect if
            ``optimization_level`` is :math:`0`. Defaults to ``False``.
        post_layout (:class:`qibo.transpiler.abstract.PostLayout`, optional): Pass that
            chooses other physical qubits for the routed circuits, see
            :func:`qibo.transpiler.pipeline.generate_preset_passes`. All the circuits use
            it, and each worker process gets its own copy. It requires ``connectivity``.
            Defaults to ``None``, which is a post layout of ``hardware`` if it is given and
            the default layout search is run, see
            :func:`qibo.transpiler.pipeline.generate_preset_passes`.
        hardware (:class:`qibo.transpiler.hardware.HardwareModel`, optional): Hardware model
            whose errors the placer of the layout search, the post layout and the
            single-qubit optimizers use to choose the layout and the single-qubit gates with
            the least noise, see :func:`qibo.transpiler.pipeline.generate_preset_passes`. It
            requires ``connectivity``. Defaults to ``None``.
        num_processes (int, optional): Number of worker processes that transpile the
            circuits of a list in parallel. If ``None`` or :math:`1`, the circuits are
            transpiled one after the other in the current process. It has no effect on a
            single circuit, and no more processes than circuits are started.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): Backend used by the
            unrollers to compute the gate matrices, and passed to the other passes that take
            one. If ``None``, they use the global backend. Defaults to ``None``.

    Returns:
        (:class:`qibo.models.circuit.Circuit`,
        :class:`qibo.transpiler.layout.TranspileLayout`) or list: Transpiled circuit and
        final {logical: physical} qubit mapping, see
        :class:`qibo.transpiler.pipeline.Passes`. If ``circuits`` is a list or tuple, the
        list of these pairs, in the same order.

    Raises:
        ValueError: If ``num_processes`` is not a positive integer, if ``callback`` is
            given with more than one process, if ``approximation_degree`` is not between
            :math:`0` and :math:`1`, if ``minimize_toffolis`` and ``minimize_depth`` are
            both ``True``, if ``routing_trials`` is not positive and the router is
            ``"sabre"``, or if ``optimization_level`` is not a valid level.
        BrokenProcessPool: If a worker process ends abruptly. A common cause is a script
            that calls this function with several processes without the
            ``if __name__ == "__main__":`` guard.
    """
    if num_processes is not None and (
        not isinstance(num_processes, int) or num_processes < 1
    ):
        raise_error(
            ValueError,
            f"``num_processes`` must be a positive integer, but it is {num_processes!r}.",
        )
    if callback is not None and num_processes is not None and num_processes > 1:
        raise_error(
            ValueError,
            "``callback`` cannot be used with more than one process, because it would "
            + "be called in the worker processes.",
        )
    if sampling_only and optimization_level > 0:
        for circuit in [circuits] if isinstance(circuits, Circuit) else circuits:
            if not circuit.measurements:
                log.warning(
                    "``sampling_only`` has no effect on a circuit without measurements."
                )

    passes = generate_preset_passes(
        connectivity=connectivity,
        native_gates=native_gates,
        initial_layout=initial_layout,
        seed=seed,
        on_qubits=on_qubits,
        qubits_initially_zero=qubits_initially_zero,
        layout_method=layout_method,
        routing_method=routing_method,
        layout_trials=layout_trials,
        callback=callback,
        approximation_degree=approximation_degree,
        threshold=threshold,
        weight=weight,
        use_dirty_ancillas=use_dirty_ancillas,
        minimize_toffolis=minimize_toffolis,
        minimize_depth=minimize_depth,
        routing_trials=routing_trials,
        optimization_level=optimization_level,
        sampling_only=sampling_only,
        post_layout=post_layout,
        hardware=hardware,
        backend=backend,
    )

    if isinstance(circuits, Circuit):
        results = passes(circuits, backend=backend)
    elif num_processes is None or num_processes == 1 or len(circuits) < 2:
        results = [passes(circuit, backend=backend) for circuit in circuits]
    else:
        # each task gets its own copy of the pipeline, so that the results are independent
        try:
            with ProcessPoolExecutor(
                max_workers=min(num_processes, len(circuits)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                results = list(executor.map(passes, circuits, repeat(backend)))
        except BrokenProcessPool:
            raise_error(
                BrokenProcessPool,
                "A worker process ended abruptly. The workers start by importing the "
                + "script that calls ``transpile``, which fails if the call is not "
                + 'protected by ``if __name__ == "__main__":``. Running out of memory is '
                + "another possible cause.",
            )

    return results


def _is_not_barrier(gate: Gate, nqubits: int) -> bool:
    """Check that a gate is not a barrier on all the qubits of the circuit.

    It is a function of the module, and not a lambda, so that the pipeline can be sent to
    the worker processes of :func:`qibo.transpiler.pipeline.transpile`.

    Args:
        gate (:class:`qibo.gates.abstract.Gate`): Gate to be checked.
        nqubits (int): Number of qubits of the circuit.

    Returns:
        bool: ``False`` if the gate is a barrier that acts on ``nqubits`` qubits, and ``True``
        otherwise.
    """
    return bool(not (isinstance(gate, gates.Barrier) and len(gate.qubits) == nqubits))


def _optimization_stages(
    optimization_level: int,
    native_gates: NativeGates,
    sampling_only: bool,
    approximation_degree: float,
    hardware: HardwareModel | None,
) -> tuple[list[Optimizer], list[Optimizer], list[Optimizer]]:
    """Optimizers of the optimization level, see :func:`generate_preset_passes`.

    Args:
        optimization_level (int): Optimization level.
        native_gates (:class:`qibo.transpiler.unroller.NativeGates`): Native gates of the
            hardware.
        sampling_only (bool): If ``True``, the optimizers that keep only the distribution
            of the measured qubits are included.
        approximation_degree (float): Fidelity of each entangling gate, which sets how
            close to the identity a gate has to be to be removed.
        hardware (:class:`qibo.transpiler.hardware.HardwareModel` or None): Hardware model
            whose errors the single-qubit optimizers use to choose the gates that replace a
            run.

    Returns:
        tuple[list, list, list]: Optimizers to run before the layout, before the last
        unroller, and after it.
    """
    before_layout, before_unroller, after_unroller = [], [], []
    if optimization_level == 0:
        return before_layout, before_unroller, after_unroller

    if sampling_only:
        before_layout.extend(
            [
                OptimizeSwapBeforeMeasure(),
                RemoveDiagonalGatesBeforeMeasurement(),
                RemoveFinalReset(),
            ]
        )
    before_layout.extend([InverseCancellation(), CommutativeCancellation()])
    before_unroller.append(ConsolidateBlocks())

    # the single-qubit optimizer does nothing if the native gates have no Euler basis
    basis = native_gates.gate_names
    has_euler_basis = bool(Optimize1qGatesDecomposition(basis=basis).euler_bases)
    loop = [InverseCancellation()]
    if has_euler_basis:
        loop.insert(0, Optimize1qGatesDecomposition(basis=basis, hardware=hardware))
    after_unroller.append(FixedPoint(loop))

    if optimization_level >= 2:
        before_layout.append(RemoveIdentityEquivalent(approximation_degree))
        before_unroller.append(CommutativeCancellation())
        if has_euler_basis:
            after_unroller.append(
                FixedPoint(
                    [
                        Optimize1qGatesDecomposition(basis=basis, hardware=hardware),
                        Optimize1qGatesSimpleCommutation(basis=basis),
                        InverseCancellation(commutation=True),
                    ]
                )
            )

    return before_layout, before_unroller, after_unroller
