class TranspileLayout(dict):
    """Layout of the qubits of a circuit before and after the routing.

    It is the final layout returned by :class:`qibo.transpiler.pipeline.Passes`. As a
    dictionary, it maps the label of the physical qubit on which each qubit of the
    circuit is placed before the routing, to the index of the physical qubit on which it
    ends after the routing. It can therefore be used wherever a final
    {logical: physical} qubit mapping is expected. It also stores the layout before the
    routing. If the pipeline has no router, the qubits are not moved, so the mapping and
    the routing permutation are the identity, and the initial layout is the one of the
    transpiled circuit.

    Example:
        A circuit on four qubits is transpiled to a line of five qubits. It is drawn
        before and after the transpilation with ``parallel=True``, so that the gates
        on disjoint qubits share a column, and with ``line_wrap=12``, which splits the
        wider transpiled circuit in two blocks. The routing adds the SWAP gates, drawn as
        ``x``, and the preprocessing adds the fifth qubit of the line as an auxiliary
        qubit, which is not used. The :math:`i`-th element of the routing permutation is
        the physical qubit on which the :math:`i`-th qubit ends: the first qubit does
        not move, while the second, third and fourth qubits end on the third, fourth and
        second physical qubits.

        .. testcode::

            import networkx as nx

            from qibo import Circuit, gates
            from qibo.transpiler import Passes, Preprocessing, ShortestPaths

            circuit = Circuit(4)
            circuit.add(gates.H(0))
            circuit.add(gates.H(2))
            circuit.add(gates.CNOT(0, 1))
            circuit.add(gates.CNOT(2, 3))
            circuit.add(gates.CNOT(0, 3))
            circuit.add(gates.CNOT(1, 2))
            circuit.add(gates.M(0, 1, 2, 3))

            circuit.draw(parallel=True)
            print()

            pipeline = Passes(
                [Preprocessing(), ShortestPaths(seed=1)], connectivity=nx.path_graph(5)
            )
            transpiled_circuit, layout = pipeline(circuit)

            transpiled_circuit.draw(parallel=True)
            print()

            print(dict(layout))
            print(layout.initial_layout)
            print(layout.routing_permutation)
            print(layout.nqubits_input)
            print(layout.auxiliary_qubits)
            print(layout.initial_logical_layout(filter_auxiliary=True))
            print(layout.final_logical_layout(filter_auxiliary=True))

        .. testoutput::

            0: ─H─o─o───M─
            1: ───X─|─o─M─
            2: ─H─o─|─X─M─
            3: ───X─X───M─

            0: ─H─o───────o─M─
            1: ───X─o───x─X─M─
            2: ─H─o─X─x─x───M─
            3: ───X───x─────M─
            4: ───────────────

            {0: 0, 1: 2, 2: 3, 3: 1, 4: 4}
            [0, 1, 2, 3, 4]
            [0, 2, 3, 1, 4]
            4
            [4]
            {0: 0, 1: 1, 2: 2, 3: 3}
            {0: 0, 1: 2, 2: 3, 3: 1}

    Args:
        final_layout (dict): Mapping from the label of the physical qubit on which each
            qubit is placed before the routing, to the index of the physical qubit on
            which it ends after the routing.
        initial_layout (list): Labels of the physical qubits on which the qubits of the
            circuit are placed before the routing. The :math:`i`-th element is the
            physical qubit of the :math:`i`-th qubit of the circuit.
        nqubits_input (int): Number of qubits of the circuit before it is padded with
            the unused qubits of the hardware, which are the auxiliary qubits.
    """

    def __init__(self, final_layout: dict, initial_layout: list, nqubits_input: int):
        super().__init__(final_layout)
        self.initial_layout = list(initial_layout)
        self.nqubits_input = nqubits_input

    @property
    def auxiliary_qubits(self) -> list:
        """Labels of the physical qubits that the preprocessing adds to the circuit as
        auxiliary qubits."""
        return self.initial_layout[self.nqubits_input :]

    @property
    def final_layout(self) -> dict:
        """Mapping from the physical qubit on which each qubit is placed before the
        routing, to the one on which it ends after the routing, as a plain dictionary.
        """
        return dict(self)

    @property
    def routing_permutation(self) -> list[int]:
        """Permutation of the qubits that the routing applies. Its :math:`i`-th element
        is the index of the physical qubit on which the :math:`i`-th qubit of the
        circuit ends after the routing. The elements from ``nqubits_input`` onward are
        the auxiliary qubits."""
        return [self[label] for label in self.initial_layout]

    def final_logical_layout(self, filter_auxiliary: bool = False) -> dict[int, int]:
        """Index of the physical qubit on which each qubit of the circuit ends.

        Args:
            filter_auxiliary (bool, optional): If ``True``, the auxiliary qubits are left
                out. Defaults to ``False``.

        Returns:
            dict: Mapping from the index of each qubit of the circuit to the index of the
            physical qubit on which it ends after the routing, see
            :attr:`qibo.transpiler.layout.TranspileLayout.routing_permutation`.
        """
        stop = self.nqubits_input if filter_auxiliary else len(self.initial_layout)
        return dict(enumerate(self.routing_permutation[:stop]))

    def initial_logical_layout(
        self, filter_auxiliary: bool = False
    ) -> dict[int, int | str]:
        """Label of the physical qubit on which each qubit of the circuit is placed.

        Args:
            filter_auxiliary (bool, optional): If ``True``, the auxiliary qubits are left
                out. Defaults to ``False``.

        Returns:
            dict: Mapping from the index of each qubit of the circuit to the label of the
            physical qubit on which it is placed before the routing.
        """
        stop = self.nqubits_input if filter_auxiliary else len(self.initial_layout)
        return dict(enumerate(self.initial_layout[:stop]))
