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
        A CNOT gate between the first and last qubits of a line of three qubits needs a
        SWAP gate, which exchanges the second and last qubits.

        .. testcode::

            import networkx as nx

            from qibo import Circuit, gates
            from qibo.transpiler import Passes, Preprocessing, ShortestPaths

            circuit = Circuit(3)
            circuit.add(gates.CNOT(0, 2))

            connectivity = nx.path_graph(3)
            pipeline = Passes(
                [Preprocessing(), ShortestPaths(seed=1)], connectivity=connectivity
            )
            _, layout = pipeline(circuit)

            print(dict(layout))
            print(layout.initial_layout)
            print(layout.routing_permutation)

        .. testoutput::

            {0: 0, 1: 2, 2: 1}
            [0, 1, 2]
            [0, 2, 1]

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
