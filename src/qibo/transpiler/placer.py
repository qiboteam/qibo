from typing import Optional

import networkx as nx

from qibo import gates
from qibo.backends import _check_backend_and_local_state
from qibo.config import raise_error
from qibo.models import Circuit
from qibo.transpiler._exceptions import ConnectivityError, PlacementError
from qibo.transpiler.abstract import Placer, Router
from qibo.transpiler.asserts import assert_placement
from qibo.transpiler.router import _find_connected_qubit


def _find_gates_qubits_pairs(circuit: Circuit):
    """Helper method for :meth:`qibo.transpiler.placer`.
    Translate circuit into a list of pairs of qubits to be used by the router and placer.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be translated.

    Returns:
        (list): Pairs of qubits targeted by two qubits gates.
    """
    gates_qubits_pairs = []
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            pass
        elif len(gate.qubits) == 2:
            gates_qubits_pairs.append(sorted(gate.qubits))
        elif len(gate.qubits) >= 3:
            raise_error(
                ValueError, "Gates targeting more than 2 qubits are not supported"
            )
    return gates_qubits_pairs


class StarConnectivityPlacer(Placer):
    """Find an optimized qubit placement for the following connectivity:

             q
             |
        q -- q -- q
             |
             q

    Args:
        connectivity (:class:`networkx.Graph`): Star connectivity graph.
    """

    def __init__(self, connectivity: Optional[nx.Graph] = None):
        self.connectivity = connectivity
        self.middle_qubit = None

    def __call__(self, circuit: Circuit):
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): The original Qibo circuit to transform.
                Only single qubit gates and two qubits gates are supported by the router.
        """
        assert_placement(circuit, self.connectivity)
        self._check_star_connectivity()

        middle_qubit_idx = circuit.wire_names.index(self.middle_qubit)
        wire_names = circuit.wire_names.copy()

        for i, gate in enumerate(circuit.queue):
            if len(gate.qubits) > 2:
                raise_error(
                    PlacementError,
                    "Gates targeting more than 2 qubits are not supported",
                )
            if len(gate.qubits) == 2:
                if middle_qubit_idx not in gate.qubits:
                    new_middle = _find_connected_qubit(
                        gate.qubits,
                        circuit.queue[i + 1 :],
                        error=PlacementError,
                        mapping=list(range(circuit.nqubits)),
                    )

                    (
                        wire_names[middle_qubit_idx],
                        wire_names[new_middle],
                    ) = (
                        wire_names[new_middle],
                        wire_names[middle_qubit_idx],
                    )
                    break

        circuit.wire_names = wire_names

    def _check_star_connectivity(self):
        """Check if the connectivity graph is a star graph."""
        if len(self.connectivity.nodes) != 5:
            raise_error(
                ConnectivityError,
                f"This connectivity graph is not a star graph. Length of nodes provided: {len(self.connectivity.nodes)} != 5.",
            )
        for node in self.connectivity.nodes:
            if self.connectivity.degree(node) == 4:
                self.middle_qubit = node
            elif self.connectivity.degree(node) != 1:
                raise_error(
                    ConnectivityError,
                    "This connectivity graph is not a star graph. There is a node with degree different from 1 and 4.",
                )


class Subgraph(Placer):
    """
    Subgraph isomorphism qubit placer.

    Since it is a :math:`NP`-complete problem, it can take exponential time for large circuits.
    This initialization method may fail for very short circuits.

    Attributes:
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
    """

    def __init__(self, connectivity: Optional[nx.Graph] = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        """Find the initial layout of the given circuit using subgraph isomorphism.
        Circuit must contain at least two two-qubit gates to implement subgraph placement.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
        """
        assert_placement(circuit, self.connectivity)
        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        if len(gates_qubits_pairs) < 2:
            raise_error(
                ValueError,
                "Circuit must contain at least two two-qubit gates "
                + "to implement subgraph placement.",
            )
        circuit_subgraph = nx.Graph()
        circuit_subgraph.add_nodes_from(list(range(circuit.nqubits)))
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            self.connectivity, circuit_subgraph
        )
        i = 0
        circuit_subgraph.add_edge(gates_qubits_pairs[i][0], gates_qubits_pairs[i][1])
        while matcher.subgraph_is_monomorphic():
            result = matcher
            i += 1
            circuit_subgraph.add_edge(
                gates_qubits_pairs[i][0], gates_qubits_pairs[i][1]
            )
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                self.connectivity, circuit_subgraph
            )
            if (
                self.connectivity.number_of_edges()
                == circuit_subgraph.number_of_edges()
                or i == len(gates_qubits_pairs) - 1
            ):
                break

        circuit.wire_names = sorted(result.mapping, key=lambda k: result.mapping[k])


class Random(Placer):
    """
    Random initialization with greedy policy, let a maximum number of 2-qubit
    gates can be applied without introducing any SWAP gate.

    Attributes:
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
        samples (int, optional): Number of random initializations to try.
            Defaults to :math:`100`.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
    """

    def __init__(
        self, connectivity: Optional[nx.Graph] = None, samples: int = 100, seed=None
    ):
        self.connectivity = connectivity
        self.samples = samples
        self.seed = seed

    def __call__(self, circuit):
        """Find an initial layout of the given circuit using random greedy algorithm.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
        """
        assert_placement(circuit, self.connectivity)
        _, local_state = _check_backend_and_local_state(self.seed, backend=None)
        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        nodes = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())

        final_mapping = dict(zip(keys, range(nodes)))
        final_graph = nx.relabel_nodes(self.connectivity, final_mapping)
        final_cost = self._cost(final_graph, gates_qubits_pairs)
        for _ in range(self.samples):
            mapping = dict(
                zip(keys, local_state.choice(range(nodes), nodes, replace=False))
            )
            graph = nx.relabel_nodes(self.connectivity, mapping)
            cost = self._cost(graph, gates_qubits_pairs)

            if cost == 0:
                final_layout = dict(zip(keys, list(mapping.values())))
                circuit.wire_names = sorted(final_layout, key=final_layout.get)
                return

            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost

        final_layout = dict(zip(keys, list(final_mapping.values())))
        circuit.wire_names = sorted(final_layout, key=final_layout.get)

    def _cost(self, graph: nx.Graph, gates_qubits_pairs: list):
        """
        Compute the cost associated to an initial layout as the lengh of the reduced circuit.

        Args:
            graph (:class:`networkx.Graph`): Hardware connectivity.
            gates_qubits_pairs (list): Circuit representation.

        Returns:
            (int): lengh of the reduced circuit.
        """
        for allowed, gate in enumerate(gates_qubits_pairs):
            if gate not in graph.edges():
                return len(gates_qubits_pairs) - allowed - 1

        return 0


class ReverseTraversal(Placer):
    """
    Places qubits based on the algorithm proposed in Reference [1].

    Compatible with all the available ``Router``s.

    Args:
        connectivity (:class:`networkx.Graph`): Hardware connectivity.
        routing_algorithm (:class:`qibo.transpiler.abstract.Router`): Router to be used.
        depth (int, optional): Number of two-qubit gates to be considered for routing.
            If ``None`` just one backward step will be implemented.
            If depth is greater than the number of two-qubit gates in the circuit,
            the circuit will be routed more than once.
            Example: on a circuit with four two-qubit gates :math:`A-B-C-D`
            using depth :math:`d = 6`, the routing will be performed
            on the circuit :math:`C-D-D-C-B-A`.

    References:
        1. G. Li, Y. Ding, and Y. Xie,
        *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*.
        `arXiv:1809.02573 [cs.ET] <https://arxiv.org/abs/1809.02573>`_.
    """

    def __init__(
        self,
        routing_algorithm: Router,
        connectivity: Optional[nx.Graph] = None,
        depth: Optional[int] = None,
    ):
        self.connectivity = connectivity
        self.routing_algorithm = routing_algorithm
        self.depth = depth

    def __call__(self, circuit: Circuit):
        """Find the initial layout of the given circuit using Reverse Traversal placement.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.
        """
        assert_placement(circuit, self.connectivity)
        self.routing_algorithm.connectivity = self.connectivity
        new_circuit = self._assemble_circuit(circuit)
        self._routing_step(new_circuit)

    def _assemble_circuit(self, circuit: Circuit):
        """Assemble a single circuit to apply Reverse Traversal placement based on depth.

        Example: for a circuit with four two-qubit gates :math:`A-B-C-D`
        using depth :math:`d = 6`, the function will return the circuit :math:`C-D-D-C-B-A`.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be assembled.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): Assembled circuit to perform Reverse Traversal placement.
        """

        if self.depth is None:
            return circuit.invert()

        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        circuit_gates = len(gates_qubits_pairs)
        if circuit_gates == 0:
            raise_error(
                ValueError, "The circuit must contain at least a two-qubit gate."
            )
        repetitions, remainder = divmod(self.depth, circuit_gates)

        assembled_gates_qubits_pairs = []
        for _ in range(repetitions):
            assembled_gates_qubits_pairs += gates_qubits_pairs[:]
            gates_qubits_pairs.reverse()
        assembled_gates_qubits_pairs += gates_qubits_pairs[0:remainder]

        new_circuit = Circuit(circuit.nqubits, wire_names=circuit.wire_names)
        for qubits in assembled_gates_qubits_pairs:
            # As only the connectivity is important here we can replace everything with CZ gates
            new_circuit.add(gates.CZ(qubits[0], qubits[1]))

        return new_circuit.invert()

    def _routing_step(self, circuit: Circuit):
        """Perform routing of the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be routed.
        """
        _, final_mapping = self.routing_algorithm(circuit)

        return final_mapping
