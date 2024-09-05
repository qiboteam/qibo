from typing import Optional, Union

import networkx as nx

from qibo import gates
from qibo.backends import _check_backend_and_local_state
from qibo.config import log, raise_error
from qibo.models import Circuit
from qibo.transpiler._exceptions import PlacementError
from qibo.transpiler.abstract import Placer, Router
from qibo.transpiler.router import _find_connected_qubit


def assert_placement(
    circuit: Circuit, layout: dict, connectivity: nx.Graph = None
) -> bool:
    """Check if layout is in the correct form and matches the number of qubits of the circuit.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): Circuit model to check.
        layout (dict): physical to logical qubit mapping.
        connectivity (:class:`networkx.Graph`, optional): Chip connectivity.
            This argument is necessary if the layout is applied to a subset of
            qubits of the original connectivity graph. Defaults to ``None``.
    """
    assert_mapping_consistency(layout=layout, connectivity=connectivity)
    if circuit.nqubits > len(layout):
        raise_error(
            PlacementError,
            "Layout can't be used on circuit. The circuit requires more qubits.",
        )
    if circuit.nqubits < len(layout):
        raise_error(
            PlacementError,
            "Layout can't be used on circuit. "
            + "Ancillary extra qubits need to be added to the circuit.",
        )


def assert_mapping_consistency(layout: dict, connectivity: nx.Graph = None):
    """Check if layout is in the correct form.

    Args:
        layout (dict): physical to logical qubit mapping.
        connectivity (:class:`networkx.Graph`, optional):  Chip connectivity.
            This argument is necessary if the layout is applied to a subset of
            qubits of the original connectivity graph. Defaults to ``None``.
    """
    values = sorted(layout.values())
    physical_qubits = list(layout)
    nodes = (
        list(range(len(values))) if connectivity is None else list(connectivity.nodes)
    )
    ref_keys = (
        ["q" + str(i) for i in nodes] if isinstance(physical_qubits[0], str) else nodes
    )
    if sorted(physical_qubits) != sorted(ref_keys):
        raise_error(
            PlacementError,
            "Some physical qubits in the layout may be missing or duplicated.",
        )
    if values != list(range(len(values))):
        raise_error(
            PlacementError,
            "Some logical qubits in the layout may be missing or duplicated.",
        )


def _find_gates_qubits_pairs(circuit: Circuit):
    """Helper method for :meth:`qibo.transpiler.placer`.
    Translate circuit into a list of pairs of qubits to be used by the router and placer.

    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

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
        connectivity (:class:`networkx.Graph`): chip connectivity, not used for this transpiler.
        middle_qubit (int, optional): qubit id of the qubit that is in the middle of the star.
    """

    def __init__(self, connectivity=None, middle_qubit: int = 2):
        self.middle_qubit = middle_qubit
        if connectivity is not None:  # pragma: no cover
            log.warning(
                "StarConnectivityRouter does not use the connectivity graph."
                "The connectivity graph will be ignored."
            )

    def __call__(self, circuit: Circuit):
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): The original Qibo circuit to transform.
                Only single qubit gates and two qubits gates are supported by the router.

        Returns:
            dict: physical to logical qubit mapping.
        """

        # find the number of qubits for hardware circuit
        nqubits = max(circuit.nqubits, self.middle_qubit + 1)
        hardware_qubits = list(range(nqubits))

        for i, gate in enumerate(circuit.queue):
            if len(gate.qubits) > 2:
                raise_error(
                    PlacementError,
                    "Gates targeting more than 2 qubits are not supported",
                )
            if len(gate.qubits) == 2:
                if self.middle_qubit not in gate.qubits:
                    new_middle = _find_connected_qubit(
                        gate.qubits,
                        circuit.queue[i + 1 :],
                        hardware_qubits,
                        error=PlacementError,
                    )
                    hardware_qubits[self.middle_qubit], hardware_qubits[new_middle] = (
                        new_middle,
                        self.middle_qubit,
                    )
                    break

        return dict(zip(["q" + str(i) for i in range(nqubits)], hardware_qubits))


class Trivial(Placer):
    """Place qubits according to the following notation:

    .. math::
        \\{\\textup{"q0"} : 0, \\textup{"q1"} : 1, ..., \\textup{"qn"} : n}.

    Args:
        connectivity (networkx.Graph, optional): chip connectivity.
    """

    def __init__(self, connectivity: nx.Graph = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        """Find the trivial placement for the circuit.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        if self.connectivity is not None:
            if self.connectivity.number_of_nodes() != circuit.nqubits:
                raise_error(
                    PlacementError,
                    "The number of nodes of the connectivity graph must match "
                    + "the number of qubits in the circuit",
                )
            trivial_layout = dict(
                zip(
                    ["q" + str(i) for i in list(self.connectivity.nodes())],
                    range(circuit.nqubits),
                )
            )
        else:
            trivial_layout = dict(
                zip(
                    ["q" + str(i) for i in range(circuit.nqubits)],
                    range(circuit.nqubits),
                )
            )
        return trivial_layout


class Custom(Placer):
    """Define a custom initial qubit mapping.

    Args:
        map (list or dict): physical to logical qubit mapping.
            Examples: :math:`[1,2,0]` or
            :math:`{\\textup{"q0"}: 1, \\textup{"q1"}: 2, \\textup{"q2"}:0}`
            to assign the physical qubits :math:`\\{0, 1, 2\\}`
            to the logical qubits :math:`[1, 2, 0]`.
        connectivity (:class:`networkx.Graph`, optional): chip connectivity.
            This argument is necessary if the layout applied to a subset of
            qubits of the original connectivity graph. Defaults to ``None``.
    """

    def __init__(self, initial_map: Union[list, dict], connectivity: nx.Graph = None):
        self.connectivity = connectivity
        self.initial_map = initial_map

    def __call__(self, circuit=None):
        """Return the custom placement if it can be applied to the given circuit (if given).

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        if isinstance(self.initial_map, dict):
            pass
        elif isinstance(self.initial_map, list):
            if self.connectivity is not None:
                self.initial_map = dict(
                    zip(
                        ["q" + str(i) for i in self.connectivity.nodes()],
                        self.initial_map,
                    )
                )
            else:
                self.initial_map = dict(
                    zip(
                        ["q" + str(i) for i in range(len(self.initial_map))],
                        self.initial_map,
                    )
                )
        else:
            raise_error(TypeError, "Use dict or list to define mapping.")
        if circuit is not None:
            assert_placement(circuit, self.initial_map, connectivity=self.connectivity)
        else:
            assert_mapping_consistency(self.initial_map, connectivity=self.connectivity)
        return self.initial_map


class Subgraph(Placer):
    """
    Subgraph isomorphism qubit placer.

    Since it is a :math:`NP`-complete problem, it can take exponential time for large circuits.
    This initialization method may fail for very short circuits.

    Attributes:
        connectivity (:class:`networkx.Graph`): chip connectivity.
    """

    def __init__(self, connectivity: nx.Graph):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        """Find the initial layout of the given circuit using subgraph isomorphism.
        Circuit must contain at least two two-qubit gates to implement subgraph placement.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        if len(gates_qubits_pairs) < 3:
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

        sorted_result = dict(sorted(result.mapping.items()))

        return {"q" + str(k): v for k, v in sorted_result.items()}


class Random(Placer):
    """
    Random initialization with greedy policy, let a maximum number of 2-qubit
    gates can be applied without introducing any SWAP gate.

    Attributes:
        connectivity (:class:`networkx.Graph`): chip connectivity.
        samples (int, optional): number of initial random layouts tested.
            Defaults to :math:`100`.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
    """

    def __init__(self, connectivity, samples: int = 100, seed=None):
        self.connectivity = connectivity
        self.samples = samples
        self.seed = seed

    def __call__(self, circuit):
        """Find an initial layout of the given circuit using random greedy algorithm.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): Circuit to be transpiled.

        Returns:
            (dict): physical-to-logical qubit mapping.
        """
        _, local_state = _check_backend_and_local_state(self.seed, backend=None)
        gates_qubits_pairs = _find_gates_qubits_pairs(circuit)
        nodes = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())
        dict_keys = ["q" + str(i) for i in keys]

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
                final_layout = dict(zip(dict_keys, list(mapping.values())))
                return dict(sorted(final_layout.items()))

            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost

        final_layout = dict(zip(dict_keys, list(final_mapping.values())))
        return dict(sorted(final_layout.items()))

    def _cost(self, graph: nx.Graph, gates_qubits_pairs: list):
        """
        Compute the cost associated to an initial layout as the lengh of the reduced circuit.

        Args:
            graph (:class:`networkx.Graph`): current hardware qubit mapping.
            gates_qubits_pairs (list): circuit representation.

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
        connectivity (:class:`networkx.Graph`): chip connectivity.
        routing_algorithm (:class:`qibo.transpiler.abstract.Router`): routing algorithm.
        depth (int, optional): number of two-qubit gates considered before finding initial layout.
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
        connectivity: nx.Graph,
        routing_algorithm: Router,
        depth: Optional[int] = None,
    ):
        self.connectivity = connectivity
        self.routing_algorithm = routing_algorithm
        self.depth = depth

    def __call__(self, circuit: Circuit):
        """Find the initial layout of the given circuit using Reverse Traversal placement.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        initial_placer = Trivial(self.connectivity)
        initial_placement = initial_placer(circuit=circuit)
        self.routing_algorithm.connectivity = self.connectivity
        new_circuit = self._assemble_circuit(circuit)
        final_placement = self._routing_step(initial_placement, new_circuit)

        return final_placement

    def _assemble_circuit(self, circuit: Circuit):
        """Assemble a single circuit to apply Reverse Traversal placement based on depth.

        Example: for a circuit with four two-qubit gates :math:`A-B-C-D`
        using depth :math:`d = 6`, the function will return the circuit :math:`C-D-D-C-B-A`.

        Args:
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be transpiled.

        Returns:
            (:class:`qibo.models.circuit.Circuit`): assembled circuit to perform
                Reverse Traversal placement.
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

        new_circuit = Circuit(circuit.nqubits)
        for qubits in assembled_gates_qubits_pairs:
            # As only the connectivity is important here we can replace everything with CZ gates
            new_circuit.add(gates.CZ(qubits[0], qubits[1]))

        return new_circuit.invert()

    def _routing_step(self, layout: dict, circuit: Circuit):
        """Perform routing of the circuit.

        Args:
            layout (dict): intial qubit layout.
            circuit (:class:`qibo.models.circuit.Circuit`): circuit to be routed.
        """
        _, final_mapping = self.routing_algorithm(circuit, layout)

        return final_mapping
