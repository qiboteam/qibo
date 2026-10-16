from collections.abc import Callable
from functools import reduce

import networkx as nx
from numpy.typing import ArrayLike

from qibo import symbols
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians.hamiltonians import Hamiltonian, SymbolicHamiltonian


def _build_spin_model(
    nqubits: int, matrix: ArrayLike, condition: Callable, backend: Backend
) -> ArrayLike:
    """Helper method for building nearest-neighbor spin model Hamiltonians."""
    h = sum(
        reduce(
            backend.kron,
            (
                matrix if condition(qubit_i, qubit_j) else backend.matrices.I()
                for qubit_j in range(nqubits)
            ),
        )
        for qubit_i in range(nqubits)
    )
    return h


def _gpp_dense(
    adjacency_matrix: ArrayLike,
    penalty_coeff: ArrayLike,
    node_weights: ArrayLike,
    backend: Backend,
) -> Hamiltonian:
    def term(nqubits, ind_1, ind_2=None):
        diag = [id_diag] * nqubits
        diag[ind_1] = term_diag
        if ind_2 is not None:
            diag[ind_2] = term_diag
        diag = _multikron(diag, backend)

        return diag

    nqubits = len(adjacency_matrix)

    id_diag = backend.cast([1, 1])
    term_diag = backend.cast([0, 1])

    hamiltonian = 0
    rows, columns = backend.nonzero(backend.tril(adjacency_matrix, -1))
    for ind_j, ind_k in zip(columns, rows):
        ind_j, ind_k = int(ind_j), int(ind_k)

        diag = term(nqubits, ind_j)
        diag += term(nqubits, ind_k)
        diag -= 2 * term(nqubits, ind_j, ind_k)

        hamiltonian += diag

    if penalty_coeff != 0.0:
        penalty = 0
        for elem, weight in enumerate(node_weights):
            penalty += weight * (term(nqubits, elem) - 1 / 2)

        hamiltonian += penalty_coeff * (penalty**2)

    return Hamiltonian(nqubits, backend.diag(hamiltonian), backend=backend)


def _gpp_symbolic(
    adjacency_matrix: ArrayLike,
    penalty_coeff: float,
    node_weights: ArrayLike,
    backend: Backend,
) -> SymbolicHamiltonian:
    def term(index: int):
        return (
            symbols.I(index, backend=backend) - symbols.Z(index, backend=backend)
        ) / 2

    hamiltonian = 0
    rows, columns = backend.nonzero(backend.tril(adjacency_matrix, -1))
    for ind_j, ind_k in zip(columns, rows):
        ind_j, ind_k = int(ind_j), int(ind_k)
        x_j = term(ind_j)
        x_k = term(ind_k)
        hamiltonian += float(adjacency_matrix[ind_j, ind_k]) * (
            x_j + x_k - 2 * x_j * x_k
        )

    if penalty_coeff != 0.0:
        penalty = 0
        for elem, weight in enumerate(node_weights):
            penalty += float(weight) * (
                term(elem) - symbols.I(elem, backend=backend) / 2
            )

        hamiltonian += penalty_coeff * (penalty**2)

    return SymbolicHamiltonian(hamiltonian, backend=backend)


def _graph_bonds(graph: nx.Graph) -> tuple[int, dict[tuple[int, int], float]]:
    """Helper method for extracting the bonds of a connectivity graph.

    Integer nodes from :math:`0` to :math:`n - 1` are their own labels. Otherwise, nodes are
    labeled by the order in which they are stored in the graph. Edges between the same pair of
    nodes (*e.g.*, antiparallel edges of a directed graph) are merged into a single bond,
    and they must have the same weight.

    Args:
        graph (:class:`networkx.Graph`): connectivity graph, with an optional ``"weight"``
            attribute per edge. Edges without this attribute have weight :math:`1`.

    Returns:
        tuple: number of nodes :math:`n`, and dictionary that maps every bond :math:`(j, \\, k)`,
        with :math:`j < k` the labels of the connected nodes, to its weight. The bonds are sorted.
    """
    if graph.number_of_nodes() < 1:
        raise_error(ValueError, "The graph must have at least one node.")

    if nx.number_of_selfloops(graph) > 0:
        raise_error(ValueError, "The graph must not have self-loops.")

    nodes = list(graph.nodes)
    # integer nodes from 0 to n - 1 are used as labels, otherwise nodes are
    # labeled by their order in the graph
    if set(nodes) == set(range(len(nodes))):
        labels = {node: int(node) for node in nodes}
    else:
        labels = {node: label for label, node in enumerate(nodes)}

    # edges between the same pair of nodes (e.g. antiparallel edges of a directed graph)
    # are merged into a single bond, and they must have the same weight
    weights = {}
    for node_a, node_b, weight in graph.edges(data="weight", default=1):
        bond = tuple(sorted((labels[node_a], labels[node_b])))
        if bond in weights and weights[bond] != weight:
            raise_error(
                ValueError,
                f"Edges between nodes {node_a} and {node_b} have different weights.",
            )
        weights[bond] = weight

    return len(nodes), dict(sorted(weights.items()))


def _multikron(matrix_list: list[ArrayLike], backend: Backend) -> ArrayLike:
    """Calculates Kronecker product of a list of matrices.

    Args:
        matrix_list (list): List of matrices as ``ndarray``.

    Returns:
        ndarray: Kronecker product of all matrices in ``matrix_list``.
    """
    return reduce(backend.kron, matrix_list)


def _OneBodyPauli(
    nqubits: int,
    operator: Callable,
    dense: bool = True,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """Helper method for constructing non-interacting
    :math:`X`, :math:`Y`, and :math:`Z` Hamiltonians."""
    backend = _check_backend(backend)
    if dense:
        condition = lambda i, j: i == j % nqubits
        ham = -_build_spin_model(
            nqubits, operator(0, backend=backend).matrix, condition, backend
        )
        return Hamiltonian(nqubits, ham, backend=backend)

    form = sum([-1 * operator(qubit, backend=backend) for qubit in range(nqubits)])
    ham = SymbolicHamiltonian(form=form, backend=backend)
    return ham
