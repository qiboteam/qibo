"""Tests methods from `qibo/src/hamiltonians/models.py`."""

import math
from functools import reduce
from itertools import combinations, product

import networkx as nx
import numpy as np
import pytest

from qibo import hamiltonians, matrices, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.hamiltonians._models import _multikron
from qibo.hamiltonians.models import (
    ANNNI,
    GPP,
    LABS,
    TFIM,
    XXX,
    ClusterIsing,
    FermiHubbard,
    FoldedXXZ,
    Heisenberg,
    Ising,
    MaxCut,
    SpinGlass,
)

models_config = [
    ("X", {"nqubits": 3}, "x_N3.out"),
    ("Y", {"nqubits": 4}, "y_N4.out"),
    ("Z", {"nqubits": 5}, "z_N5.out"),
    ("TFIM", {"nqubits": 3, "h": 0.0}, "tfim_N3h0.0.out"),
    ("TFIM", {"nqubits": 3, "h": 0.5}, "tfim_N3h0.5.out"),
    ("TFIM", {"nqubits": 3, "h": 1.0}, "tfim_N3h1.0.out"),
    ("MaxCut", {"nqubits": 3}, "maxcut_N3.out"),
    ("MaxCut", {"nqubits": 4}, "maxcut_N4.out"),
    ("MaxCut", {"nqubits": 5}, "maxcut_N5.out"),
    ("XXZ", {"nqubits": 3, "delta": 0.0}, "heisenberg_N3delta0.0.out"),
    ("XXZ", {"nqubits": 3, "delta": 0.5}, "heisenberg_N3delta0.5.out"),
    ("XXZ", {"nqubits": 3, "delta": 1.0}, "heisenberg_N3delta1.0.out"),
]


@pytest.mark.parametrize(("model", "kwargs", "filename"), models_config)
def test_hamiltonian_models(backend, model, kwargs, filename):
    """Test pre-coded Hamiltonian models generate the proper matrices."""
    from .test_models_variational import assert_regression_fixture

    H = getattr(hamiltonians, model)(**kwargs, backend=backend)
    matrix = backend.to_numpy(H.matrix).flatten().real
    assert_regression_fixture(backend, matrix, filename)


@pytest.mark.parametrize("nqubits,adj_matrix", zip([3, 4], [None, [[0, 1], [2, 3]]]))
@pytest.mark.parametrize("dense", [True, False])
def test_maxcut(backend, nqubits, adj_matrix, dense):
    if adj_matrix is not None:
        with pytest.raises(RuntimeError):
            final_ham = MaxCut(nqubits, dense, adj_matrix=adj_matrix, backend=backend)
    else:
        size = 2**nqubits
        ham = np.zeros(shape=(size, size), dtype=np.complex128)
        for i in range(nqubits):
            for j in range(nqubits):
                h = np.eye(1)
                for k in range(nqubits):
                    if (k == i) ^ (k == j):
                        h = np.kron(h, matrices.Z)
                    else:
                        h = np.kron(h, matrices.I)
                M = np.eye(2**nqubits) - h
                ham += M
        target_ham = backend.cast(-ham / 2)
        final_ham = MaxCut(nqubits, dense, adj_matrix=adj_matrix, backend=backend)
        backend.assert_allclose(final_ham.matrix, target_ham)


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("nqubits", [3, 4])
def test_labs(backend, nqubits, dense):
    with pytest.raises(ValueError):
        hamiltonian = LABS(1, dense=dense, backend=backend)

    Z = lambda x: symbols.Z(x, backend=backend)

    if nqubits == 3:
        target = (Z(0) * Z(2)) ** 2 + (Z(0) * Z(1) + Z(1) * Z(2)) ** 2
    elif nqubits == 4:
        target = (
            (Z(0) * Z(3)) ** 2
            + (Z(0) * Z(2) + Z(1) * Z(3)) ** 2
            + (Z(0) * Z(1) + Z(1) * Z(2) + Z(2) * Z(3)) ** 2
        )

    target = SymbolicHamiltonian(target, nqubits, backend=backend)

    hamiltonian = LABS(nqubits, dense=dense, backend=backend)

    backend.assert_allclose(hamiltonian.matrix, target.matrix)


@pytest.mark.parametrize("model", ["XXZ", "TFIM"])
def test_missing_neighbour_qubit(backend, model):
    with pytest.raises(ValueError):
        getattr(hamiltonians, model)(nqubits=1, backend=backend)


@pytest.mark.parametrize("dense", [True, False])
def test_xxx(backend, dense):
    nqubits = 2

    with pytest.raises(ValueError):
        XXX(
            nqubits,
            coupling_constant=1,
            external_field_strengths=[0, 1],
            dense=dense,
            backend=backend,
        )

    with pytest.raises(TypeError):
        XXX(nqubits, coupling_constant=[1], dense=dense, backend=backend)

    with pytest.raises(ValueError):
        Heisenberg(
            nqubits,
            coupling_constants=[0, 1],
            external_field_strengths=1,
            dense=dense,
            backend=backend,
        )


@pytest.mark.parametrize("node_weights", [False, True])
@pytest.mark.parametrize("is_list", [False, True])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("penalty_coeff", [0.0, 2])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_gpp(backend, nqubits, penalty_coeff, dense, is_list, node_weights):
    with pytest.raises(ValueError):
        GPP(np.random.rand(3, 3), penalty_coeff, np.random.rand(4), backend=backend)

    with pytest.raises(ValueError):
        GPP(np.random.rand(3, 2), penalty_coeff, np.random.rand(3), backend=backend)

    adj_matrix = np.ones((nqubits, nqubits)) - np.diag(np.ones(nqubits))
    adj_matrix = (
        list(adj_matrix) if is_list else backend.cast(adj_matrix, dtype=np.int8)
    )

    node_weights = [1] * nqubits if node_weights else None

    hamiltonian = GPP(
        adj_matrix, penalty_coeff, node_weights, dense=dense, backend=backend
    )

    term = (backend.matrices.I() - backend.matrices.Z) / 2
    base_string = [backend.matrices.I()] * nqubits
    rows, columns = backend.nonzero(backend.tril(adj_matrix, -1))
    target = 0
    for col, row in zip(columns, rows):
        term_col = base_string.copy()
        term_col[int(col)] = term
        term_col = reduce(backend.kron, term_col)

        term_row = base_string.copy()
        term_row[int(row)] = term
        term_row = reduce(backend.kron, term_row)

        target += term_row + term_col - 2 * (term_col @ term_row)

    if penalty_coeff != 0.0:
        penalty = 0
        for elem in range(len(adj_matrix)):
            term_weight = base_string.copy()
            term_weight[elem] = term - backend.matrices.I() / 2
            term_weight = reduce(backend.kron, term_weight)
            penalty += term_weight

        target += penalty_coeff * (penalty**2)

    backend.assert_allclose(hamiltonian.matrix, target)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("h", [0.0, 0.5])
def test_tfim_boundary(backend, h, closed_boundary, dense):
    nqubits = 3

    lambda x: symbols.I(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)
    Z = lambda x: symbols.Z(x, backend=backend)

    target = Z(0) * Z(1) + Z(1) * Z(2)
    if closed_boundary:
        target += Z(2) * Z(0)
    if h != 0.0:
        for qubit in range(nqubits):
            target += h * X(qubit)

    target *= -1
    target = SymbolicHamiltonian(target, nqubits=nqubits, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = TFIM(
        nqubits, h=h, closed_boundary=closed_boundary, dense=dense, backend=backend
    )
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("nsites", [2, 3])
def test_fermi_hubbard(backend, nsites, dense, closed_boundary):
    t, U = -1.5, 3 / 2

    I, X, Y, Z = (
        backend.matrices.I(),
        backend.matrices.X,
        backend.matrices.Y,
        backend.matrices.Z,
    )

    target = 0
    if nsites == 2:
        target += (t / 2) * (
            _multikron([X, Z, X, I], backend=backend)
            + _multikron([Y, Z, Y, I], backend=backend)
            + _multikron([I, X, Z, X], backend=backend)
            + _multikron([I, Y, Z, Y], backend=backend)
        )
        target += (U / 4) * (
            nsites * backend.identity(4**nsites)
            - _multikron([Z, I, I, I], backend=backend)
            - _multikron([I, Z, I, I], backend=backend)
            - _multikron([I, I, Z, I], backend=backend)
            - _multikron([I, I, I, Z], backend=backend)
            + _multikron([Z, Z, I, I], backend=backend)
            + _multikron([I, I, Z, Z], backend=backend)
        )
    else:
        target = (t / 2) * (
            _multikron([X, Z, X, I, I, I], backend=backend)
            + _multikron([Y, Z, Y, I, I, I], backend=backend)
            + _multikron([I, X, Z, X, I, I], backend=backend)
            + _multikron([I, Y, Z, Y, I, I], backend=backend)
            + _multikron([I, I, X, Z, X, I], backend=backend)
            + _multikron([I, I, Y, Z, Y, I], backend=backend)
            + _multikron([I, I, I, X, Z, X], backend=backend)
            + _multikron([I, I, I, Y, Z, Y], backend=backend)
        )
        if closed_boundary:
            target += (t / 2) * (
                _multikron([X, Z, Z, Z, X, I], backend=backend)
                + _multikron([Y, Z, Z, Z, Y, I], backend=backend)
                + _multikron([I, X, Z, Z, Z, X], backend=backend)
                + _multikron([I, Y, Z, Z, Z, Y], backend=backend)
            )
        target += (U / 4) * (
            nsites * backend.identity(4**nsites)
            - _multikron([Z, I, I, I, I, I], backend=backend)
            - _multikron([I, Z, I, I, I, I], backend=backend)
            - _multikron([I, I, Z, I, I, I], backend=backend)
            - _multikron([I, I, I, Z, I, I], backend=backend)
            - _multikron([I, I, I, I, Z, I], backend=backend)
            - _multikron([I, I, I, I, I, Z], backend=backend)
            + _multikron([Z, Z, I, I, I, I], backend=backend)
            + _multikron([I, I, Z, Z, I, I], backend=backend)
            + _multikron([I, I, I, I, Z, Z], backend=backend)
        )

    hamiltonian = FermiHubbard(
        nsites, t, U, dense=dense, closed_boundary=closed_boundary, backend=backend
    )

    backend.assert_allclose(hamiltonian.matrix, target)


@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("lattice", [(1, 3), (2, 2), (3, 1)])
def test_fermi_hubbard_2d_dense_symbolic(backend, lattice, closed_boundary):
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "closed_boundary": closed_boundary,
        "backend": backend,
    }
    dense = FermiHubbard(lattice, dense=True, **kwargs)
    symbolic = FermiHubbard(lattice, dense=False, **kwargs)

    assert dense.nqubits == 2 * lattice[0] * lattice[1]
    backend.assert_allclose(dense.matrix, symbolic.matrix, atol=1e-10)


@pytest.mark.parametrize("labelling", ["row_major", "snake"])
@pytest.mark.parametrize("lattice", [(1, 3), (2, 2), (3, 1)])
def test_fermi_hubbard_2d_free_fermions(backend, lattice, labelling):
    """At :math:`U = 0`, the ground-state energy of the open lattice must be equal to
    the sum, over both spin species, of the negative single-particle energies."""
    nrows, ncols = lattice
    t = -1.3

    labels = [
        [
            row * ncols
            + (col if labelling == "row_major" or row % 2 == 0 else ncols - 1 - col)
            for col in range(ncols)
        ]
        for row in range(nrows)
    ]
    hopping = backend.zeros((nrows * ncols, nrows * ncols), dtype=backend.float64)
    for row in range(nrows):
        for col in range(ncols):
            if col < ncols - 1:
                a, b = labels[row][col], labels[row][col + 1]
                hopping[a, b] = hopping[b, a] = t
            if row < nrows - 1:
                a, b = labels[row][col], labels[row + 1][col]
                hopping[a, b] = hopping[b, a] = t
    levels = backend.eigvalsh(hopping)
    target = 2 * backend.sum(levels[levels < 0])

    hamiltonian = FermiHubbard(
        lattice,
        hopping_strength=t,
        interaction_strength=0.0,
        closed_boundary=False,
        labelling=labelling,
        backend=backend,
    )
    energies = hamiltonian.eigenvalues()

    backend.assert_allclose(backend.min(energies), target, atol=1e-8)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "labelling, pairs",
    [
        # bonds 0-1, 2-3, 0-2, and 1-3
        (
            "row_major",
            [(0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)],
        ),
        # bonds 0-1, 2-3, 0-3, and 1-2
        (
            "snake",
            [(0, 2), (1, 3), (4, 6), (5, 7), (0, 6), (1, 7), (2, 4), (3, 5)],
        ),
    ],
)
def test_fermi_hubbard_2d_labelling(backend, labelling, pairs, dense):
    """Qubit pairs connected by hopping in a :math:`2 \\times 2` lattice, for which
    sites are labeled as ``[[0, 1], [2, 3]]`` (row major) or ``[[0, 1], [3, 2]]`` (snake).
    Even (odd) qubits encode spin up (down)."""
    t = -1.5
    I, X, Y, Z = (
        backend.matrices.I(),
        backend.matrices.X,
        backend.matrices.Y,
        backend.matrices.Z,
    )

    target = 0
    for qubit_a, qubit_b in pairs:
        for pauli in (X, Y):
            base_string = [I] * 8
            base_string[qubit_a] = pauli
            base_string[qubit_b] = pauli
            for qubit in range(qubit_a + 1, qubit_b):
                base_string[qubit] = Z
            target += (t / 2) * _multikron(base_string, backend=backend)

    hamiltonian = FermiHubbard(
        (2, 2),
        hopping_strength=t,
        interaction_strength=0.0,
        dense=dense,
        closed_boundary=False,
        labelling=labelling,
        backend=backend,
    )

    backend.assert_allclose(hamiltonian.matrix, target, atol=1e-10)


def test_fermi_hubbard_2d_labelling_spectrum(backend):
    """Relabeling sites permutes fermionic modes, and it must not change the spectrum."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "closed_boundary": False,
        "backend": backend,
    }
    row_major = FermiHubbard((2, 2), labelling="row_major", **kwargs)
    snake = FermiHubbard((2, 2), labelling="snake", **kwargs)

    backend.assert_allclose(snake.eigenvalues(), row_major.eigenvalues(), atol=1e-10)


@pytest.mark.parametrize("labelling", ["row_major", "snake"])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("nsites", [2, 3, 4])
def test_fermi_hubbard_chain_lattice(
    backend, nsites, closed_boundary, dense, labelling
):
    """A chain with :math:`n` sites and a :math:`(1, n)` lattice are the same model."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "closed_boundary": closed_boundary,
        "dense": dense,
        "labelling": labelling,
        "backend": backend,
    }
    chain = FermiHubbard(nsites, **kwargs)
    lattice = FermiHubbard((1, nsites), **kwargs)

    backend.assert_allclose(lattice.matrix, chain.matrix)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "graph, lattice, closed_boundary",
    [
        (nx.path_graph(3), 3, False),
        (nx.cycle_graph(3), 3, True),
        (nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 2)), (2, 2), False),
    ],
)
def test_fermi_hubbard_graph_regular(backend, graph, lattice, closed_boundary, dense):
    """Graphs with chain and rectangular connectivity must reproduce the lattice models."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "dense": dense,
        "backend": backend,
    }
    target = FermiHubbard(
        lattice,
        closed_boundary=closed_boundary,
        labelling="row_major",
        **kwargs,
    )
    hamiltonian = FermiHubbard(graph, **kwargs)

    assert hamiltonian.nqubits == target.nqubits
    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize(
    "graph",
    [
        nx.complete_graph(3),
        nx.Graph([(0, 2), (1, 2)]),
        nx.Graph([("a", "c"), ("b", "c")]),
        nx.star_graph(3),
    ],
)
def test_fermi_hubbard_graph_dense_symbolic(backend, graph):
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "backend": backend,
    }
    dense = FermiHubbard(graph, dense=True, **kwargs)
    symbolic = FermiHubbard(graph, dense=False, **kwargs)

    assert dense.nqubits == 2 * graph.number_of_nodes()
    backend.assert_allclose(dense.matrix, symbolic.matrix, atol=1e-10)


@pytest.mark.parametrize("graph", [nx.complete_graph(3), nx.star_graph(3)])
def test_fermi_hubbard_graph_free_fermions(backend, graph):
    """At :math:`U = 0`, the ground-state energy must be equal to the sum, over both spin
    species, of the negative single-particle energies of the hopping matrix."""
    t = -1.3

    hopping = backend.cast(t * nx.to_numpy_array(graph), dtype=backend.float64)
    levels = backend.eigvalsh(hopping)
    target = 2 * backend.sum(levels[levels < 0])

    hamiltonian = FermiHubbard(
        graph, hopping_strength=t, interaction_strength=0.0, backend=backend
    )
    energies = hamiltonian.eigenvalues()

    backend.assert_allclose(backend.min(energies), target, atol=1e-8)


@pytest.mark.parametrize("dense", [False, True])
def test_fermi_hubbard_graph_integer_nodes(backend, dense):
    """Integer nodes from 0 to n - 1 are site labels, regardless of insertion order."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "dense": dense,
        "backend": backend,
    }
    target = FermiHubbard(nx.Graph([(0, 1), (0, 2)]), **kwargs)
    hamiltonian = FermiHubbard(nx.Graph([(2, 0), (0, 1)]), **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_fermi_hubbard_graph_other_nodes(backend, dense):
    """Nodes that are not integers from 0 to n - 1 are labeled by their order in the graph."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "closed_boundary": False,
        "dense": dense,
        "backend": backend,
    }
    target = FermiHubbard(3, **kwargs)
    hamiltonian = FermiHubbard(nx.Graph([("a", "b"), ("b", "c")]), **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_fermi_hubbard_graph_isolated_node(backend, dense):
    """Isolated nodes are sites without hopping."""
    graph = nx.Graph([(0, 1)])
    graph.add_node(2)
    kwargs = {"hopping_strength": -1.5, "dense": dense, "backend": backend}

    free = FermiHubbard(graph, interaction_strength=0.0, **kwargs)
    target = _multikron(
        [
            FermiHubbard(2, interaction_strength=0.0, **kwargs).matrix,
            backend.identity(4),
        ],
        backend=backend,
    )
    backend.assert_allclose(free.matrix, target, atol=1e-10)

    interacting = FermiHubbard(graph, interaction_strength=0.75, **kwargs)
    interaction = FermiHubbard(nx.empty_graph(3), interaction_strength=0.75, **kwargs)
    backend.assert_allclose(
        interacting.matrix - free.matrix, interaction.matrix, atol=1e-10
    )


@pytest.mark.parametrize("dense", [False, True])
def test_fermi_hubbard_graph_weights_linear(backend, dense):
    """At :math:`U = 0`, the Hamiltonian is linear in the edge weights."""
    weights = {(0, 1): 0.5, (1, 2): -2.0, (0, 2): 1.5}
    kwargs = {
        "hopping_strength": -1.3,
        "interaction_strength": 0.0,
        "dense": dense,
        "backend": backend,
    }

    graph = nx.Graph()
    graph.add_weighted_edges_from([(*edge, weight) for edge, weight in weights.items()])
    hamiltonian = FermiHubbard(graph, **kwargs)

    target = 0
    for edge, weight in weights.items():
        single = nx.empty_graph(3)
        single.add_edge(*edge)
        target = target + weight * FermiHubbard(single, **kwargs).matrix

    backend.assert_allclose(hamiltonian.matrix, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_fermi_hubbard_graph_weights_uniform(backend, dense):
    """A uniform weight :math:`w` is equivalent to rescaling the hopping strength by :math:`w`."""
    kwargs = {"interaction_strength": 0.75, "dense": dense, "backend": backend}

    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, 2.5, "weight")
    hamiltonian = FermiHubbard(graph, hopping_strength=-1.5, **kwargs)
    target = FermiHubbard(nx.complete_graph(3), hopping_strength=-3.75, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_fermi_hubbard_graph_weights_default(backend, dense):
    """Edges without a ``"weight"`` attribute have weight one, and other attributes are ignored."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "dense": dense,
        "backend": backend,
    }
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, cost=10.0)

    target = FermiHubbard(nx.path_graph(3), **kwargs)
    hamiltonian = FermiHubbard(graph, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_fermi_hubbard_graph_weights_zero(backend, dense):
    """An edge with zero weight is equivalent to a missing edge."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "dense": dense,
        "backend": backend,
    }
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 0.0)])

    target = nx.empty_graph(3)
    target.add_edge(0, 1)

    hamiltonian = FermiHubbard(graph, **kwargs)
    target = FermiHubbard(target, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("graph", [nx.complete_graph(3), nx.star_graph(3)])
def test_fermi_hubbard_graph_weights_free_fermions(backend, graph):
    """At :math:`U = 0`, the ground-state energy must be equal to the sum, over both spin
    species, of the negative single-particle energies of the weighted hopping matrix."""
    t = -1.3
    for index, (node_a, node_b) in enumerate(graph.edges):
        graph[node_a][node_b]["weight"] = 0.5 + index

    hopping = backend.cast(
        t * nx.to_numpy_array(graph, weight="weight"), dtype=backend.float64
    )
    levels = backend.eigvalsh(hopping)
    target = 2 * backend.sum(levels[levels < 0])

    hamiltonian = FermiHubbard(
        graph, hopping_strength=t, interaction_strength=0.0, backend=backend
    )
    energies = hamiltonian.eigenvalues()

    backend.assert_allclose(backend.min(energies), target, atol=1e-8)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_fermi_hubbard_graph_weights_edge_multiplicity(backend, graph_type):
    """Edges between the same pair of nodes are merged if their weights are equal."""
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "backend": backend,
    }

    graph = graph_type()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 0, 2.0), (1, 2, 0.5), (1, 2, 0.5)])

    target = nx.Graph()
    target.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 0.5)])

    hamiltonian = FermiHubbard(graph, **kwargs)
    target = FermiHubbard(target, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_fermi_hubbard_graph_weights_value_error(backend, graph_type):
    graph = graph_type()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 0, 2.0)])

    with pytest.raises(ValueError):
        FermiHubbard(graph, backend=backend)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_fermi_hubbard_graph_edge_multiplicity(backend, graph_type):
    """Edge directions and multiplicities are ignored."""
    edges = [(0, 1), (1, 0), (1, 2), (1, 2)]
    kwargs = {
        "hopping_strength": -1.5,
        "interaction_strength": 0.75,
        "backend": backend,
    }

    target = FermiHubbard(nx.Graph(edges), **kwargs)
    hamiltonian = FermiHubbard(graph_type(edges), **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


def test_fermi_hubbard_graph_ignored_arguments(backend):
    """``closed_boundary`` and ``labelling`` do not change a model defined by a graph."""
    graph = nx.cycle_graph(3)
    target = FermiHubbard(graph, backend=backend)
    hamiltonian = FermiHubbard(
        graph, closed_boundary=False, labelling="row_major", backend=backend
    )

    backend.assert_allclose(hamiltonian.matrix, target.matrix)


def test_fermi_hubbard_graph_value_error(backend):
    with pytest.raises(ValueError):
        FermiHubbard(nx.Graph(), backend=backend)

    with pytest.raises(ValueError):
        FermiHubbard(nx.Graph([(0, 0), (0, 1)]), backend=backend)


@pytest.mark.parametrize("nsites", [(2,), (2, 2, 2), "2", 2.0])
def test_fermi_hubbard_2d_type_error(backend, nsites):
    with pytest.raises(TypeError):
        FermiHubbard(nsites, backend=backend)


@pytest.mark.parametrize("nsites", [(0, 2), (2, 0), (-1, 2)])
def test_fermi_hubbard_2d_value_error(backend, nsites):
    with pytest.raises(ValueError):
        FermiHubbard(nsites, backend=backend)


@pytest.mark.parametrize("labelling", ["column_major", "Snake", 1])
def test_fermi_hubbard_2d_labelling_error(backend, labelling):
    with pytest.raises(ValueError):
        FermiHubbard((2, 2), labelling=labelling, backend=backend)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("local_field_strengths", ["number", "tuple", "array"])
@pytest.mark.parametrize("coupling_constants", ["number", "tuple", "array"])
def test_ising(
    backend, coupling_constants, local_field_strengths, closed_boundary, dense
):
    with pytest.raises(ValueError):
        Ising(
            3,
            (0.1, 0.2),
            local_field_strengths=1.0,
            closed_boundary=True,
            backend=backend,
        )

    lambda x: symbols.I(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)
    Z = lambda x: symbols.Z(x, backend=backend)

    nqubits = 3

    if coupling_constants == "number":
        coupling_constants = 1.0
    elif coupling_constants == "tuple":
        coupling_constants = [1.0, 2.0]
        if closed_boundary:
            coupling_constants.append(1.0)
    else:
        coupling_constants = [1.0, 2.0]
        if closed_boundary:
            coupling_constants.append(1.0)
        coupling_constants = backend.cast(coupling_constants, dtype=backend.float64)

    if local_field_strengths == "number":
        local_field_strengths = 1.0
    elif local_field_strengths == "tuple":
        local_field_strengths = (1.0, 0.5)
        if closed_boundary:
            local_field_strengths += (1.0,)
    else:
        local_field_strengths = [[1.0, 0.5], [0.1, 2.0]]
        if closed_boundary:
            local_field_strengths.append([1.0, 0.5])
        local_field_strengths = backend.cast(
            local_field_strengths, dtype=backend.float64
        )

    if isinstance(coupling_constants, (int, float)):
        _coupling_constants = [coupling_constants] * 2
        if closed_boundary:
            _coupling_constants.append(_coupling_constants[-1])
    else:
        _coupling_constants = coupling_constants.copy()

    target = float(_coupling_constants[0]) * Z(0) * Z(1)
    target += float(_coupling_constants[1]) * Z(1) * Z(2)
    if closed_boundary:
        target += float(_coupling_constants[2]) * Z(2) * Z(0)

    if isinstance(local_field_strengths, float):
        target += local_field_strengths * (Z(0) + Z(1) + Z(2))
        target += local_field_strengths * (X(0) + X(1) + X(2))
    elif isinstance(local_field_strengths, (tuple, list)):
        target += float(local_field_strengths[0]) * (Z(0) + Z(1) + Z(2))
        target += float(local_field_strengths[1]) * (X(0) + X(1) + X(2))
    else:
        target += sum(
            float(strength) * Z(qubit)
            for qubit, strength in zip([0, 1, 2], local_field_strengths[:, 0])
        )
        target += sum(
            float(strength) * X(qubit)
            for qubit, strength in zip([0, 1, 2], local_field_strengths[:, 1])
        )
    target = SymbolicHamiltonian(target, nqubits=nqubits, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = Ising(
        nqubits=nqubits,
        coupling_constants=coupling_constants,
        local_field_strengths=local_field_strengths,
        dense=dense,
        closed_boundary=closed_boundary,
        backend=backend,
    )
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("nqubits", [3, 4])
def test_ising_graph_chain(backend, nqubits, closed_boundary, dense):
    """Path and cycle graphs are the open and closed chains, and ``closed_boundary`` is ignored."""
    graph = nx.cycle_graph(nqubits) if closed_boundary else nx.path_graph(nqubits)
    kwargs = {"local_field_strengths": (0.4, -0.7), "dense": dense, "backend": backend}

    target = Ising(nqubits, -1.3, closed_boundary=closed_boundary, **kwargs)
    hamiltonian = Ising(graph, -1.3, closed_boundary=not closed_boundary, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_ising_graph_weighted(backend, dense):
    """Bonds beyond nearest neighbors, edge weights and local fields per qubit."""
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    nqubits = 4
    coupling = 0.8
    fields = [[0.1, 0.2], [0.3, -0.4], [0.5, 0.6], [-0.7, 0.8]]

    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (0, 2, -0.5), (1, 3, 1.5)])
    # edges without a weight have weight 1
    graph.add_edge(2, 3)

    target = Z(0) * Z(1) * 2.0 + Z(0) * Z(2) * (-0.5) + Z(1) * Z(3) * 1.5
    target += Z(2) * Z(3)
    target *= coupling
    for qubit, (field_z, field_x) in enumerate(fields):
        target += field_z * Z(qubit) + field_x * X(qubit)
    target = SymbolicHamiltonian(target, nqubits=nqubits, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = Ising(
        graph,
        coupling_constants=coupling,
        local_field_strengths=backend.cast(fields, dtype=backend.float64),
        dense=dense,
        backend=backend,
    )
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_ising_graph_next_nearest_neighbors(backend, dense):
    """Ising model with first and second neighbors on an open chain."""
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    nqubits = 4
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(qubit, qubit + 1, -1.0) for qubit in range(nqubits - 1)]
        + [(qubit, qubit + 2, 0.5) for qubit in range(nqubits - 2)]
    )

    target = -(Z(0) * Z(1) + Z(1) * Z(2) + Z(2) * Z(3))
    target += 0.5 * (Z(0) * Z(2) + Z(1) * Z(3))
    target += -0.3 * sum(X(qubit) for qubit in range(nqubits))
    target = SymbolicHamiltonian(target, nqubits=nqubits, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = Ising(graph, 1.0, (0.0, -0.3), dense=dense, backend=backend)
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_ising_graph_integer_nodes(backend, dense):
    """Integer nodes from 0 to n - 1 are qubit labels, regardless of insertion order."""
    kwargs = {"local_field_strengths": (0.2, 0.4), "dense": dense, "backend": backend}
    target = Ising(nx.Graph([(0, 1), (0, 2)]), 1.5, **kwargs)
    hamiltonian = Ising(nx.Graph([(2, 0), (0, 1)]), 1.5, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_ising_graph_other_nodes(backend, dense):
    """Nodes that are not integers from 0 to n - 1 are labeled by their order in the graph."""
    kwargs = {"local_field_strengths": (0.2, 0.4), "dense": dense, "backend": backend}
    target = Ising(3, 1.5, closed_boundary=False, **kwargs)
    hamiltonian = Ising(nx.Graph([("a", "b"), ("b", "c")]), 1.5, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_ising_graph_isolated_node(backend, dense):
    """Isolated nodes are qubits with no coupling, but they feel the local fields."""
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    graph = nx.Graph([(0, 1)])
    graph.add_node(2)

    target = 1.5 * Z(0) * Z(1) + sum(0.2 * Z(q) + 0.4 * X(q) for q in range(3))
    target = SymbolicHamiltonian(target, nqubits=3, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = Ising(graph, 1.5, (0.2, 0.4), dense=dense, backend=backend)
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_ising_graph_no_bonds(backend, dense):
    """Graphs without edges only have local fields, and may even be the null Hamiltonian."""
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    graph = nx.empty_graph(3)

    target = sum(0.2 * Z(q) + 0.4 * X(q) for q in range(3))
    target = SymbolicHamiltonian(target, nqubits=3, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = Ising(graph, 1.5, (0.2, 0.4), dense=dense, backend=backend)
    backend.assert_allclose(backend.real(hamiltonian.matrix), target, atol=1e-10)

    hamiltonian = Ising(graph, 1.5, 0.0, dense=dense, backend=backend)
    backend.assert_allclose(hamiltonian.matrix, backend.zeros((8, 8)), atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_ising_graph_edge_multiplicity(backend, graph_type, dense):
    """Edges between the same pair of nodes are merged if their weights are equal."""
    kwargs = {"local_field_strengths": (0.2, 0.4), "dense": dense, "backend": backend}

    graph = graph_type()
    graph.add_weighted_edges_from([(0, 1, 2.0), (1, 0, 2.0), (1, 2, 0.5), (1, 2, 0.5)])

    target = nx.Graph()
    target.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 0.5)])

    hamiltonian = Ising(graph, -1.5, **kwargs)
    target = Ising(target, -1.5, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_ising_graph_value_error(backend, graph_type):
    graph = graph_type()
    graph.add_weighted_edges_from([(0, 1, 1.0), (1, 0, 2.0)])

    with pytest.raises(ValueError):
        Ising(graph, 1.0, 0.5, backend=backend)

    with pytest.raises(ValueError):
        Ising(nx.Graph(), 1.0, 0.5, backend=backend)

    with pytest.raises(ValueError):
        Ising(nx.Graph([(0, 0), (0, 1)]), 1.0, 0.5, backend=backend)


@pytest.mark.parametrize("coupling_constants", [[1.0, 1.0], (1.0, 1.0), np.ones(2)])
def test_ising_graph_type_error(backend, coupling_constants):
    """Per-bond couplings are given by the edge weights of the graph."""
    with pytest.raises(TypeError):
        Ising(nx.path_graph(3), coupling_constants, 0.5, backend=backend)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
def test_tfim_two_qubits(backend, closed_boundary, dense):
    """Two qubits share a single bond, regardless of the boundary condition."""
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    target = -Z(0) * Z(1) - 0.5 * (X(0) + X(1))
    target = SymbolicHamiltonian(target, nqubits=2, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = TFIM(
        2, h=0.5, closed_boundary=closed_boundary, dense=dense, backend=backend
    )
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("as_list", [False, True])
def test_maxcut_adjacency_matrix(backend, as_list, dense):
    """Asymmetric adjacency matrices with self-loops match the explicit definition."""
    Z = lambda x: symbols.Z(x, backend=backend)

    nqubits = 3
    adj_matrix = np.array([[1.0, 2.0, 0.0], [0.5, 0.0, 3.0], [1.5, -1.0, 2.0]])

    target = -sum(
        adj_matrix[j][k] * (1 - Z(j) * Z(k))
        for j in range(nqubits)
        for k in range(nqubits)
    )
    target = SymbolicHamiltonian(target / 2, nqubits=nqubits, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = MaxCut(
        nqubits,
        dense,
        adj_matrix=adj_matrix.tolist() if as_list else adj_matrix,
        backend=backend,
    )
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_maxcut_ising(backend, dense):
    """Max Cut is the Ising model on the weighted graph, up to a constant energy shift."""
    adj_matrix = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    graph = nx.from_numpy_array(adj_matrix)

    maxcut = MaxCut(3, dense, adj_matrix=adj_matrix, backend=backend)
    ising = Ising(graph, 1.0, 0.0, dense=dense, backend=backend)

    # sum of the weights of all the bonds
    shift = 5.0
    identity = backend.cast(np.eye(8), dtype=backend.complex128)
    backend.assert_allclose(maxcut.matrix, ising.matrix - shift * identity, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_maxcut_zero_adjacency_matrix(backend, dense):
    hamiltonian = MaxCut(3, dense, adj_matrix=np.zeros((3, 3)), backend=backend)

    backend.assert_allclose(hamiltonian.matrix, backend.zeros((8, 8)), atol=1e-10)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (2, 2, 2)])
def test_maxcut_adjacency_matrix_shape(backend, shape):
    with pytest.raises(RuntimeError):
        MaxCut(3, adj_matrix=np.ones(shape), backend=backend)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("nqubits", [3, 4, 5, 6])
def test_annni(backend, nqubits, closed_boundary, dense):
    """Pairs one and two sites apart, coupled at most once even in small periodic rings."""
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    j1, j2, h = 1.2, 0.7, 0.4

    if closed_boundary:
        first = [(qubit, (qubit + 1) % nqubits) for qubit in range(nqubits)]
        second = {
            tuple(sorted((qubit, (qubit + 2) % nqubits))) for qubit in range(nqubits)
        }
        second -= {tuple(sorted(pair)) for pair in first}
    else:
        first = [(qubit, qubit + 1) for qubit in range(nqubits - 1)]
        second = [(qubit, qubit + 2) for qubit in range(nqubits - 2)]

    target = -j1 * sum(Z(a) * Z(b) for a, b in first)
    target += j2 * sum(Z(a) * Z(b) for a, b in second)
    target -= h * sum(X(qubit) for qubit in range(nqubits))
    target = SymbolicHamiltonian(target, nqubits=nqubits, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = ANNNI(
        nqubits, j1, j2, h, dense, closed_boundary=closed_boundary, backend=backend
    )
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
def test_annni_tfim(backend, closed_boundary, dense):
    """For :math:`J_{1} = 1` and :math:`J_{2} = 0`, the ANNNI model is the TFIM."""
    kwargs = {"dense": dense, "closed_boundary": closed_boundary, "backend": backend}
    hamiltonian = ANNNI(5, 1.0, 0.0, 0.3, **kwargs)
    target = TFIM(5, 0.3, **kwargs)

    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("nqubits", [0, 1, 2])
def test_annni_value_error(backend, nqubits):
    with pytest.raises(ValueError):
        ANNNI(nqubits, backend=backend)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("nqubits", [3, 4, 5])
def test_cluster_ising(backend, nqubits, closed_boundary, dense):
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    cluster, ising, field = 0.9, 0.5, 0.3

    centers = range(nqubits) if closed_boundary else range(1, nqubits - 1)
    bonds = [
        (q, (q + 1) % nqubits)
        for q in range(nqubits if closed_boundary else nqubits - 1)
    ]

    target = -cluster * sum(
        Z((q - 1) % nqubits) * X(q) * Z((q + 1) % nqubits) for q in centers
    )
    target -= ising * sum(Z(a) * Z(b) for a, b in bonds)
    target -= field * sum(X(q) for q in range(nqubits))
    target = SymbolicHamiltonian(target, nqubits=nqubits, backend=backend)
    target = backend.real(target.matrix)

    hamiltonian = ClusterIsing(
        nqubits,
        cluster,
        ising,
        field,
        dense=dense,
        closed_boundary=closed_boundary,
        backend=backend,
    )
    hamiltonian = backend.real(hamiltonian.matrix)

    backend.assert_allclose(hamiltonian, target, atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("closed_boundary", [False, True])
def test_cluster_ising_ising(backend, closed_boundary, dense):
    """Without cluster interaction, the model is Ising, and TFIM for unit Ising strength."""
    kwargs = {"dense": dense, "closed_boundary": closed_boundary, "backend": backend}

    hamiltonian = ClusterIsing(4, 0.0, 0.8, 0.3, **kwargs)
    target = Ising(4, -0.8, (0.0, -0.3), **kwargs)
    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)

    hamiltonian = ClusterIsing(4, 0.0, 1.0, 0.3, **kwargs)
    target = TFIM(4, 0.3, **kwargs)
    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


@pytest.mark.parametrize("closed_boundary", [False, True])
@pytest.mark.parametrize("nqubits", [4, 5])
def test_cluster_ising_cluster_state(backend, nqubits, closed_boundary):
    """All cluster stabilizers are satisfied in the ground state."""
    hamiltonian = ClusterIsing(
        nqubits, closed_boundary=closed_boundary, backend=backend
    )
    energy = hamiltonian.eigenvalues()[0]

    nstabilizers = nqubits if closed_boundary else nqubits - 2
    backend.assert_allclose(energy, -nstabilizers, atol=1e-10)


@pytest.mark.parametrize("nqubits", [0, 1, 2])
def test_cluster_ising_value_error(backend, nqubits):
    with pytest.raises(ValueError):
        ClusterIsing(nqubits, backend=backend)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("distribution", ["bimodal", "gaussian"])
def test_spin_glass_sk(backend, distribution, dense):
    """Infinite-range model, with couplings that are drawn bond by bond in lexicographic order."""
    nqubits, seed, strength = 4, 7, 2.0
    bonds = list(combinations(range(nqubits), 2))
    scale = strength / math.sqrt(nqubits)

    if distribution == "gaussian":
        target = backend.random_normal(0, 1, len(bonds), seed=seed)
    else:
        target = backend.random_choice([-1.0, 1.0], size=len(bonds), seed=seed)

    hamiltonian = SpinGlass(
        nqubits,
        strength,
        distribution=distribution,
        seed=seed,
        dense=dense,
        backend=backend,
    )

    # couplings are read from the diagonal, since the Hamiltonian is a sum of Z strings
    energies = backend.real(backend.diag(hamiltonian.matrix))
    spins = np.array(list(product([1.0, -1.0], repeat=nqubits)))
    couplings = (spins * energies[:, None]).T @ spins / 2**nqubits

    backend.assert_allclose(
        [couplings[a, b] for a, b in bonds], scale * target, atol=1e-10
    )


@pytest.mark.parametrize("dense", [False, True])
def test_spin_glass_ea(backend, dense):
    """Short-range model on a graph, with edge weights scaling the couplings."""
    seed, strength = 3, 1.5
    graph = nx.grid_2d_graph(2, 2)
    graph[(0, 0)][(0, 1)]["weight"] = 2.0

    # nodes are labeled as (0, 0) -> 0, (0, 1) -> 1, (1, 0) -> 2, (1, 1) -> 3
    bonds = [(0, 1), (0, 2), (1, 3), (2, 3)]
    weights = backend.cast([2.0, 1.0, 1.0, 1.0], dtype=backend.float64)
    target = backend.random_normal(0, 1, len(bonds), seed=seed)

    hamiltonian = SpinGlass(graph, strength, seed=seed, dense=dense, backend=backend)

    energies = backend.real(backend.diag(hamiltonian.matrix))
    spins = np.array(list(product([1.0, -1.0], repeat=4)))
    couplings = (spins * energies[:, None]).T @ spins / 2**4

    backend.assert_allclose(
        [couplings[a, b] for a, b in bonds], strength * weights * target, atol=1e-10
    )
    # qubits that are not connected by an edge are not coupled
    backend.assert_allclose([couplings[0, 3], couplings[1, 2]], [0.0, 0.0], atol=1e-10)


@pytest.mark.parametrize("dense", [False, True])
def test_spin_glass_seed(backend, dense):
    kwargs = {"dense": dense, "backend": backend}

    hamiltonian = SpinGlass(4, seed=1, **kwargs)
    backend.assert_allclose(hamiltonian.matrix, SpinGlass(4, seed=1, **kwargs).matrix)
    # a generator is equivalent to the seed that initializes it
    target = SpinGlass(4, seed=np.random.default_rng(1), **kwargs)
    backend.assert_allclose(hamiltonian.matrix, target.matrix)

    other = SpinGlass(4, seed=2, **kwargs)
    assert not backend.allclose(hamiltonian.matrix, other.matrix)


@pytest.mark.parametrize("dense", [False, True])
def test_spin_glass_local_fields(backend, dense):
    """Local fields are added to the random couplings, and a single qubit has no coupling."""
    Z = lambda x: symbols.Z(x, backend=backend)
    X = lambda x: symbols.X(x, backend=backend)

    fields = SymbolicHamiltonian(
        sum(0.2 * Z(q) + 0.4 * X(q) for q in range(3)), nqubits=3, backend=backend
    )
    couplings = SpinGlass(3, seed=1, dense=dense, backend=backend)
    hamiltonian = SpinGlass(3, 1.0, (0.2, 0.4), seed=1, dense=dense, backend=backend)
    backend.assert_allclose(
        hamiltonian.matrix, couplings.matrix + fields.matrix, atol=1e-10
    )

    hamiltonian = SpinGlass(1, 1.0, (0.2, 0.4), seed=1, dense=dense, backend=backend)
    target = SymbolicHamiltonian(0.2 * Z(0) + 0.4 * X(0), nqubits=1, backend=backend)
    backend.assert_allclose(hamiltonian.matrix, target.matrix, atol=1e-10)


def test_spin_glass_value_error(backend):
    with pytest.raises(ValueError):
        SpinGlass(3, distribution="uniform", backend=backend)

    with pytest.raises(ValueError):
        SpinGlass(0, backend=backend)

    with pytest.raises(ValueError):
        SpinGlass(nx.Graph(), backend=backend)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("nqubits", [4, 5])
def test_folded_xxz(backend, nqubits, dense):
    with pytest.raises(ValueError):
        FoldedXXZ(3, dense=dense, backend=backend)

    I, X = backend.matrices.I(), backend.matrices.X
    Y, Z = backend.matrices.Y, backend.matrices.Z

    if nqubits == 4:
        target = reduce(backend.kron, [I, X, X, I]) + reduce(backend.kron, [I, Y, Y, I])
        target += reduce(backend.kron, [Z, X, X, Z]) + reduce(
            backend.kron, [Z, Y, Y, Z]
        )

    if nqubits == 5:
        target = reduce(backend.kron, [I, X, X, I, I]) + reduce(
            backend.kron, [I, Y, Y, I, I]
        )
        target += reduce(backend.kron, [Z, X, X, Z, I]) + reduce(
            backend.kron, [Z, Y, Y, Z, I]
        )
        target += reduce(backend.kron, [I, I, X, X, I]) + reduce(
            backend.kron, [I, I, Y, Y, I]
        )
        target += reduce(backend.kron, [I, Z, X, X, Z]) + reduce(
            backend.kron, [I, Z, Y, Y, Z]
        )

    target *= -1 / 8

    hamiltonian = FoldedXXZ(nqubits, dense=dense, backend=backend)

    backend.assert_allclose(hamiltonian.matrix, target)
