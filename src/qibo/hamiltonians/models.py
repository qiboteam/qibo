from functools import reduce

import networkx as nx
import numpy as np
import sympy
from numpy.typing import ArrayLike

from qibo import symbols
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.hamiltonians._models import (
    _build_spin_model,
    _gpp_dense,
    _gpp_symbolic,
    _graph_bonds,
    _multikron,
    _OneBodyPauli,
)
from qibo.hamiltonians.hamiltonians import Hamiltonian, SymbolicHamiltonian


def FermiHubbard(
    nsites: int | tuple[int, int] | nx.Graph,
    hopping_strength: float = -1.0,
    interaction_strength: float = 0.5,
    dense: bool = True,
    closed_boundary: bool = True,
    labelling: str = "snake",
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """Jordan-Wigner-transformed Fermi-Hubbard model on a one-dimensional chain,
    on a two-dimensional rectangular lattice, or on a lattice with arbitrary connectivity.

    In second quantization, the Hamiltonian is defined as

    .. math::
        H_{\\textrm{FH}} = t \\, \\sum_{\\langle j, \\, k \\rangle, \\, \\sigma} \\, \\left(
            a_{j,\\sigma}^{\\dagger} \\, a_{k,\\sigma}
            + a_{k,\\sigma}^{\\dagger} \\, a_{j,\\sigma}
            \\right)
            + U \\, \\sum_{j} \\, n_{j,\\uparrow} \\, n_{j,\\downarrow} \\, ,

    where :math:`\\langle j, \\, k \\rangle` runs over all pairs of nearest-neighbor
    (*i.e.*, connected) lattice sites, :math:`a_{j,\\sigma}`
    (:math:`a_{j,\\sigma}^{\\dagger}`) is the annihilation (creation) operator for spin
    :math:`\\sigma \\in \\{\\uparrow, \\, \\downarrow \\}`
    at site :math:`j`, :math:`n_{j,\\sigma} = a_{j,\\sigma}^{\\dagger}\\,a_{j,\\sigma}` is the
    number operator for spin :math:`\\sigma` at site :math:`j`, :math:`t` is the tunneling
    amplitude, and :math:`U` is the Coulomb potential.

    For chains and rectangular lattices, sites are labeled according to ``labelling``. For a
    lattice with :math:`n_{c}` columns, site :math:`(r, \\, c)` has label :math:`j = r \\, n_{c} + c` in the
    ``"row_major"`` scheme, in which every row is traversed from left to right. In the
    ``"snake"`` scheme, the first row is traversed from left to right, the second row from
    right to left, and so on. Thus, the label is the same as in the ``"row_major"`` scheme if
    :math:`r` is even, and :math:`j = r \\, n_{c} + n_{c} - 1 - c` if :math:`r` is odd.
    Qubits :math:`2j` and :math:`2j + 1` encode the spin-up and spin-down modes of site
    :math:`j`, respectively. Both schemes are equivalent for chains and single-row lattices.

    For an arbitrary connectivity, ``nsites`` can be a :class:`networkx.Graph`, in which every
    node is a lattice site and every edge is a pair of connected sites. If the nodes of the graph
    are exactly the integers :math:`\\{0, \\, 1, \\, \\dots, \\, n - 1\\}`, every node is its
    own site label :math:`j`. Otherwise, sites are labeled by the order in which the nodes
    are stored in the graph, *i.e.*, the :math:`j`-th node of ``list(nsites.nodes)`` has label
    :math:`j`. The graph defines all the bonds of the model. Therefore, ``closed_boundary`` and
    ``labelling`` are ignored, and any periodicity must be encoded as edges of the graph.
    Isolated nodes are lattice sites with no hopping, and self-loops are not allowed.

    The graph can be weighted. The tunneling amplitude of the bond between sites :math:`j` and
    :math:`k` is :math:`t \\, w_{j,k}`, where :math:`w_{j,k}` is the ``"weight"`` attribute
    of the corresponding edge. Edges without this attribute have weight :math:`1`, and
    therefore an unweighted graph has a uniform tunneling amplitude :math:`t`. Edge directions
    are ignored. Edges between the same pair of nodes, *e.g.*, parallel edges of a multigraph,
    are merged into a single bond, and they must have the same weight.

    Every hopping term between qubits :math:`p < q` is multiplied by the Jordan-Wigner
    string :math:`\\prod_{p < m < q} Z_{m}`, so that fermionic anticommutation is
    correctly accounted for. A chain with :math:`n` sites, ``nsites=n``, is the
    same model as a lattice with a single row, ``nsites=(1, n)``.

    .. note::
        For periodic boundaries, bonds that wrap around a dimension are implemented
        as the same Jordan-Wigner-string hopping terms as the remaining bonds, without
        the correction that depends on the parity of the total number of particles.

    Args:
        nsites (int or tuple[int, int] or :class:`networkx.Graph`): if an ``int``, total number
            of sites in a one-dimensional chain. If a ``tuple`` :math:`(n_{r}, \\, n_{c})`,
            number of rows and columns of a two-dimensional lattice. If a
            :class:`networkx.Graph`, connectivity graph of the lattice, with one node per site
            and, optionally, a ``"weight"`` attribute per edge that scales ``hopping_strength``.
        hopping_strength (float or int, optional): tunneling amplitude. Defaults to :math:`-1.0`.
        interaction_strength (float or int, optional): Coulomb potential.
            Defaults to :math:`0.5`.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        closed_boundary (bool, optional): If ``True``, returns Fermi-Hubbard model with periodic
            boundary condition along every dimension with more than two sites. If ``False``,
            returns Hamiltonian with open boundaries. Ignored if ``nsites`` is a
            :class:`networkx.Graph`. Defaults to ``True``.
        labelling (str, optional): scheme used to label the lattice sites, either ``"snake"``
            or ``"row_major"``. Ignored if ``nsites`` is a :class:`networkx.Graph`.
            Defaults to ``"snake"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Fermi-Hubbard Hamiltonian :math:`H_{\\textrm{FH}}`.
    """
    backend = _check_backend(backend)

    if isinstance(nsites, nx.Graph):
        nnodes, weights = _graph_bonds(nsites)
        nqubits = 2 * nnodes
    else:
        if isinstance(nsites, (int, np.integer)):
            nrows, ncols = 1, int(nsites)
        elif isinstance(nsites, tuple) and len(nsites) == 2:
            nrows, ncols = nsites
        else:
            raise_error(
                TypeError,
                "``nsites`` must be either an ``int``, a ``tuple`` of two ``int``s, "
                + f"or a ``networkx.Graph``, but it is {nsites}.",
            )

        if nrows < 1 or ncols < 1:
            raise_error(
                ValueError,
                f"Lattice dimensions must be positive, but they are {(nrows, ncols)}.",
            )

        if labelling not in ("row_major", "snake"):
            raise_error(
                ValueError,
                f"``labelling`` must be either ``'row_major'`` or ``'snake'``, but it is {labelling}.",
            )

        nqubits = 2 * nrows * ncols

        # in the snake scheme, rows are traversed in alternating directions
        labels = [
            [
                row * ncols
                + (col if labelling == "row_major" or row % 2 == 0 else ncols - 1 - col)
                for col in range(ncols)
            ]
            for row in range(nrows)
        ]

        # nearest-neighbor bonds between sites
        bonds = [
            (labels[row][col], labels[row][col + 1])
            for row in range(nrows)
            for col in range(ncols - 1)
        ]
        bonds += [
            (labels[row][col], labels[row + 1][col])
            for row in range(nrows - 1)
            for col in range(ncols)
        ]
        if closed_boundary and ncols > 2:
            bonds += [(labels[row][0], labels[row][-1]) for row in range(nrows)]
        if closed_boundary and nrows > 2:
            bonds += [(labels[0][col], labels[-1][col]) for col in range(ncols)]

        weights = dict.fromkeys(bonds, 1)

    # pairs of qubits (spin-up with spin-up, spin-down with spin-down) connected by hopping,
    # ordered such that the first qubit has the smallest label, and the bond weights
    hoppings = [
        (2 * min(j, k) + spin, 2 * max(j, k) + spin, weight)
        for (j, k), weight in weights.items()
        for spin in (0, 1)
    ]

    if dense:
        I, X, Y, Z = (
            backend.matrices.I(),
            backend.matrices.X,
            backend.matrices.Y,
            backend.matrices.Z,
        )
        number_op = (I - Z) / 2

        hamiltonian = backend.zeros((2**nqubits, 2**nqubits), dtype=backend.complex128)

        # interaction terms
        base_string = [I] * nqubits
        for site in range(0, nqubits - 1, 2):
            base_string[site] = number_op
            base_string[site + 1] = number_op
            term = interaction_strength * _multikron(base_string, backend=backend)
            hamiltonian += term
            del term
            base_string[site] = I
            base_string[site + 1] = I

        # hopping terms
        for qubit_a, qubit_b, weight in hoppings:
            for pauli in (X, Y):
                base_string[qubit_a] = pauli
                base_string[qubit_b] = pauli
                for qubit in range(qubit_a + 1, qubit_b):
                    base_string[qubit] = Z
                term = (hopping_strength * weight / 2) * _multikron(
                    base_string, backend=backend
                )
                hamiltonian += term
                del term
                base_string[qubit_a : qubit_b + 1] = [I] * (qubit_b - qubit_a + 1)

        return Hamiltonian(nqubits, hamiltonian, backend=backend)

    I, X, Y, Z = (
        lambda q: symbols.I(q, backend=backend),
        lambda q: symbols.X(q, backend=backend),
        lambda q: symbols.Y(q, backend=backend),
        lambda q: symbols.Z(q, backend=backend),
    )
    number_op = lambda q: (I(q) - Z(q)) / 2

    # interaction terms
    hamiltonian = interaction_strength * sum(
        number_op(site) * number_op(site + 1) for site in range(0, nqubits - 1, 2)
    )

    # hopping terms
    for qubit_a, qubit_b, weight in hoppings:
        string = [Z(qubit) for qubit in range(qubit_a + 1, qubit_b)]
        hamiltonian += (hopping_strength * weight / 2) * sum(
            reduce(lambda x, y: x * y, [pauli(qubit_a), *string, pauli(qubit_b)])
            for pauli in (X, Y)
        )

    hamiltonian = SymbolicHamiltonian(
        form=hamiltonian, nqubits=nqubits, backend=backend
    )

    return hamiltonian


def FoldedXXZ(
    nqubits: int, dense: bool = True, backend: Backend | None = None
) -> Hamiltonian | SymbolicHamiltonian:
    """Folded XXZ model on an one-dimensional chain with open boundary conditions.

    This model is the :math:`\\Delta \\rightarrow \\infty` limit of the XXZ model.

    .. math::
        H_{\\Delta \\rightarrow \\infty} = -\\frac{1}{8} \\, \\sum_{j=0}^{n-4} \\,
            (1 + Z_{j} \\, Z_{j+3}) \\, (X_{j+1} \\, X_{j+2} + Y_{j+1} \\, Y_{j+2})

    Args:
        nqubits (int): number of qubits. It is equivalent to the number of chain sites :math:`N`
            plus :math:`2` boundary qubits, *i.e.*, :math:`N + 2`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        The folded XXZ Hamiltonian.

    References:
        1. R. Ruiz, A. Sopena, B. Pozsgay, and E. López, *Efficient eigenstate preparation in an
        integrable model with Hilbert space fragmentation*,
        `PRX Quantum 6, 030316 (2025) <https://doi.org/10.1103/g9f9-p8ks>`_.
    """
    if nqubits < 4:
        raise_error(ValueError, f"``nqubits`` must be at least 4, but it is {nqubits}.")

    backend = _check_backend(backend)

    if dense:
        dim = 2**nqubits
        X = backend.matrices.X
        Y = backend.matrices.Y
        Z = backend.matrices.Z

        hamiltonian = backend.zeros((dim, dim), dtype=backend.dtype)
        for j in range(nqubits - 3):
            for pauli in (X, Y):
                base = [backend.matrices.I()] * nqubits

                base[j + 1] = pauli
                base[j + 2] = pauli
                term = reduce(backend.kron, base)
                hamiltonian += term

                base[j] = Z
                base[j + 3] = Z
                term = reduce(backend.kron, base)
                hamiltonian += term

        return Hamiltonian(nqubits, (-1 / 8) * hamiltonian, backend=backend)

    X = lambda j: symbols.X(j, backend=backend)
    Y = lambda j: symbols.Y(j, backend=backend)
    Z = lambda j: symbols.Z(j, backend=backend)

    ham = (-1 / 8) * sum(
        (1 + Z(j) * Z(j + 3)) * (X(j + 1) * X(j + 2) + Y(j + 1) * Y(j + 2))
        for j in range(nqubits - 3)
    )

    return SymbolicHamiltonian(ham, backend=backend)


def GPP(
    adjacency_matrix: ArrayLike,
    penalty_coeff: float = 0.0,
    node_weights: ArrayLike | None = None,
    dense: bool = True,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """The Graph Partitioning Problem (GPP) as a quadratic function.

    For a (possibly weighted) graph :math:`G = (V, E)` defined by its set :math:`V` of vertices
    and set :math:`E` of edges, the GPP is the task of dividing a graph's vertices into :math:`m`
    subsets of approximately equal size, such that :math:`\\bigcup_{k=1}^{m} \\, V_{k} = V`,
    while minimizing the number of edges that connect vertices in different subsets.

    The formulation of the GPP as a quadratic unconstrained binary optimization (QUBO) reduces to
    minimizing the following objective function :math:`C(x)`:

    .. math::
        C(x) = \\sum_{(j,k) \\in E} \\, A_{jk} \\, (x_{j} + x_{k} - 2 \\, x_{j} \\, x_{k})
        \\, + \\lambda \\, P(x),

    where :math:`x_{j} \\in \\{0, \\, 1\\}` is a binary variable,

    .. math::
        P(x) = \\left(\\sum_{k} \\, v_{k} \\, x_{k} - \\sum_{k} \\, \\frac{v_{k}}{2}\\right)^{2}

    is a term designed to penalize deviations in the total node weight on each partition,
    and :math:`\\lambda > 0` is a hyperparameter.

    Args:
        adjacency_matrix (ndarray): Square symmetric matrix with weigths :math:`A_{jk}`
            representing the edges of the graph. For an unweighted graph,
            :math:`A_{jk} = 1, \\,\\, \\forall \\, j,k`.
        penalty_coeff (float or int, optional): hyperparameter :math:`\\lambda` defining the
            strength of the penalty term. Defaults to :math:`0.0`.
        node_weights (ArrayLike, optional): Weight of the nodes of the graph.
            Used when :math:`\\lambda \\neq 0`. If ``None``, all node weights
            default to :math:`1`. Defaults to ``None``.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        GPP Hamiltonian :math:`H_{C}`.


    References:
        1. W. Aboumrad, D. Zhu, C. Girotto, F.-H. Rouet, J. Jojo, R. Lucas, J. Pathak,
        A. Kaushik, and M. Roetteler, *Accelerating large-scale linear algebra using
        variational quantum imaginary time evolution*, `arXiv:2503.13128 (2025)
        <https://doi.org/10.48550/arXiv.2406.16142>`_.

    """
    if len(adjacency_matrix) != len(adjacency_matrix[0]):
        raise_error(
            ValueError,
            "``adjacency_matrix`` must be a square matrix, "
            + f"but it has shape ({len(adjacency_matrix)}, {len(adjacency_matrix[0])}).",
        )

    if node_weights is not None and len(node_weights) != len(adjacency_matrix):
        raise_error(
            ValueError,
            "``node_weights`` and ``adjacency_matrix`` must have the same dimensions.",
        )

    backend = _check_backend(backend)

    if isinstance(adjacency_matrix, list):
        adjacency_matrix = backend.cast(
            adjacency_matrix, dtype=type(adjacency_matrix[0][0])
        )

    if node_weights is not None and isinstance(node_weights, list):
        node_weights = backend.cast(node_weights, dtype=type(node_weights[0]))

    if penalty_coeff != 0.0 and node_weights is None:
        node_weights = backend.ones(len(adjacency_matrix), dtype=backend.int8)

    if not dense:
        return _gpp_symbolic(adjacency_matrix, penalty_coeff, node_weights, backend)

    return _gpp_dense(adjacency_matrix, penalty_coeff, node_weights, backend)


def Heisenberg(
    nqubits: int,
    coupling_constants: float | list | tuple,
    external_field_strengths: float,
    dense: bool = True,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """Heisenberg model on a :math:`1`-dimensional periodic lattice.

    The general :math:`n`-qubit Hamiltonian is given by

    .. math::
        \\begin{align}
        H &= -\\sum_{k = 1}^{n} \\, \\left(
            J_{x} \\, X_{k} \\, X_{k + 1}
            + J_{y} \\, Y_{k} \\, Y_{k + 1}
            + J_{z} \\, Z_{k} \\, Z_{k + 1} \\right) \\\\
        &\\quad\\,\\, - \\sum_{k = 1}^{n} \\left(
            h_{x} \\, X_{k} + h_{y} \\, Y_{k} + h_{z} \\, Z_{k}
            \\right) \\, ,
        \\end{align}

    where :math:`\\{J_{x}, \\, J_{y}, \\, J_{z}\\}` are called the ``coupling constants``,
    :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}` are called the ``external field strengths``,
    and :math:`\\{X, \\, Y, \\, Z\\}` are the usual Pauli operators.

    Args:
        nqubits (int): number of qubits.
        coupling_constants (list or tuple or float or int): list or tuple with the
            three coupling constants :math:`\\{J_{x}, J_{y}, J{z}\\}`.
            If ``int`` or ``float``, then :math:`J_{x} = J_{y} = J_{z}`.
        external_field_strength (list or tuple or float or int): list or tuple with the
            external magnetic field strengths :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}`.
            If ``int`` or ``float``, then :math:`h_{x} = h_{y} = h_{z}`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Heisenberg Hamiltonian.
    """
    if isinstance(coupling_constants, (list, tuple)) and len(coupling_constants) != 3:
        raise_error(
            ValueError,
            "When `coupling_constants` is type `int` or `list`, it must have length == 3.",
        )

    if isinstance(coupling_constants, (float, int)):
        coupling_constants = [coupling_constants] * 3

    if (
        isinstance(external_field_strengths, (list, tuple))
        and len(external_field_strengths) != 3
    ):
        raise_error(
            ValueError,
            "When `external_field_strengths` is type `int` or `list`, it must have length == 3.",
        )

    if isinstance(external_field_strengths, (float, int)):
        external_field_strengths = [external_field_strengths] * 3

    backend = _check_backend(backend)

    paulis = (symbols.X, symbols.Y, symbols.Z)

    if dense:
        condition = lambda i, j: i in {j % nqubits, (j + 1) % nqubits}
        matrix = backend.zeros((2**nqubits, 2**nqubits), dtype=backend.complex128)
        for ind, pauli in enumerate(paulis):
            double_term = _build_spin_model(
                nqubits, pauli(0, backend=backend).matrix, condition, backend
            )
            double_term = backend.cast(double_term, dtype=double_term.dtype)
            matrix = matrix - coupling_constants[ind] * double_term
            matrix = (
                matrix
                + external_field_strengths[ind]
                * _OneBodyPauli(nqubits, pauli, dense, backend).matrix
            )

        return Hamiltonian(nqubits, matrix, backend=backend)

    def h(symbol):
        return lambda q1, q2: symbol(q1, backend=backend) * symbol(q2, backend=backend)

    def term(q1, q2):
        return sum(
            coeff * h(operator)(q1, q2)
            for coeff, operator in zip(coupling_constants, paulis)
        )

    form = -1 * sum(term(qubit, qubit + 1) for qubit in range(nqubits - 1)) - term(
        nqubits - 1, 0
    )
    form -= sum(
        field_strength * pauli(qubit)
        for qubit in range(nqubits)
        for field_strength, pauli in zip(external_field_strengths, paulis)
        if field_strength != 0.0
    )

    ham = SymbolicHamiltonian(form=form, backend=backend)
    return ham


def Ising(
    nqubits: int | nx.Graph,
    coupling_constants: float | ArrayLike,
    local_field_strengths: float | tuple | ArrayLike,
    dense: bool = True,
    closed_boundary: bool = True,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """Ising model on a one-dimensional chain or on a lattice with arbitrary connectivity.

    .. math::
        H = \\sum_{\\langle j, \\, k \\rangle} \\, J_{jk} \\, Z_{j} \\, Z_{k}
            + \\sum_{j} h_{j}^{(z)} \\, Z_{j} + \\sum_{j} h_{j}^{(x)} \\, X_{j} \\, ,

    where :math:`\\langle j, \\, k \\rangle` runs over all pairs of coupled (*i.e.*, connected)
    qubits, :math:`\\{J_{jk}\\}_{j,k}` are called the ``coupling constants``,
    :math:`\\{h_{j}^{(z)}, h_{j}^{(x)}\\}_{j}` are called the ``local field strengths``,
    and :math:`\\{X, \\, Y, \\, Z\\}` are the usual Pauli operators.

    If ``nqubits`` is an ``int``, the qubits form a one-dimensional chain and
    :math:`\\langle j, \\, k \\rangle` runs over the nearest neighbors :math:`(j, \\, j + 1)`,
    with :math:`J_{jk} \\equiv J_{j}`. The pair :math:`(n - 1, \\, 0)` is also coupled if
    ``closed_boundary=True``.

    For an arbitrary connectivity, ``nqubits`` can be a :class:`networkx.Graph`, in which every
    node is a qubit and every edge is a pair of coupled qubits. If the nodes of the graph
    are exactly the integers :math:`\\{0, \\, 1, \\, \\dots, \\, n - 1\\}`, every node is its
    own qubit label :math:`j`. Otherwise, qubits are labeled by the order in which the nodes
    are stored in the graph, *i.e.*, the :math:`j`-th node of ``list(nqubits.nodes)`` has label
    :math:`j`. The graph defines all the bonds of the model. Therefore, ``closed_boundary`` is
    ignored, and any periodicity must be encoded as edges of the graph. Isolated nodes are
    qubits with no coupling, and self-loops are not allowed.

    The graph can be weighted. The coupling constant of the bond between qubits :math:`j` and
    :math:`k` is :math:`J_{jk} = J \\, w_{j,k}`, where :math:`J` is ``coupling_constants`` and
    :math:`w_{j,k}` is the ``"weight"`` attribute of the corresponding edge. Edges without this
    attribute have weight :math:`1`, and therefore an unweighted graph has a uniform coupling
    constant :math:`J`. Edge directions are ignored. Edges between the same pair of nodes,
    *e.g.*, parallel edges of a multigraph, are merged into a single bond, and they must have
    the same weight. The local field strengths are indexed by the qubit label.

    Args:
        nqubits (int or :class:`networkx.Graph`): if an ``int``, number of qubits in a
            one-dimensional chain. If a :class:`networkx.Graph`, connectivity graph of the
            qubits, with one node per qubit and, optionally, a ``"weight"`` attribute per edge
            that scales ``coupling_constants``.
        coupling_constants (float or int or ArrayLike): ``list`` or ``ndarray`` containing all
            coupling constants :math:`\\{J_{k}\\}_{k}` of the chain. If ``int`` or ``float``,
            then :math:`J_{k}` is equal to ``coupling_constants`` for all :math:`k`. If
            ``nqubits`` is a :class:`networkx.Graph`, it must be an ``int`` or ``float``.
        local_field_strengths (list or tuple or float or int): array containing all local
            field strengths :math:`\\left\\{\\left(h_{j}^{(z)}, \\,
            h_{j}^{(x)}\\right)\\right\\}_{j}`, *e.g.*, :math:`\\left\\{h_{j}^{(z)}\\right\\}_{j}`
            in the first column and :math:`\\left\\{h_{j}^{(x)}\\right\\}_{j}` in the second column.
            If ``int`` or ``float``, the coefficients :math:`h_{j}^{(z)}` and :math:`h_{j}^{(x)}`
            are equal to ``local_field_strengths``, :math:`\\forall \\, j`. If ``Tuple[int, int]``,
            :math:`h_{j}^{(z)}` :math:`\\left(h_{j}^{(x)}\\right)` is set to the first (second)
            element of the tuple, :math:`\\forall \\, j`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        closed_boundary (bool, optional): If ``True``, returns Ising Hamiltonian
            with periodic boundary condition. If ``False``, returns Hamiltonian
            with open boundaries. Ignored if ``nqubits`` is a :class:`networkx.Graph`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Ising Hamiltonian.
    """
    backend = _check_backend(backend)

    if isinstance(nqubits, nx.Graph):
        if not isinstance(coupling_constants, (int, float, np.integer, np.floating)):
            raise_error(
                TypeError,
                "``coupling_constants`` must be type float or int when ``nqubits`` is "
                + f"a ``networkx.Graph``, but it is type {type(coupling_constants)}.",
            )

        nnodes, weights = _graph_bonds(nqubits)

        bonds = [
            (qubit_a, qubit_b, coupling_constants * weight)
            for (qubit_a, qubit_b), weight in weights.items()
        ]
        nqubits = nnodes
    else:
        if isinstance(coupling_constants, (list, tuple)) and (
            (closed_boundary and len(coupling_constants) != nqubits)
            or (not closed_boundary and len(coupling_constants) != (nqubits - 1))
        ):
            raise_error(
                ValueError,
                f"``coupling_constants`` does not have the correct len ({len(coupling_constants)}) "
                + f" for the given ``closed_boundary`` condition ({closed_boundary}).",
            )

        if isinstance(coupling_constants, (int, float)):
            coupling_constants = [coupling_constants] * (nqubits - 1)
            if closed_boundary:
                coupling_constants.append(coupling_constants[-1])

        # nearest-neighbor bonds of the chain
        bonds = [
            (qubit, qubit + 1, coeff)
            for qubit, coeff in zip(range(nqubits - 1), coupling_constants)
        ]
        if closed_boundary and nqubits > 2:
            bonds.append((nqubits - 1, 0, coupling_constants[-1]))

    if isinstance(local_field_strengths, (int, float)):
        coeffs_z = [local_field_strengths] * nqubits
        coeffs_x = [local_field_strengths] * nqubits
    elif isinstance(local_field_strengths, (tuple, list)):
        coeffs_z = [local_field_strengths[0]] * nqubits
        coeffs_x = [local_field_strengths[1]] * nqubits
    else:
        coeffs_z = local_field_strengths[:, 0]
        coeffs_x = local_field_strengths[:, 1]

    qubits = list(range(nqubits))

    if dense:
        matrix = backend.zeros((2**nqubits, 2**nqubits), dtype=backend.complex128)
        base_string = [backend.matrices.I()] * nqubits
        for qubit_a, qubit_b, coeff in bonds:
            base_string[qubit_a] = backend.matrices.Z
            base_string[qubit_b] = backend.matrices.Z
            matrix += float(coeff) * reduce(backend.kron, base_string)
            base_string[qubit_a] = backend.matrices.I()
            base_string[qubit_b] = backend.matrices.I()

        if any(coeff != 0 for coeff in coeffs_z):
            base_string = [backend.matrices.I()] * nqubits
            for qubit, coeff in zip(qubits, coeffs_z):
                base_string[qubit] = backend.matrices.Z
                matrix += float(coeff) * reduce(backend.kron, base_string)
                base_string[qubit] = backend.matrices.I()

        if any(coeff != 0 for coeff in coeffs_x):
            base_string = [backend.matrices.I()] * nqubits
            for qubit, coeff in zip(qubits, coeffs_x):
                base_string[qubit] = backend.matrices.X
                matrix += float(coeff) * reduce(backend.kron, base_string)
                base_string[qubit] = backend.matrices.I()

        return Hamiltonian(nqubits, matrix, backend=backend)

    interaction = lambda q1, q2: (
        symbols.Z(q1, backend=backend) * symbols.Z(q2, backend=backend)
    )

    # a sympy zero keeps the form a valid expression even if there are no bonds
    form = sum(
        (
            float(coeff) * interaction(qubit_a, qubit_b)
            for qubit_a, qubit_b, coeff in bonds
        ),
        start=sympy.S.Zero,
    )

    if any(coeff != 0 for coeff in coeffs_z):
        form += sum(
            float(coeff) * symbols.Z(qubit, backend=backend)
            for qubit, coeff in zip(qubits, coeffs_z)
        )

    if any(coeff != 0 for coeff in coeffs_x):
        form += sum(
            float(coeff) * symbols.X(qubit, backend=backend)
            for qubit, coeff in zip(qubits, coeffs_x)
        )

    return SymbolicHamiltonian(form=form, nqubits=nqubits, backend=backend)


def LABS(
    nqubits: int, dense: bool = True, backend: Backend | None = None
) -> Hamiltonian | SymbolicHamiltonian:
    """Create Hamiltonian of the Low Autocorrelation Binary Sequences (LABS) problem.

    Given an integer :math:`n > 2`, the LABS problem consists of finding a binary sequence
    :math:`b \\in \\{0, \\, 1\\}^{n}` the minimizes the *sidelobe energy* :math:`E(b)`
    defined as

    .. math::
        E(b) = \\sum_{j=1}^{n-1} \\, C_{j}^{2}(b) \\, ;
        \\quad C_{j}(b) = \\sum_{k=0}^{n-j} \\, b_{k} \\, b_{k + j} \\, ,

    where :math:`C_{j}(b)` is the :math:`j`-th *autocorrelation* of :math:`b`.

    Args:
        nqubits (int): Total number of qubits.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    References:
        1. T. Packebusch and S. Mertens, *Low autocorrelation binary sequences*,
        `J. Phys. A: Math. Theor. 49 (2016) 165001
        <https://doi.org/10.1088/1751-8113/49/16/165001>`_.
    """
    if nqubits < 2:
        raise_error(
            ValueError,
            f"LABS problem only defined for ``nqubits > 2``, but ``nqubits = {nqubits}``.",
        )

    hamiltonian = 0
    for ind in range(1, nqubits):
        term = sum(
            symbols.Z(qubit, backend=backend) * symbols.Z(qubit + ind, backend=backend)
            for qubit in range(nqubits - ind)
        )
        hamiltonian += term**2

    hamiltonian = SymbolicHamiltonian(hamiltonian, nqubits=nqubits, backend=backend)

    if dense:
        return hamiltonian.dense

    return hamiltonian


def MaxCut(
    nqubits: int,
    dense: bool = True,
    adj_matrix: list[list[float]] | ArrayLike | None = None,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """Max Cut Hamiltonian.

    .. math::
        H = -\\frac{1}{2} \\, \\sum_{j, k = 0}^{n - 1} \\, A_{jk} \\,
            \\left(1 - Z_{j} \\, Z_{k}\\right) \\, ,

    where :math:`A` is the adjacency matrix of the graph. Up to a constant energy shift, this is
    the :class:`qibo.hamiltonians.Ising` Hamiltonian, without local fields, on the graph with
    weights :math:`w_{jk} = (A_{jk} + A_{kj}) / 2` for :math:`j \\neq k`, *i.e.*,

    .. math::
        H = \\sum_{j < k} \\, w_{jk} \\, Z_{j} \\, Z_{k} - \\sum_{j < k} \\, w_{jk} \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
        adj_matrix (list[list[float]] | np.ndarray): Adjecency matrix of the graph. Defaults to a
            homogeneous fully connected graph with all edges having an equal 1.0 weight.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    if adj_matrix is None:
        adj_matrix = np.ones((nqubits, nqubits))

    adj_matrix = np.asarray(adj_matrix)
    if adj_matrix.shape != (nqubits, nqubits):
        raise_error(
            RuntimeError,
            f"Expected an adjacency matrix of shape ({nqubits},{nqubits}) for a {nqubits}-qubits system.",
        )

    # the ordered pairs (j, k) and (k, j) both contribute to the same bond, and
    # self-loops only contribute to the constant, which vanishes since Z_{j}^{2} = 1
    weights = (adj_matrix + adj_matrix.T) / 2
    np.fill_diagonal(weights, 0.0)

    ham = Ising(
        nx.from_numpy_array(weights),
        coupling_constants=1.0,
        local_field_strengths=0.0,
        dense=dense,
        backend=backend,
    )

    return ham - float(weights.sum() / 2)


def TFIM(
    nqubits: int,
    h: float = 0.0,
    dense: bool = True,
    closed_boundary: bool = True,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """One-dimensional, :math:`n`-qubit Transverse field Ising model.

    .. math::
        H = - \\sum _{k=0}^{n-1} \\, \\left(Z_{k} \\, Z_{k + 1} + h \\, X_{k}\\right) \\, ,

    with :math:`Z_{n-1} Z_{n} \\equiv Z_{n-1} Z_{0}` if ``closed_boundary=True``.

    It is the :class:`qibo.hamiltonians.Ising` model with coupling constants :math:`-1` and
    local field strengths :math:`(0, \\, -h)`.

    Args:
        nqubits (int): number of qubits.
        h (float, optional): value of the transverse field. Defaults to :math:`0.0`.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        closed_boundary (bool, optional): If ``True``, returns TFIM with periodic boundary
            condition. If ``False``, returns Hamiltonian with open boundaries.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    if nqubits < 2:
        raise_error(ValueError, "Number of qubits must be larger than one.")

    return Ising(
        nqubits,
        coupling_constants=-1,
        local_field_strengths=(0, -h),
        dense=dense,
        closed_boundary=closed_boundary,
        backend=backend,
    )


def XXX(
    nqubits: int,
    coupling_constant: float = 1,
    external_field_strengths: float | list | tuple | None = None,
    dense: bool = True,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """Heisenberg :math:`\\mathrm{XXX}` model with periodic boundary conditions.

    The :math:`n`-qubit :math:`\\mathrm{XXX}` Hamiltonian is given by

    .. math::
        H = -J \\, \\sum_{k = 1}^{n} \\, \\left(
            X_{k} \\, X_{k + 1} + Y_{k} \\, Y_{k + 1} + Z_{k} \\, Z_{k + 1}
            \\right)
            - \\sum_{k = 1}^{n} \\left(h_{x} \\, X_{k} + h_{y} \\, Y_{k}
            + h_{z} \\, Z_{k} \\right) \\, ,

    where :math:`J` is the ``coupling_constant``, :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}`
    are called the ``external field strengths``, and :math:`\\{X, \\, Y, \\, Z\\}` are the usual
    Pauli operators.

    Args:
        nqubits (int): number of qubits.
        coupling_constant (float or int, optional): coupling constant :math:`J`.
            Defaults to :math:`1`.
        external_field_strengths (float or int or list or tuple, optional): list or tuple with the
            external magnetic field strengths :math:`\\{h_{x}, \\, h_{y}, \\, h_{z}\\}`.
            If ``int`` or ``float``, then :math:`h_{x} = h_{y} = h_{z}`.
            Defaults to :math:`[1/2, \\, 0, \\, 0]`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Heisenberg :math:`\\mathrm{XXX}` Hamiltonian.
    """
    if external_field_strengths is None:
        external_field_strengths = [0.5, 0, 0]

    if not isinstance(coupling_constant, (float, int)):
        raise_error(
            TypeError,
            "`coupling_constant` must be type float or int, "
            + f"but it is type {type(coupling_constant)}.",
        )

    return Heisenberg(
        nqubits,
        coupling_constants=coupling_constant,
        external_field_strengths=external_field_strengths,
        dense=dense,
        backend=backend,
    )


def XXZ(
    nqubits: int,
    delta: float = 0.5,
    dense: bool = True,
    backend: Backend | None = None,
) -> Hamiltonian | SymbolicHamiltonian:
    """Heisenberg :math:`\\mathrm{XXZ}` model with periodic boundary conditions.

    .. math::
        H = \\sum _{k=0}^N \\, \\left( X_{k} \\, X_{k + 1} + Y_{k} \\, Y_{k + 1}
            + \\delta Z_{k} \\, Z_{k + 1} \\right) \\, .

    Example:
        .. testcode::

            from qibo.hamiltonians import XXZ
            h = XXZ(3) # initialized XXZ model with 3 qubits

    Args:
        nqubits (int): number of qubits.
        delta (float or int, optional): coefficient for the :math:`Z` component.
            Defaults to :math:`0.5`.
        dense (bool, optional): If ``True``, creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.

    Returns:
        :class:`qibo.hamiltonians.Hamiltonian` or :class:`qibo.hamiltonians.SymbolicHamiltonian`:
        Heisenberg :math:`\\mathrm{XXZ}` Hamiltonian.
    """
    if nqubits < 2:
        raise_error(ValueError, "Number of qubits must be larger than one.")

    return Heisenberg(nqubits, [-1, -1, -delta], 0, dense=dense, backend=backend)


def X(
    nqubits: int, dense: bool = True, backend: Backend | None = None
) -> Hamiltonian | SymbolicHamiltonian:
    """Non-interacting Pauli-:math:`X` Hamiltonian.

    .. math::
        H = - \\sum _{k=0}^N \\, X_{k} \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
            Defaults to ``True``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    return _OneBodyPauli(nqubits, symbols.X, dense, backend=backend)


def Y(
    nqubits: int, dense: bool = True, backend: Backend | None = None
) -> Hamiltonian | SymbolicHamiltonian:
    """Non-interacting Pauli-:math:`Y` Hamiltonian.

    .. math::
        H = - \\sum _{k=0}^{N} \\, Y_{k} \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    return _OneBodyPauli(nqubits, symbols.Y, dense, backend=backend)


def Z(
    nqubits: int, dense: bool = True, backend: Backend | None = None
) -> Hamiltonian | SymbolicHamiltonian:
    """Non-interacting Pauli-:math:`Z` Hamiltonian.

    .. math::
        H = - \\sum _{k=0}^{N} \\, Z_{k} \\, .

    Args:
        nqubits (int): number of qubits.
        dense (bool, optional): If ``True`` it creates the Hamiltonian as a
            :class:`qibo.core.hamiltonians.Hamiltonian`, otherwise it creates
            a :class:`qibo.core.hamiltonians.SymbolicHamiltonian`.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend.
            Defaults to ``None``.
    """
    return _OneBodyPauli(nqubits, symbols.Z, dense, backend=backend)
