"""Tests methods from `qibo/src/hamiltonians/models.py`."""

from functools import reduce

import numpy as np
import pytest

from qibo import hamiltonians, matrices, symbols
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.hamiltonians.models import (
    GPP,
    LABS,
    TFIM,
    XXX,
    FermiHubbard,
    FoldedXXZ,
    Heisenberg,
    Ising,
    MaxCut,
    _multikron,
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