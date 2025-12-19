"""Tests methods from `qibo/src/hamiltonians/models.py`."""

from functools import reduce

import numpy as np
import pytest

from qibo import hamiltonians, matrices, symbols
from qibo.hamiltonians import MaxCut, SymbolicHamiltonian
from qibo.hamiltonians.models import GPP, LABS, XXX, Heisenberg

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
        H = getattr(hamiltonians, model)(nqubits=1, backend=backend)


@pytest.mark.parametrize("dense", [True, False])
def test_xxx(backend, dense):
    nqubits = 2

    with pytest.raises(ValueError):
        test = XXX(
            nqubits,
            coupling_constant=1,
            external_field_strengths=[0, 1],
            dense=dense,
            backend=backend,
        )

    with pytest.raises(TypeError):
        test = XXX(nqubits, coupling_constant=[1], dense=dense, backend=backend)

    with pytest.raises(ValueError):
        test = Heisenberg(
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
    rows, columns = backend.np.nonzero(backend.np.tril(adj_matrix, -1))
    target = 0
    for col, row in zip(columns, rows):
        term_col = base_string.copy()
        term_col[int(col)] = term
        term_col = reduce(backend.np.kron, term_col)

        term_row = base_string.copy()
        term_row[int(row)] = term
        term_row = reduce(backend.np.kron, term_row)

        target += term_row + term_col - 2 * (term_col @ term_row)

    if penalty_coeff != 0.0:
        penalty = 0
        for elem in range(len(adj_matrix)):
            term_weight = base_string.copy()
            term_weight[elem] = term - backend.matrices.I() / 2
            term_weight = reduce(backend.np.kron, term_weight)
            penalty += term_weight

        target += penalty_coeff * (penalty**2)

    backend.assert_allclose(hamiltonian.matrix, target)
