"""Tests methods from `qibo/src/hamiltonians/models.py`."""

import numpy as np
import pytest

from qibo import hamiltonians, matrices
from qibo.hamiltonians.models import XXX, Heisenberg

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
            final_ham = hamiltonians.MaxCut(
                nqubits, dense, adj_matrix=adj_matrix, backend=backend
            )
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
        final_ham = hamiltonians.MaxCut(
            nqubits, dense, adj_matrix=adj_matrix, backend=backend
        )
        backend.assert_allclose(final_ham.matrix, target_ham)


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
