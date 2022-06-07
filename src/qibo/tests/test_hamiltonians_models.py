"""Tests methods from `qibo/src/hamiltonians/models.py`."""
import pytest
import numpy as np
from qibo import hamiltonians, matrices


models_config = [
    ("TFIM", {"nqubits": 3, "h": 0.0}, "tfim_N3h0.0.out"),
    ("TFIM", {"nqubits": 3, "h": 0.5}, "tfim_N3h0.5.out"),
    ("TFIM", {"nqubits": 3, "h": 1.0}, "tfim_N3h1.0.out"),
    ("XXZ", {"nqubits": 3, "delta": 0.0}, "heisenberg_N3delta0.0.out"),
    ("XXZ", {"nqubits": 3, "delta": 0.5}, "heisenberg_N3delta0.5.out"),
    ("XXZ", {"nqubits": 3, "delta": 1.0}, "heisenberg_N3delta1.0.out"),
    ("X", {"nqubits": 3}, "x_N3.out"),
    ("Y", {"nqubits": 4}, "y_N4.out"),
    ("Z", {"nqubits": 5}, "z_N5.out"),
    ("MaxCut", {"nqubits": 3}, "maxcut_N3.out"),
    ("MaxCut", {"nqubits": 4}, "maxcut_N4.out"),
    ("MaxCut", {"nqubits": 5}, "maxcut_N5.out"),
]
@pytest.mark.parametrize(("model", "kwargs", "filename"), models_config)
def test_hamiltonian_models(backend, model, kwargs, filename):
    """Test pre-coded Hamiltonian models generate the proper matrices."""
    from qibo.tests.test_models_variational import assert_regression_fixture
    H = getattr(hamiltonians, model)(**kwargs, backend=backend)
    matrix = backend.to_numpy(H.matrix).flatten().real
    assert_regression_fixture(backend, matrix, filename)


@pytest.mark.parametrize("nqubits", [3, 4])
@pytest.mark.parametrize("dense,calcterms", [(True, False), (False, False), (False, True)])
def test_maxcut(backend, nqubits, dense, calcterms):
    size = 2 ** nqubits
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
    target_ham = backend.cast(- ham / 2)
    final_ham = hamiltonians.MaxCut(nqubits, dense, backend=backend)
    if (not dense) and calcterms:
        _ = final_ham.terms
    backend.assert_allclose(final_ham.matrix, target_ham)
