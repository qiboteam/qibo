"""Tests for the backend-level functions in ``qibo.quantum_info._quantum_info``."""

import numpy as np
import pytest


@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize("dims", [2, 4])
@pytest.mark.parametrize("order", ["row", "column"])
def test_bcsz_super_op_is_cptp(backend, order, dims, rank):
    if backend.name == "qibojit":
        pytest.skip("``qibojit`` provides its own implementation of the BCSZ measure.")

    function = getattr(backend.qinfo, f"_super_op_from_bcsz_measure_{order}")

    for seed in range(5):
        backend.set_seed(seed)
        choi = backend.to_numpy(function(dims, rank))

        # complete positivity
        eigenvalues = np.linalg.eigvalsh((choi + choi.conj().T) / 2)
        assert eigenvalues.min() > -1e-8

        # trace preservation: the output system is the first subsystem
        # in row-vectorization and the second one in column-vectorization
        choi = choi.reshape((dims,) * 4)
        subscripts = "ajak->jk" if order == "row" else "jaka->jk"
        reduced = np.einsum(subscripts, choi)
        np.testing.assert_allclose(reduced, np.eye(dims), atol=1e-8)


@pytest.mark.parametrize("threshold", [3, 8, 50])
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6, 10, 33])
def test_inverse_tril(backend, dim, threshold):
    for seed in range(5):
        backend.set_seed(seed)
        matrix = backend.identity(dim, dtype=backend.uint8)
        backend.qinfo._fill_tril(matrix, False)
        matrix = backend.to_numpy(matrix).astype(np.int64)
        matrix = np.tril(matrix)

        matrix_backend = backend.cast(matrix, dtype=backend.uint8)
        inverse = backend.qinfo._inverse_tril(matrix_backend, threshold)
        inverse = backend.to_numpy(inverse).astype(np.int64) % 2

        np.testing.assert_array_equal((matrix @ inverse) % 2, np.eye(dim))


@pytest.mark.parametrize("complex_environment", [False, True])
@pytest.mark.parametrize("nkraus", [2, 3, 4])
@pytest.mark.parametrize("nqubits", [1, 2])
def test_stinespring_kraus_round_trip(backend, nqubits, nkraus, complex_environment):
    if backend.name == "qibojit":
        pytest.skip("``qibojit`` provides its own implementation of the conversion.")

    from qibo.quantum_info import kraus_to_stinespring, stinespring_to_kraus

    dims = 2**nqubits
    rng = np.random.default_rng(1234)
    # Kraus operators of a trace-preserving channel from the blocks of an isometry
    isometry, _ = np.linalg.qr(
        rng.normal(size=(nkraus * dims, dims))
        + 1j * rng.normal(size=(nkraus * dims, dims))
    )
    kraus = [isometry[k * dims : (k + 1) * dims] for k in range(nkraus)]
    kraus_ops = [(tuple(range(nqubits)), backend.cast(op)) for op in kraus]

    environment = None
    if complex_environment:
        environment = rng.normal(size=nkraus) + 1j * rng.normal(size=nkraus)
        environment = backend.cast(environment / np.linalg.norm(environment))

    stinespring = kraus_to_stinespring(
        kraus_ops, nqubits=nqubits, initial_state_env=environment, backend=backend
    )
    recovered = stinespring_to_kraus(
        stinespring,
        dim_env=nkraus,
        initial_state_env=environment,
        nqubits=nqubits,
        backend=backend,
    )

    np.testing.assert_allclose(backend.to_numpy(recovered), np.array(kraus), atol=1e-10)
