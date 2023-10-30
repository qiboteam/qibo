"""Tests for qibo.models.encodings"""
import numpy as np
import pytest

from qibo.models.encodings import unary_encoder


@pytest.mark.parametrize("nqubits", [2, 4, 8])
def test_unary_encoder(backend, nqubits):
    sampler = np.random.default_rng(1)

    with pytest.raises(TypeError):
        data = sampler.random((nqubits, nqubits))
        data = backend.cast(data, dtype=data.dtype)
        unary_encoder(data)
    with pytest.raises(ValueError):
        data = sampler.random(nqubits + 1)
        data = backend.cast(data, dtype=data.dtype)
        unary_encoder(data)

    # sampling random data in interval [-1, 1]
    sampler = np.random.default_rng(1)
    data = 2 * sampler.random(nqubits) - 1
    data = backend.cast(data, dtype=data.dtype)

    circuit = unary_encoder(data)
    state = backend.execute_circuit(circuit).state()
    indexes = np.flatnonzero(state)
    state = np.sort(state[indexes])

    backend.assert_allclose(
        state, np.sort(data) / backend.calculate_norm(data, order=2)
    )
