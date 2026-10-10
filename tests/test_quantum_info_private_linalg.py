"""Tests for the protected functions in ``qibo.quantum_info._linalg_operations``."""

import numpy as np
import pytest

from qibo.quantum_info._linalg_operations import (
    _gram_schmidt_process,
    _vector_projection,
)


def _random_vectors(rng, shape, complex_valued):
    vectors = rng.normal(size=shape)
    if complex_valued:
        vectors = vectors + 1j * rng.normal(size=shape)

    return vectors


@pytest.mark.parametrize("complex_valued", [False, True])
def test_vector_projection_single_direction(backend, complex_valued):
    rng = np.random.default_rng(1234)
    for _ in range(5):
        vector = _random_vectors(rng, 6, complex_valued)
        direction = _random_vectors(rng, 6, complex_valued)

        projection = _vector_projection(
            backend.cast(vector), backend.cast(direction), backend=backend
        )
        residual = backend.to_numpy(backend.cast(vector) - projection)

        # the residual is orthogonal to the direction
        np.testing.assert_allclose(np.vdot(direction, residual), 0.0, atol=1e-10)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("complex_valued", [False, True])
def test_gram_schmidt_process(backend, complex_valued, as_list):
    rng = np.random.default_rng(4321)
    dims, ndirections = 6, 3

    for _ in range(5):
        # orthonormal directions, one per row
        directions, _ = np.linalg.qr(
            _random_vectors(rng, (dims, ndirections), complex_valued)
        )
        directions = directions.T
        vector = _random_vectors(rng, dims, complex_valued)

        if as_list:
            directions_backend = [backend.cast(direction) for direction in directions]
        else:
            directions_backend = backend.cast(directions)
        orthogonalized = _gram_schmidt_process(
            backend.cast(vector), directions_backend, backend=backend
        )
        orthogonalized = backend.to_numpy(orthogonalized)

        for direction in directions:
            np.testing.assert_allclose(
                np.vdot(direction, orthogonalized), 0.0, atol=1e-10
            )


def test_vector_projection_default_backend():
    vector = np.array([1.0, 2.0, 3.0])
    direction = np.array([1.0, 0.0, 0.0])

    projection = _vector_projection(vector, direction)

    np.testing.assert_allclose(projection, [1.0, 0.0, 0.0], atol=1e-12)
