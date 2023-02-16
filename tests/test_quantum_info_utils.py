import numpy as np
import pytest

from qibo.quantum_info import *


def test_shannon_entropy_errors(backend):
    with pytest.raises(ValueError):
        p = np.asarray([1.0, 0.0])
        p = backend.cast(p, dtype=p.dtype)
        shannon_entropy(p, -2)
    with pytest.raises(TypeError):
        p = np.asarray([[1.0], [0.0]])
        p = backend.cast(p, dtype=p.dtype)
        shannon_entropy(p)
    with pytest.raises(TypeError):
        p = np.asarray([])
        p = backend.cast(p, dtype=p.dtype)
        shannon_entropy(p)
    with pytest.raises(ValueError):
        p = np.asarray([1.0, -1.0])
        p = backend.cast(p, dtype=p.dtype)
        shannon_entropy(p)
    with pytest.raises(ValueError):
        p = np.asarray([1.1, 0.0])
        p = backend.cast(p, dtype=p.dtype)
        shannon_entropy(p)
    with pytest.raises(ValueError):
        p = np.asarray([0.5, 0.4999999])
        p = backend.cast(p, dtype=p.dtype)
        shannon_entropy(p)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_shannon_entropy(backend, base):
    prob_array = np.asarray([1.0, 0.0])
    prob_array = backend.cast(prob_array, dtype=prob_array.dtype)
    result = shannon_entropy(prob_array, base)
    backend.assert_allclose(result, 0.0)

    if base == 2:
        prob_array = np.asarray([0.5, 0.5])
        prob_array = backend.cast(prob_array, dtype=prob_array.dtype)
        result = shannon_entropy(prob_array, base)
        backend.assert_allclose(result, 1.0)


def test_hellinger(backend):
    with pytest.raises(TypeError):
        p = np.random.rand(1, 2)
        q = np.random.rand(1, 5)
        p = backend.cast(p, dtype=p.dtype)
        q = backend.cast(q, dtype=q.dtype)
        hellinger_distance(p, q)
    with pytest.raises(TypeError):
        p = np.random.rand(1, 2)[0]
        q = np.array([])
        p = backend.cast(p, dtype=p.dtype)
        q = backend.cast(q, dtype=q.dtype)
        hellinger_distance(p, q)
    with pytest.raises(ValueError):
        p = np.array([-1, 2.0])
        q = np.random.rand(1, 5)[0]
        p = backend.cast(p, dtype=p.dtype)
        q = backend.cast(q, dtype=q.dtype)
        hellinger_distance(p, q, validate=True)
    with pytest.raises(ValueError):
        p = np.random.rand(1, 2)[0]
        q = np.array([1.0, 0.0])
        p = backend.cast(p, dtype=p.dtype)
        q = backend.cast(q, dtype=q.dtype)
        hellinger_distance(p, q, validate=True)
    with pytest.raises(ValueError):
        p = np.array([1.0, 0.0])
        q = np.random.rand(1, 2)[0]
        p = backend.cast(p, dtype=p.dtype)
        q = backend.cast(q, dtype=q.dtype)
        hellinger_distance(p, q, validate=True)

    p = np.array([1.0, 0.0])
    q = np.array([1.0, 0.0])
    p = backend.cast(p, dtype=p.dtype)
    q = backend.cast(q, dtype=q.dtype)
    backend.assert_allclose(hellinger_distance(p, q), 0.0)
    backend.assert_allclose(hellinger_fidelity(p, q), 1.0)
