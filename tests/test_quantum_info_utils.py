from re import finditer

import numpy as np
import pytest

from qibo.quantum_info.utils import (
    hamming_weight,
    hellinger_distance,
    hellinger_fidelity,
    shannon_entropy,
)


@pytest.mark.parametrize("bitstring", range(2**5))
@pytest.mark.parametrize(
    "kind",
    [
        str,
        list,
        tuple,
        lambda b: np.array(list(b)),
        lambda b: int(b, 2),
        lambda b: list(map(int, b)),
    ],
)
def test_hamming_weight(bitstring, kind):
    with pytest.raises(TypeError):
        hamming_weight("0101", return_indexes="True")

    bitstring = f"{bitstring:b}"
    weight_test = len(bitstring.replace("0", ""))
    indexes_test = [item.start() for item in finditer("1", bitstring)]

    weight = hamming_weight(kind(bitstring), False)
    indexes = hamming_weight(kind(bitstring), True)

    assert weight == weight_test
    assert indexes == indexes_test


def test_shannon_entropy_errors(backend):
    with pytest.raises(ValueError):
        prob = np.asarray([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob, -2)
    with pytest.raises(TypeError):
        prob = np.asarray([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob)
    with pytest.raises(TypeError):
        prob = np.asarray([])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob)
    with pytest.raises(ValueError):
        prob = np.asarray([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob)
    with pytest.raises(ValueError):
        prob = np.asarray([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob)
    with pytest.raises(ValueError):
        prob = np.asarray([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        shannon_entropy(prob)


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
        prob = np.random.rand(1, 2)
        prob_q = np.random.rand(1, 5)
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        hellinger_distance(prob, prob_q)
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        hellinger_distance(prob, prob_q)
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        hellinger_distance(prob, prob_q, validate=True)
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        hellinger_distance(prob, prob_q, validate=True)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        hellinger_distance(prob, prob_q, validate=True)

    prob = np.array([1.0, 0.0])
    prob_q = np.array([1.0, 0.0])
    prob = backend.cast(prob, dtype=prob.dtype)
    prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
    backend.assert_allclose(hellinger_distance(prob, prob_q), 0.0)
    backend.assert_allclose(hellinger_fidelity(prob, prob_q), 1.0)
