"""Tests for the Dürr-Høyer algorithm in `qibo/models/algorithms/dh.py`."""

import numpy as np
import pytest

from qibo.models import DH


def _table(backend, nvalues, seed):
    return backend.random_normal(0.0, 1.0, size=nvalues, seed=seed, dtype="float64")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("nvalues", [2, 4, 8, 16, 32])
def test_dh(backend, nvalues, seed):
    values = _table(backend, nvalues, seed)

    # the time-out is long, so the probability of not finding the minimum is below 2^-10
    backend.set_seed(seed)
    index, napplications = DH(values, timeout=10, seed=seed, backend=backend)

    assert isinstance(index, int)
    assert napplications >= 0
    assert float(values[index]) == min(float(value) for value in values)


def test_dh_success_probability(backend):
    # Theorem 1: the minimum is found with probability at least 1 / 2
    values = _table(backend, 16, seed=3)
    minimum = min(float(value) for value in values)

    successes = 0
    for seed in range(40):
        backend.set_seed(seed)
        index, _ = DH(values, seed=seed, backend=backend)
        successes += float(values[index]) == minimum

    assert successes >= 20


def test_dh_ties(backend):
    values = backend.cast([3, 1, 4, 1, 5, 1, 9, 2], dtype=backend.float64)

    backend.set_seed(0)
    index, _ = DH(values, timeout=10, seed=0, backend=backend)

    assert index in (1, 3, 5)


def test_dh_constant(backend):
    # no value is smaller than the threshold, so that nothing has to be searched
    index, napplications = DH([2.0] * 8, backend=backend)

    assert 0 <= index < 8
    assert napplications == 0


def test_dh_timeout(backend):
    values = _table(backend, 8, seed=5)

    # it is interrupted after the first measurement, or it does not need any
    index, napplications = DH(values, timeout=1e-9, seed=0, backend=backend)

    assert 0 <= index < 8
    assert napplications in (0, 1)


def test_dh_generator(backend):
    values = _table(backend, 8, seed=6)

    backend.set_seed(0)
    index, _ = DH(values, timeout=10, seed=np.random.default_rng(0), backend=backend)

    assert float(values[index]) == min(float(value) for value in values)


def test_dh_errors(backend):
    for values in ([], [1.0], [1.0, 2.0, 3.0], [[1.0, 2.0], [3.0, 4.0]]):
        with pytest.raises(ValueError):
            DH(values, backend=backend)

    for growth in (1.0, 2.0, 0.5, True, "a"):
        with pytest.raises(ValueError):
            DH([1.0, 2.0], growth=growth, backend=backend)

    for timeout in (0, -1, True, "a"):
        with pytest.raises(ValueError):
            DH([1.0, 2.0], timeout=timeout, backend=backend)
