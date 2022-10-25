# -*- coding: utf-8 -*-
import numpy as np
import pytest

from qibo.quantum_info import *


def test_shannon_entropy_errors():
    with pytest.raises(ValueError):
        p = np.asarray([1.0, 0.0])
        shannon_entropy(p, -2)
    with pytest.raises(TypeError):
        p = np.asarray([[1.0], [0.0]])
        shannon_entropy(p)
    with pytest.raises(TypeError):
        p = np.asarray([])
        shannon_entropy(p)
    with pytest.raises(ValueError):
        p = np.asarray([1.0, -1.0])
        shannon_entropy(p)
    with pytest.raises(ValueError):
        p = np.asarray([1.1, 0.0])
        shannon_entropy(p)
    with pytest.raises(ValueError):
        p = np.asarray([0.5, 0.4999999])
        shannon_entropy(p)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_shannon_entropy(backend, base):
    prob_array = np.asarray([1.0, 0.0])
    result = shannon_entropy(prob_array, base)
    backend.assert_allclose(result, 0.0)

    if base == 2:
        prob_array = np.asarray([0.5, 0.5])
        result = shannon_entropy(prob_array, base)
        backend.assert_allclose(result, 1.0)


def test_hellinger_distance(backend):
    with pytest.raises(TypeError):
        p = np.random.rand(1, 2)
        q = np.random.rand(1, 5)
        hellinger_distance(p, q)
    with pytest.raises(TypeError):
        p = np.random.rand(1, 2)[0]
        q = np.array([])
        hellinger_distance(p, q)
    with pytest.raises(ValueError):
        p = np.array([-1, 2.0])
        q = np.random.rand(1, 5)[0]
        hellinger_distance(p, q, validate=True)
    with pytest.raises(ValueError):
        p = np.random.rand(1, 2)[0]
        q = np.array([1.0, 0.0])
        hellinger_distance(p, q, validate=True)
    with pytest.raises(ValueError):
        p = np.array([1.0, 0.0])
        q = np.random.rand(1, 2)[0]
        hellinger_distance(p, q, validate=True)

    p = np.array([1.0, 0.0])
    q = np.array([1.0, 0.0])
    backend.assert_allclose(hellinger_distance(p, q), 0.0)


def test_purity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        purity(state)
    state = np.asarray([1.0, 0.0, 0.0, 0.0])
    backend.assert_allclose(purity(state), 1.0)

    state = np.outer(np.conj(state), state)
    backend.assert_allclose(purity(state), 1.0)

    d = 4
    state = np.eye(d) / d
    backend.assert_allclose(purity(state), 1.0 / d)


def test_trace_distance(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        trace_distance(state, target)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        trace_distance(state, target)
    with pytest.raises(TypeError):
        state = np.asarray([])
        target = np.asarray([])
        trace_distance(state, target)

    state = np.asarray([1.0, 0.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    backend.assert_allclose(trace_distance(state, target), 0.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    backend.assert_allclose(trace_distance(state, target), 0.0)

    state = np.asarray([0.0, 1.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    backend.assert_allclose(trace_distance(state, target), 1.0)


def test_hilbert_schmidt_distance(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        hilbert_schmidt_distance(state, target)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        hilbert_schmidt_distance(state, target)
    with pytest.raises(TypeError):
        state = np.asarray([])
        target = np.asarray([])
        hilbert_schmidt_distance(state, target)

    state = np.asarray([1.0, 0.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 0.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 0.0)

    state = np.asarray([0.0, 1.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 2.0)


def test_fidelity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        fidelity(state, target)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        fidelity(state, target)
    with pytest.raises(ValueError):
        state = np.random.rand(2, 2)
        target = np.random.rand(2, 2)
        fidelity(state, target, validate=True)

    state = np.asarray([0.0, 0.0, 0.0, 1.0])
    target = np.asarray([0.0, 0.0, 0.0, 1.0])
    backend.assert_allclose(fidelity(state, target), 1.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    backend.assert_allclose(fidelity(state, target), 1.0)

    state = np.asarray([0.0, 1.0, 0.0, 0.0])
    target = np.asarray([0.0, 0.0, 0.0, 1.0])
    backend.assert_allclose(fidelity(state, target), 0.0)


def test_process_fidelity(backend):
    d = 2
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2, 1)
        process_fidelity(channel, target)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        process_fidelity(channel, validate=True)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2)
        process_fidelity(channel, target, validate=True)

    channel = np.eye(d**2)
    backend.assert_allclose(process_fidelity(channel), 1.0)
    backend.assert_allclose(process_fidelity(channel, channel), 1.0)
    backend.assert_allclose(average_gate_fidelity(channel), 1.0)
    backend.assert_allclose(average_gate_fidelity(channel, channel), 1.0)