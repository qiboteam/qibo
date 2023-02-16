import numpy as np
import pytest

from qibo.quantum_info import *


def test_purity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        purity(state)
    state = np.asarray([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0)

    state = np.outer(np.conj(state), state)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0)

    d = 4
    state = np.eye(d) / d
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0 / d)


def test_entropy_errors(backend):
    with pytest.raises(ValueError):
        state = np.asarray([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        entropy(state, 0)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        entropy(state)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_entropy(backend, base):
    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(entropy(state), 0.0)

    d = 4
    Id = np.eye(d)
    Id = backend.cast(Id, dtype=Id.dtype)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = np.outer(state, state)
    state = backend.cast(state, dtype=state.dtype)

    state = Id / d
    state = backend.cast(state, dtype=state.dtype)
    if base == 2:
        backend.assert_allclose(entropy(state, base), 2.0)
        backend.assert_allclose(entropy(state, base, validate=True), 2.0)
    elif base == 10:
        backend.assert_allclose(entropy(state, base), 0.6020599913279624)
        backend.assert_allclose(entropy(state, base, validate=True), 0.6020599913279624)
    elif base == np.e:
        backend.assert_allclose(entropy(state, base), 1.3862943611198906)
        backend.assert_allclose(entropy(state, base, validate=True), 1.3862943611198906)
    else:
        backend.assert_allclose(entropy(state, base), 0.8613531161467861)
        backend.assert_allclose(entropy(state, base, validate=True), 0.8613531161467861)


def test_trace_distance(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        trace_distance(state, target)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        trace_distance(state, target)
    with pytest.raises(TypeError):
        state = np.asarray([])
        target = np.asarray([])
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=state.dtype)
        trace_distance(state, target)

    state = np.asarray([1.0, 0.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(trace_distance(state, target), 0.0)
    backend.assert_allclose(trace_distance(state, target, validate=True), 0.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(trace_distance(state, target), 0.0)
    backend.assert_allclose(trace_distance(state, target, validate=True), 0.0)

    state = np.asarray([0.0, 1.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(trace_distance(state, target), 1.0)
    backend.assert_allclose(trace_distance(state, target, validate=True), 1.0)


def test_hilbert_schmidt_distance(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target)
    with pytest.raises(TypeError):
        state = np.asarray([])
        target = np.asarray([])
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target)

    state = np.asarray([1.0, 0.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 0.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 0.0)

    state = np.asarray([0.0, 1.0, 0.0, 0.0])
    target = np.asarray([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 2.0)


def test_fidelity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        fidelity(state, target)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        fidelity(state, target)
    with pytest.raises(ValueError):
        state = np.random.rand(2, 2)
        target = np.random.rand(2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        fidelity(state, target, validate=True)

    state = np.asarray([0.0, 0.0, 0.0, 1.0])
    target = np.asarray([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target), 1.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target), 1.0)

    state = np.asarray([0.0, 1.0, 0.0, 0.0])
    target = np.asarray([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target), 0.0)


def test_process_fidelity(backend):
    d = 2
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2, 1)
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        process_fidelity(channel, target)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        channel = backend.cast(channel, dtype=channel.dtype)
        process_fidelity(channel, validate=True)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2)
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        process_fidelity(channel, target, validate=True)

    channel = np.eye(d**2)
    channel = backend.cast(channel, dtype=channel.dtype)
    backend.assert_allclose(process_fidelity(channel), 1.0)
    backend.assert_allclose(process_fidelity(channel, channel), 1.0)
    backend.assert_allclose(average_gate_fidelity(channel), 1.0)
    backend.assert_allclose(average_gate_fidelity(channel, channel), 1.0)
    backend.assert_allclose(gate_error(channel), 0.0)
    backend.assert_allclose(gate_error(channel, channel), 0.0)
