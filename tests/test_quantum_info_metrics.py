import numpy as np
import pytest

from qibo.quantum_info import *


def test_purity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        purity(state, backend=backend)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state, backend=backend), 1.0)

    state = np.outer(np.conj(state), state)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state, backend=backend), 1.0)

    dim = 4
    state = backend.identity_density_matrix(2)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state, backend=backend), 1.0 / dim)


def test_entropy_errors(backend):
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        entropy(state, 0, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        entropy(state, backend=backend)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_entropy(backend, base):
    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(entropy(state, backend=backend), 0.0)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = np.outer(state, state)
    state = backend.cast(state, dtype=state.dtype)

    nqubits = 2
    state = backend.identity_density_matrix(nqubits)
    state = backend.cast(state, dtype=state.dtype)
    if base == 2:
        backend.assert_allclose(entropy(state, base, backend=backend), 2.0)
        backend.assert_allclose(
            entropy(state, base, validate=True, backend=backend), 2.0
        )
    elif base == 10:
        backend.assert_allclose(
            entropy(state, base, backend=backend), 0.6020599913279624
        )
        backend.assert_allclose(
            entropy(state, base, validate=True, backend=backend), 0.6020599913279624
        )
    elif base == np.e:
        backend.assert_allclose(
            entropy(state, base, backend=backend), 1.3862943611198906
        )
        backend.assert_allclose(
            entropy(state, base, validate=True, backend=backend), 1.3862943611198906
        )
    else:
        backend.assert_allclose(
            entropy(state, base, backend=backend), 0.8613531161467861
        )
        backend.assert_allclose(
            entropy(state, base, validate=True, backend=backend), 0.8613531161467861
        )


def test_trace_distance(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        trace_distance(state, target, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        trace_distance(state, target, backend=backend)
    with pytest.raises(TypeError):
        state = np.array([])
        target = np.array([])
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=state.dtype)
        trace_distance(state, target, backend=backend)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(trace_distance(state, target, backend=backend), 0.0)
    backend.assert_allclose(
        trace_distance(state, target, validate=True, backend=backend), 0.0
    )

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(trace_distance(state, target, backend=backend), 0.0)
    backend.assert_allclose(
        trace_distance(state, target, validate=True, backend=backend), 0.0
    )

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(trace_distance(state, target, backend=backend), 1.0)
    backend.assert_allclose(
        trace_distance(state, target, validate=True, backend=backend), 1.0
    )


def test_hilbert_schmidt_distance(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target, backend=backend)
    with pytest.raises(TypeError):
        state = np.array([])
        target = np.array([])
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target, backend=backend)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        hilbert_schmidt_distance(state, target, backend=backend), 0.0
    )

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        hilbert_schmidt_distance(state, target, backend=backend), 0.0
    )

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        hilbert_schmidt_distance(state, target, backend=backend), 2.0
    )


def test_fidelity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2)
        target = np.random.rand(4, 4)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        fidelity(state, target, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        fidelity(state, target, backend=backend)
    with pytest.raises(ValueError):
        state = np.random.rand(2, 2)
        target = np.random.rand(2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        fidelity(state, target, validate=True, backend=backend)

    state = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target, backend=backend), 1.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target, backend=backend), 1.0)

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target, backend=backend), 0.0)


def test_process_fidelity(backend):
    d = 2
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2, 1)
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        process_fidelity(channel, target, backend=backend)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        channel = backend.cast(channel, dtype=channel.dtype)
        process_fidelity(channel, validate=True, backend=backend)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2)
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        process_fidelity(channel, target, validate=True, backend=backend)

    channel = np.eye(d**2)
    channel = backend.cast(channel, dtype=channel.dtype)
    backend.assert_allclose(process_fidelity(channel, backend=backend), 1.0)
    backend.assert_allclose(process_fidelity(channel, channel, backend=backend), 1.0)
    backend.assert_allclose(average_gate_fidelity(channel, backend=backend), 1.0)
    backend.assert_allclose(
        average_gate_fidelity(channel, channel, backend=backend), 1.0
    )
    backend.assert_allclose(gate_error(channel, backend=backend), 0.0)
    backend.assert_allclose(gate_error(channel, channel, backend=backend), 0.0)
