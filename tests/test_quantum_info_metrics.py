import numpy as np
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.quantum_info import *


def test_purity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        purity(state)
    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0)

    state = np.outer(np.conj(state), state)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0)

    dim = 4
    state = backend.identity_density_matrix(2)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0 / dim)


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

    if backend.__class__.__name__ == "CupyBackend":
        state = np.array([[1, 1e-3], [0, 0]])
        with pytest.raises(NotImplementedError):
            entropy(state, base, validate=True, backend=backend)


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
        hilbert_schmidt_distance(
            state,
            target,
        )
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target)
    with pytest.raises(TypeError):
        state = np.array([])
        target = np.array([])
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        hilbert_schmidt_distance(state, target)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 0.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(hilbert_schmidt_distance(state, target), 0.0)

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
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

    state = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target), 1.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target), 1.0)

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
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


def test_meyer_wallach_entanglement(backend):
    from qibo.gates import CNOT, RX
    from qibo.models import Circuit

    nqubits = 2
    circuit1 = Circuit(nqubits)
    circuit1.add([RX(0, np.pi / 4, trainable=True)] for _ in range(nqubits))
    circuit2 = Circuit(nqubits)
    circuit2.add([RX(0, np.pi / 4, trainable=True)] for _ in range(nqubits))
    circuit2.add(CNOT(0, 1))
    backend.assert_allclose(meyer_wallach_entanglement(circuit1, backend=backend), 0)
    backend.assert_allclose(meyer_wallach_entanglement(circuit2, backend=backend), 0.5)


def test_entangling_capability(backend):
    with pytest.raises(TypeError):
        circuit = Circuit(1)
        samples = 0.5
        entangling_capability(circuit, samples, backend=backend)

    nqubits = 2
    samples = 500

    c1 = Circuit(nqubits)
    c1.add([gates.RX(q, 0, trainable=True) for q in range(nqubits)])
    c1.add(gates.CNOT(0, 1))
    c1.add([gates.RX(q, 0, trainable=True) for q in range(nqubits)])
    ent_mw1 = entangling_capability(c1, samples, backend=backend)

    c2 = Circuit(nqubits)
    c2.add(gates.H(0))
    c2.add(gates.CNOT(0, 1))
    c2.add(gates.RX(0, 0, trainable=True))
    ent_mw2 = entangling_capability(c2, samples, backend=backend)

    c3 = Circuit(nqubits)
    ent_mw3 = entangling_capability(c3, samples, backend=backend)

    backend.assert_allclose(ent_mw3 < ent_mw1 < ent_mw2, True)


def test_expressibility(backend):
    with pytest.raises(TypeError):
        circuit = Circuit(1)
        t = 0.5
        samples = 10
        expressibility(circuit, t, samples, backend=backend)
    with pytest.raises(TypeError):
        circuit = Circuit(1)
        t = 1
        samples = 0.5
        expressibility(circuit, t, samples, backend=backend)

    nqubits = 2
    samples = 500
    t = 1

    c1 = Circuit(nqubits)
    c1.add([gates.RX(q, 0, trainable=True) for q in range(nqubits)])
    c1.add(gates.CNOT(0, 1))
    c1.add([gates.RX(q, 0, trainable=True) for q in range(nqubits)])
    expr_1 = expressibility(c1, t, samples, backend=backend)

    c2 = Circuit(nqubits)
    c2.add(gates.H(0))
    c2.add(gates.CNOT(0, 1))
    c2.add(gates.RX(0, 0, trainable=True))
    expr_2 = expressibility(c2, t, samples, backend=backend)

    c3 = Circuit(nqubits)
    expr_3 = expressibility(c3, t, samples, backend=backend)

    backend.assert_allclose(expr_1 < expr_2 < expr_3, True)
