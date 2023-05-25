import numpy as np
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.quantum_info import *


def test_purity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = purity(state)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0)

    state = np.outer(np.conj(state), state)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0)

    dim = 4
    state = backend.identity_density_matrix(2)
    backend.assert_allclose(purity(state), 1.0 / dim)


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("bipartition", [[0], [1]])
def test_concurrence_and_formation(backend, bipartition, base, validate):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = concurrence(
            state, bipartition=bipartition, validate=validate, backend=backend
        )

    if validate:
        with pytest.raises(NotImplementedError):
            state = backend.identity_density_matrix(2, normalize=False)
            test = concurrence(state, bipartition=bipartition, backend=backend)

    nqubits = 2
    dim = 2**nqubits
    state = random_statevector(dim, backend=backend)
    concur = concurrence(
        state, bipartition=bipartition, validate=validate, backend=backend
    )
    ent_form = entanglement_of_formation(
        state, bipartition=bipartition, base=base, validate=validate, backend=backend
    )
    backend.assert_allclose(0.0 <= concur <= np.sqrt(2), True)
    backend.assert_allclose(0.0 <= ent_form <= 1.0, True)

    state = np.kron(
        random_density_matrix(2, pure=True, backend=backend),
        random_density_matrix(2, pure=True, backend=backend),
    )
    concur = concurrence(state, bipartition, validate=validate, backend=backend)
    ent_form = entanglement_of_formation(
        state, bipartition=bipartition, base=base, validate=validate, backend=backend
    )
    backend.assert_allclose(concur, 0.0, atol=10 * PRECISION_TOL)
    backend.assert_allclose(ent_form, 0.0, atol=PRECISION_TOL)


def test_entropy_errors(backend):
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, 0, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, backend=backend)
    if backend.__class__.__name__ == "CupyBackend":
        with pytest.raises(NotImplementedError):
            state = random_unitary(4)
            state = backend.cast(state, dtype=state.dtype)
            test = entropy(state, validate=True, backend=backend)


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_entropy(backend, base, validate):
    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(entropy(state, backend=backend), 0.0)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = np.outer(state, state)
    state = backend.cast(state, dtype=state.dtype)

    nqubits = 2
    state = backend.identity_density_matrix(nqubits)
    if base == 2:
        test = 2.0
    elif base == 10:
        test = 0.6020599913279624
    elif base == np.e:
        test = 1.3862943611198906
    else:
        test = 0.8613531161467861

    backend.assert_allclose(
        backend.calculate_norm(
            entropy(state, base, validate=validate, backend=backend) - test
        )
        < PRECISION_TOL,
        True,
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


def test_fidelity_and_bures(backend):
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
    backend.assert_allclose(bures_angle(state, target), 0.0)
    backend.assert_allclose(bures_distance(state, target), 0.0)

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target), 1.0)
    backend.assert_allclose(bures_angle(state, target), 0.0)
    backend.assert_allclose(bures_distance(state, target), 0.0)

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(fidelity(state, target), 0.0)
    backend.assert_allclose(bures_angle(state, target), np.arccos(0.0))
    backend.assert_allclose(bures_distance(state, target), np.sqrt(2))


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
    nqubits = 2

    circuit1 = Circuit(nqubits)
    circuit1.add([gates.RX(0, np.pi / 4)] for _ in range(nqubits))

    circuit2 = Circuit(nqubits)
    circuit2.add([gates.RX(0, np.pi / 4)] for _ in range(nqubits))
    circuit2.add(gates.CNOT(0, 1))

    backend.assert_allclose(
        meyer_wallach_entanglement(circuit1, backend=backend) < PRECISION_TOL, True
    )

    backend.assert_allclose(
        abs(meyer_wallach_entanglement(circuit2, backend=backend) - 0.5)
        < PRECISION_TOL,
        True,
    )


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
