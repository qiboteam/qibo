import numpy as np
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.quantum_info import *


def test_purity_and_impurity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = purity(state)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0, atol=PRECISION_TOL)
    backend.assert_allclose(impurity(state), 0.0, atol=PRECISION_TOL)

    state = np.outer(np.conj(state), state)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0, atol=PRECISION_TOL)
    backend.assert_allclose(impurity(state), 0.0, atol=PRECISION_TOL)

    dim = 4
    state = backend.identity_density_matrix(2)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state), 1.0 / dim, atol=PRECISION_TOL)
    backend.assert_allclose(impurity(state), 1.0 - 1.0 / dim, atol=PRECISION_TOL)


@pytest.mark.parametrize("check_purity", [True, False])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("bipartition", [[0], [1]])
def test_concurrence_and_formation(backend, bipartition, base, check_purity):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = concurrence(
            state, bipartition=bipartition, check_purity=check_purity, backend=backend
        )
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        test = concurrence(
            state, bipartition=bipartition, check_purity="True", backend=backend
        )

    if check_purity is True:
        with pytest.raises(NotImplementedError):
            state = backend.identity_density_matrix(2, normalize=False)
            test = concurrence(state, bipartition=bipartition, backend=backend)

    nqubits = 2
    dim = 2**nqubits
    state = random_statevector(dim, backend=backend)
    concur = concurrence(
        state, bipartition=bipartition, check_purity=check_purity, backend=backend
    )
    ent_form = entanglement_of_formation(
        state,
        bipartition=bipartition,
        base=base,
        check_purity=check_purity,
        backend=backend,
    )
    backend.assert_allclose(0.0 <= concur <= np.sqrt(2), True)
    backend.assert_allclose(0.0 <= ent_form <= 1.0, True)

    state = np.kron(
        random_density_matrix(2, pure=True, backend=backend),
        random_density_matrix(2, pure=True, backend=backend),
    )
    concur = concurrence(state, bipartition, check_purity=check_purity, backend=backend)
    ent_form = entanglement_of_formation(
        state,
        bipartition=bipartition,
        base=base,
        check_purity=check_purity,
        backend=backend,
    )
    backend.assert_allclose(concur, 0.0, atol=10 * PRECISION_TOL)
    backend.assert_allclose(ent_form, 0.0, atol=PRECISION_TOL)


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_entropy(backend, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(
            state, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, base=0, check_hermitian=check_hermitian, backend=backend)
    with pytest.raises(TypeError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, base=base, check_hermitian="False", backend=backend)

    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            state = random_unitary(4, backend=backend)
            test = entropy(state, base=base, check_hermitian=True, backend=backend)
    else:
        state = random_unitary(4, backend=backend)
        test = entropy(state, base=base, check_hermitian=True, backend=backend)

    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(entropy(state, backend=backend), 0.0, atol=PRECISION_TOL)

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
            entropy(state, base, check_hermitian=check_hermitian, backend=backend)
            - test
        )
        < PRECISION_TOL,
        True,
    )


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("bipartition", [[0], [1]])
def test_entanglement_entropy(backend, bipartition, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=base,
            check_hermitian=check_hermitian,
            backend=backend,
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=0,
            check_hermitian=check_hermitian,
            backend=backend,
        )
    if backend.__class__.__name__ == "CupyBackend":
        with pytest.raises(NotImplementedError):
            state = random_unitary(4, backend=backend)
            test = entanglement_entropy(
                state,
                bipartition=bipartition,
                base=base,
                check_hermitian=True,
                backend=backend,
            )

    # Bell state
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    state = backend.cast(state, dtype=state.dtype)

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        check_hermitian=check_hermitian,
        backend=backend,
    )

    if base == 2:
        test = 1.0
    elif base == 10:
        test = 0.30102999566398125
    elif base == np.e:
        test = 0.6931471805599454
    else:
        test = 0.4306765580733931

    backend.assert_allclose(entang_entrop, test, atol=PRECISION_TOL)

    # Product state
    state = np.kron(
        random_statevector(2, backend=backend), random_statevector(2, backend=backend)
    )

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        check_hermitian=check_hermitian,
        backend=backend,
    )

    backend.assert_allclose(entang_entrop, 0.0, atol=PRECISION_TOL)


@pytest.mark.parametrize("check_hermitian", [False, True])
def test_trace_distance(backend, check_hermitian):
    with pytest.raises(TypeError):
        state = random_density_matrix(2, pure=True, backend=backend)
        target = random_density_matrix(4, pure=True, backend=backend)
        test = trace_distance(
            state, target, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        test = trace_distance(
            state, target, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = np.array([])
        target = np.array([])
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=state.dtype)
        test = trace_distance(
            state, target, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = random_density_matrix(2, pure=True, backend=backend)
        target = random_density_matrix(2, pure=True, backend=backend)
        test = trace_distance(state, target, check_hermitian="True", backend=backend)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        trace_distance(state, target, check_hermitian=check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        trace_distance(state, target, check_hermitian=check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        trace_distance(state, target, check_hermitian=check_hermitian, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )


def test_hilbert_schmidt_distance(backend):
    with pytest.raises(TypeError):
        state = random_density_matrix(2, pure=True, backend=backend)
        target = random_density_matrix(4, pure=True, backend=backend)
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


@pytest.mark.parametrize("check_hermitian", [True, False])
def test_fidelity_and_infidelity_and_bures(backend, check_hermitian):
    with pytest.raises(TypeError):
        state = random_density_matrix(2, pure=True, backend=backend)
        target = random_density_matrix(4, pure=True, backend=backend)
        test = fidelity(state, target, check_hermitian=check_hermitian, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        test = fidelity(state, target, check_hermitian, backend=backend)
    with pytest.raises(TypeError):
        state = random_density_matrix(2, pure=True, backend=backend)
        target = random_density_matrix(2, pure=True, backend=backend)
        test = fidelity(state, target, check_hermitian="True", backend=backend)

    state = backend.identity_density_matrix(4)
    target = backend.identity_density_matrix(4)
    backend.assert_allclose(
        fidelity(state, target, check_hermitian, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )

    state = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        fidelity(state, target, check_hermitian, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        infidelity(state, target, check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_angle(state, target, check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_distance(state, target, check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = np.outer(np.conj(state), state)
    target = np.outer(np.conj(target), target)
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        fidelity(state, target, check_hermitian, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        infidelity(state, target, check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_angle(state, target, check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_distance(state, target, check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        fidelity(state, target, check_hermitian, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        infidelity(state, target, check_hermitian, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_angle(state, target, check_hermitian, backend=backend),
        np.arccos(0.0),
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_distance(state, target, check_hermitian, backend=backend),
        np.sqrt(2),
        atol=PRECISION_TOL,
    )

    state = random_unitary(4, backend=backend)
    target = random_unitary(4, backend=backend)
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            test = fidelity(state, target, check_hermitian=True, backend=backend)
    else:
        test = fidelity(state, target, check_hermitian=True, backend=backend)


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("nqubits", [4, 6])
@pytest.mark.parametrize("channel", [gates.DepolarizingChannel])
def test_entanglement_fidelity(backend, channel, nqubits, check_hermitian):
    with pytest.raises(TypeError):
        test = entanglement_fidelity(
            channel, nqubits=[0], check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        test = entanglement_fidelity(
            channel, nqubits=0, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3, 2)
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_fidelity(
            channel,
            nqubits,
            state=state,
            check_hermitian=check_hermitian,
            backend=backend,
        )
    with pytest.raises(TypeError):
        state = random_statevector(2, backend=backend)
        test = entanglement_fidelity(
            channel, nqubits, state=state, check_hermitian="False", backend=backend
        )

    channel = channel([0, 1], 0.5)

    # test on maximally entangled state
    ent_fid = entanglement_fidelity(
        channel, nqubits=nqubits, check_hermitian=check_hermitian, backend=backend
    )
    backend.assert_allclose(ent_fid, 0.625, atol=PRECISION_TOL)

    # test with a state vector
    state = backend.plus_state(nqubits)
    ent_fid = entanglement_fidelity(
        channel,
        nqubits=nqubits,
        state=state,
        check_hermitian=check_hermitian,
        backend=backend,
    )
    backend.assert_allclose(ent_fid, 0.625, atol=PRECISION_TOL)

    # test on maximally mixed state
    state = backend.identity_density_matrix(nqubits)
    ent_fid = entanglement_fidelity(
        channel,
        nqubits=nqubits,
        state=state,
        check_hermitian=check_hermitian,
        backend=backend,
    )
    backend.assert_allclose(ent_fid, 1.0, atol=PRECISION_TOL)


def test_process_fidelity_and_infidelity(backend):
    d = 2
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2, 1)
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        test = process_fidelity(channel, target, backend=backend)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        channel = backend.cast(channel, dtype=channel.dtype)
        test = process_fidelity(channel, check_unitary=True, backend=backend)
    with pytest.raises(TypeError):
        channel = np.random.rand(d**2, d**2)
        target = np.random.rand(d**2, d**2)
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        test = process_fidelity(channel, target, check_unitary=True, backend=backend)

    channel = np.eye(d**2)
    channel = backend.cast(channel, dtype=channel.dtype)

    backend.assert_allclose(
        process_fidelity(channel, backend=backend), 1.0, atol=PRECISION_TOL
    )
    backend.assert_allclose(
        process_infidelity(channel, backend=backend), 0.0, atol=PRECISION_TOL
    )

    backend.assert_allclose(
        process_fidelity(channel, channel, backend=backend), 1.0, atol=PRECISION_TOL
    )
    backend.assert_allclose(
        process_infidelity(channel, channel, backend=backend), 0.0, atol=PRECISION_TOL
    )

    backend.assert_allclose(
        average_gate_fidelity(channel, backend=backend), 1.0, atol=PRECISION_TOL
    )
    backend.assert_allclose(
        average_gate_fidelity(channel, channel, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        gate_error(channel, backend=backend), 0.0, atol=PRECISION_TOL
    )
    backend.assert_allclose(
        gate_error(channel, channel, backend=backend), 0.0, atol=PRECISION_TOL
    )


def test_meyer_wallach_entanglement(backend):
    nqubits = 2

    circuit1 = Circuit(nqubits)
    circuit1.add([gates.RX(0, np.pi / 4)] for _ in range(nqubits))

    circuit2 = Circuit(nqubits)
    circuit2.add([gates.RX(0, np.pi / 4)] for _ in range(nqubits))
    circuit2.add(gates.CNOT(0, 1))

    backend.assert_allclose(
        meyer_wallach_entanglement(circuit1, backend=backend), 0.0, atol=PRECISION_TOL
    )

    backend.assert_allclose(
        meyer_wallach_entanglement(circuit2, backend=backend), 0.5, atol=PRECISION_TOL
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
