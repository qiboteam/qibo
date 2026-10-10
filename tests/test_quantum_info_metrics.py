import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.config import PRECISION_TOL
from qibo.models._encodings import _generate_rbs_angles
from qibo.models.encodings import unary_encoder
from qibo.quantum_info.metrics import (
    a_fidelity,
    average_gate_fidelity,
    bures_angle,
    bures_distance,
    chen_fidelity,
    diamond_norm,
    expressibility,
    fidelity,
    frame_potential,
    gate_error,
    geometric_mean_fidelity,
    hilbert_schmidt_distance,
    impurity,
    infidelity,
    max_fidelity,
    n_fidelity,
    process_fidelity,
    process_infidelity,
    purity,
    quantum_fisher_information_matrix,
    trace_distance,
)
from qibo.quantum_info.random_ensembles import (
    random_density_matrix,
    random_hermitian,
    random_unitary,
)
from qibo.quantum_info.superoperator_transformations import to_choi


def test_purity_and_impurity(backend):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        purity(state, backend=backend)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state, backend=backend), 1.0, atol=PRECISION_TOL)
    backend.assert_allclose(impurity(state, backend=backend), 0.0, atol=PRECISION_TOL)

    state = backend.outer(backend.conj(state), state)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state, backend=backend), 1.0, atol=PRECISION_TOL)
    backend.assert_allclose(impurity(state, backend=backend), 0.0, atol=PRECISION_TOL)

    dim = 4
    state = backend.maximally_mixed_state(2)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(
        purity(state, backend=backend), 1.0 / dim, atol=PRECISION_TOL
    )
    backend.assert_allclose(
        impurity(state, backend=backend), 1.0 - 1.0 / dim, atol=PRECISION_TOL
    )


def test_trace_distance(backend):
    with pytest.raises(TypeError):
        state = random_density_matrix(2, pure=True, backend=backend)
        target = random_density_matrix(4, pure=True, backend=backend)
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
    backend.assert_allclose(
        trace_distance(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = backend.outer(backend.conj(state), state)
    target = backend.outer(backend.conj(target), target)
    backend.assert_allclose(
        trace_distance(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        trace_distance(state, target, backend=backend),
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
    backend.assert_allclose(
        hilbert_schmidt_distance(state, target, backend=backend), 0.0
    )

    state = backend.outer(backend.conj(state), state)
    target = backend.outer(backend.conj(target), target)
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


def test_fidelity_and_infidelity_and_bures(backend):
    with pytest.raises(TypeError):
        state = random_density_matrix(2, pure=True, backend=backend)
        target = random_density_matrix(4, pure=True, backend=backend)
        fidelity(state, target, backend=backend)
    with pytest.raises(TypeError):
        state = np.random.rand(2, 2, 2)
        target = np.random.rand(2, 2, 2)
        state = backend.cast(state, dtype=state.dtype)
        target = backend.cast(target, dtype=target.dtype)
        fidelity(state, target, backend=backend)

    state = backend.maximally_mixed_state(4)
    target = backend.maximally_mixed_state(4)
    backend.assert_allclose(
        fidelity(state, target, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )

    state = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        fidelity(state, target, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        infidelity(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_angle(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_distance(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = backend.outer(backend.conj(state), state)
    target = backend.outer(backend.conj(target), target)
    backend.assert_allclose(
        fidelity(state, target, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        infidelity(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_angle(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_distance(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )

    state = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0, 1.0])
    state = backend.cast(state, dtype=state.dtype)
    target = backend.cast(target, dtype=target.dtype)
    backend.assert_allclose(
        fidelity(state, target, backend=backend),
        0.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        infidelity(state, target, backend=backend),
        1.0,
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_angle(state, target, backend=backend),
        np.arccos(0.0),
        atol=PRECISION_TOL,
    )
    backend.assert_allclose(
        bures_distance(state, target, backend=backend),
        np.sqrt(2),
        atol=PRECISION_TOL,
    )


@pytest.mark.parametrize("target_pure", [False, True])
@pytest.mark.parametrize("state_pure", [False, True])
@pytest.mark.parametrize("nqubits", [1, 2, 5])
def test_alternative_fidelities(backend, nqubits, state_pure, target_pure):
    dims = 2**nqubits
    state = random_density_matrix(dims, pure=state_pure, seed=10, backend=backend)
    target = random_density_matrix(dims, pure=target_pure, seed=8, backend=backend)

    fid = fidelity(state, target, backend=backend)
    a_fid = a_fidelity(state, target, backend=backend)
    n_fid = n_fidelity(state, target, backend=backend)
    gm_fid = geometric_mean_fidelity(state, target, backend=backend)
    max_fid = max_fidelity(state, target, backend=backend)

    assert backend.round(fid, 10) <= backend.round(n_fid, 10)
    assert backend.round(fid, 10) >= backend.round(a_fid, 10)
    assert backend.round(gm_fid, 10) >= backend.round(max_fid, 10)

    if nqubits == 1:
        chen_fid = chen_fidelity(state, target, backend=backend)

        backend.assert_allclose(chen_fid, fid, atol=1e-6)
        backend.assert_allclose(chen_fid, n_fid, atol=1e-6)


@pytest.mark.parametrize("seed", [10])
def test_process_fidelity_and_infidelity(backend, seed):
    d = 2
    rng = np.random.default_rng(seed)
    with pytest.raises(TypeError):
        channel = rng.random((d**2, d**2))
        target = rng.random((d**2, d))
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        process_fidelity(channel, target, backend=backend)
    with pytest.raises(TypeError):
        channel = rng.random((d**2, d**2))
        target = rng.random((d**2, d))
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        process_infidelity(channel, target, backend=backend)
    with pytest.raises(TypeError):
        channel = random_hermitian(d**2, seed=seed, backend=backend)
        process_fidelity(channel, check_unitary=True, backend=backend)
    with pytest.raises(TypeError):
        channel = 10 * rng.random((d**2, d**2))
        target = 10 * rng.random((d**2, d**2))
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        process_fidelity(channel, target, check_unitary=True, backend=backend)

    channel = backend.identity(d**2)

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


@pytest.mark.skip
@pytest.mark.parametrize("nqubits", [1, 2])
def test_diamond_norm(backend, nqubits):
    with pytest.raises(TypeError):
        test = random_unitary(2**nqubits, backend=backend)
        test_2 = random_unitary(4**nqubits, backend=backend)
        test = diamond_norm(test, test_2)

    unitary = backend.identity(2**nqubits)
    unitary = to_choi(unitary, order="row", backend=backend)

    dnorm = diamond_norm(unitary, backend=backend)
    backend.assert_allclose(dnorm, 1.0, atol=PRECISION_TOL)

    dnorm = diamond_norm(unitary, unitary, backend=backend)
    backend.assert_allclose(dnorm, 0.0, atol=PRECISION_TOL)


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
    samples = 100
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


@pytest.mark.parametrize("samples", [int(1e1)])
@pytest.mark.parametrize("power_t", [2])
@pytest.mark.parametrize("nqubits", [2, 3])
def test_frame_potential(backend, nqubits, power_t, samples):
    depth = int(np.ceil(nqubits * power_t))

    circuit = Circuit(nqubits)
    circuit.add(gates.U3(q, 0.0, 0.0, 0.0) for q in range(nqubits))
    for _ in range(depth):
        circuit.add(gates.CNOT(q, q + 1) for q in range(nqubits - 1))
        circuit.add(gates.U3(q, 0.0, 0.0, 0.0) for q in range(nqubits))

    with pytest.raises(TypeError):
        frame_potential(circuit, power_t="2", samples=10, backend=backend)
    with pytest.raises(TypeError):
        frame_potential(circuit, 2, samples="1000", backend=backend)

    dim = 2**nqubits
    potential_haar = 2 / dim**4

    potential = frame_potential(
        circuit, power_t=power_t, samples=samples, backend=backend
    )

    backend.assert_allclose(potential, potential_haar, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("params_flag", [None, True])
@pytest.mark.parametrize("return_complex", [False, True])
@pytest.mark.parametrize("nqubits", [4, 8])
def test_qfim(backend, nqubits, return_complex, params_flag):
    if backend.platform in ["tensorflow", "pytorch"]:
        # QFIM from https://arxiv.org/abs/2405.20408 is known analytically
        data = np.random.rand(nqubits)
        data = backend.cast(data, dtype=data.dtype)

        params = _generate_rbs_angles(
            data, dims=nqubits, architecture="diagonal", backend=backend
        )

        target = [1]
        for param in params[:-1]:
            elem = float(target[-1] * backend.sin(param) ** 2)
            target.append(elem)
        target = 4 * backend.diag(backend.cast(target, dtype=np.float64))

        # numerical qfim from quantum_info
        circuit = unary_encoder(data, "diagonal")

        if params_flag is not None:
            circuit.set_parameters(params)
        else:
            params = params_flag

        qfim = quantum_fisher_information_matrix(
            circuit, params, return_complex=return_complex, backend=backend
        )

        backend.assert_allclose(qfim, target, atol=1e-6)
    else:
        circuit = Circuit(nqubits)
        params = np.random.rand(3)
        params = backend.cast(params, dtype=params.dtype)
        with pytest.raises(NotImplementedError):
            quantum_fisher_information_matrix(circuit, params, backend=backend)


def test_trace_distance_is_real(backend):
    state = random_density_matrix(4, seed=1, backend=backend)
    target = random_density_matrix(4, seed=2, backend=backend)

    distance = backend.to_numpy(trace_distance(state, target, backend=backend))

    assert not np.iscomplexobj(distance)
    reference = (
        0.5
        * np.linalg.svd(
            backend.to_numpy(state) - backend.to_numpy(target), compute_uv=False
        ).sum()
    )
    np.testing.assert_allclose(distance, reference, atol=1e-10)


@pytest.mark.parametrize("dims", [2, 4, 8])
def test_bures_identical_states(backend, dims):
    # the fidelity of identical states can exceed one by rounding errors
    for seed in range(20):
        state = random_density_matrix(dims, seed=seed, backend=backend)

        angle = float(bures_angle(state, state, backend=backend))
        distance = float(bures_distance(state, state, backend=backend))

        assert not np.isnan(angle)
        assert not np.isnan(distance)
        assert angle < 1e-6
        assert distance < 1e-6


@pytest.mark.parametrize("dims", [2, 4])
def test_average_gate_fidelity_unitary(backend, dims):
    unitary = backend.to_numpy(random_unitary(dims, seed=3, backend=backend))
    liouville = backend.cast(np.kron(unitary, unitary.conj()))
    identity = backend.identity(dims**2, dtype=backend.complex128)

    # F_avg = (|tr(U)|^2 + d) / (d (d + 1)) for the identity target
    target = (abs(np.trace(unitary)) ** 2 + dims) / (dims * (dims + 1))

    average = average_gate_fidelity(liouville, identity, backend=backend)
    backend.assert_allclose(average, target, atol=1e-10)
    backend.assert_allclose(
        gate_error(liouville, identity, backend=backend), 1 - target, atol=1e-10
    )


def test_average_gate_fidelity_depolarizing(backend):
    pauli = [
        np.eye(2),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.diag([1, -1]),
    ]
    probability = 0.2
    kraus = [np.sqrt(1 - probability) * pauli[0]] + [
        np.sqrt(probability / 3) * matrix for matrix in pauli[1:]
    ]
    liouville = backend.cast(sum(np.kron(k, k.conj()) for k in kraus))
    identity = backend.identity(4, dtype=backend.complex128)

    # F_pro = 1 - p, and F_avg = (d F_pro + 1) / (d + 1) with d = 2
    process = 1 - probability
    backend.assert_allclose(
        process_fidelity(liouville, identity, backend=backend), process, atol=1e-10
    )
    backend.assert_allclose(
        average_gate_fidelity(liouville, identity, backend=backend),
        (2 * process + 1) / 3,
        atol=1e-10,
    )


@pytest.mark.parametrize("function", [expressibility, frame_potential])
@pytest.mark.parametrize("arguments", [(0, 10), (1, 0), (-1, 10), (1, -3)])
def test_circuit_metrics_non_positive_arguments(backend, function, arguments):
    circuit = Circuit(1)
    circuit.add(gates.RX(0, 0.1, trainable=True))

    with pytest.raises(ValueError):
        function(circuit, *arguments, backend=backend)


def test_diamond_norm_does_not_modify_inputs(backend):
    pytest.importorskip("cvxpy")

    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    channel = backend.cast(np.kron(pauli_x, pauli_x.conj()))
    target = backend.identity(4, dtype=backend.complex128)
    channel_copy = backend.cast(channel, copy=True)
    target_copy = backend.cast(target, copy=True)

    norm = diamond_norm(channel, target, backend=backend)

    # the diamond distance between the identity and a bit flip is two
    np.testing.assert_allclose(norm, 2.0, atol=1e-4)
    backend.assert_allclose(channel, channel_copy)
    backend.assert_allclose(target, target_copy)
