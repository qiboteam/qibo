import numpy as np
import pytest

from qibo import Circuit, gates
from qibo.config import PRECISION_TOL
from qibo.models.encodings import _generate_rbs_angles, unary_encoder
from qibo.quantum_info.metrics import (
    average_gate_fidelity,
    bures_angle,
    bures_distance,
    diamond_norm,
    expressibility,
    fidelity,
    frame_potential,
    gate_error,
    hilbert_schmidt_distance,
    impurity,
    infidelity,
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
        test = purity(state, backend=backend)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state, backend=backend), 1.0, atol=PRECISION_TOL)
    backend.assert_allclose(impurity(state, backend=backend), 0.0, atol=PRECISION_TOL)

    state = backend.np.outer(backend.np.conj(state), state)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(purity(state, backend=backend), 1.0, atol=PRECISION_TOL)
    backend.assert_allclose(impurity(state, backend=backend), 0.0, atol=PRECISION_TOL)

    dim = 4
    state = backend.identity_density_matrix(2)
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(
        purity(state, backend=backend), 1.0 / dim, atol=PRECISION_TOL
    )
    backend.assert_allclose(
        impurity(state, backend=backend), 1.0 - 1.0 / dim, atol=PRECISION_TOL
    )


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

    state = backend.np.outer(backend.np.conj(state), state)
    target = backend.np.outer(backend.np.conj(target), target)
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
    backend.assert_allclose(
        hilbert_schmidt_distance(state, target, backend=backend), 0.0
    )

    state = backend.np.outer(backend.np.conj(state), state)
    target = backend.np.outer(backend.np.conj(target), target)
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

    state = backend.np.outer(backend.np.conj(state), state)
    target = backend.np.outer(backend.np.conj(target), target)
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


@pytest.mark.parametrize("seed", [10])
def test_process_fidelity_and_infidelity(backend, seed):
    d = 2
    rng = np.random.default_rng(seed)
    with pytest.raises(TypeError):
        channel = rng.random(d**2, d**2)
        target = rng.random(d**2, d**2, d**2)
        channel = backend.cast(channel, dtype=channel.dtype)
        target = backend.cast(target, dtype=target.dtype)
        test = process_fidelity(channel, target, backend=backend)
    with pytest.raises(TypeError):
        channel = random_hermitian(d**2, seed=rng, backend=backend)
        test = process_fidelity(channel, check_unitary=True, backend=backend)
    with pytest.raises(TypeError):
        channel = 10 * rng.random(d**2, d**2)
        target = 10 * rng.random(d**2, d**2)
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


@pytest.mark.skip
@pytest.mark.parametrize("nqubits", [1, 2])
def test_diamond_norm(backend, nqubits):
    with pytest.raises(TypeError):
        test = random_unitary(2**nqubits, backend=backend)
        test_2 = random_unitary(4**nqubits, backend=backend)
        test = diamond_norm(test, test_2)

    unitary = backend.identity_density_matrix(nqubits, normalize=False)
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
        frame_potential(circuit, power_t="2", backend=backend)
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

        params = _generate_rbs_angles(data, nqubits, "diagonal")
        params = backend.cast(params, dtype=np.float64)

        target = [1]
        for param in params[:-1]:
            elem = float(target[-1] * backend.np.sin(param) ** 2)
            target.append(elem)
        target = 4 * backend.np.diag(backend.cast(target, dtype=np.float64))

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
            test = quantum_fisher_information_matrix(circuit, params, backend=backend)
