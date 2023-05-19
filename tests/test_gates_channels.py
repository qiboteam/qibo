"""Test channels defined in `qibo/gates.py`."""
import numpy as np
import pytest

from qibo import gates, matrices
from qibo.config import PRECISION_TOL
from qibo.quantum_info import (
    random_density_matrix,
    random_statevector,
    random_stochastic_matrix,
)


def test_general_channel(backend):
    a1 = np.sqrt(0.4) * matrices.X
    a2 = np.sqrt(0.6) * np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    )
    initial_rho = random_density_matrix(2**2, backend=backend)
    m1 = np.kron(np.eye(2), backend.to_numpy(a1))
    m1 = backend.cast(m1, dtype=m1.dtype)
    m2 = backend.cast(a2, dtype=a2.dtype)
    target_rho = np.dot(np.dot(m1, initial_rho), np.transpose(np.conj(m1)))
    target_rho += np.dot(np.dot(m2, initial_rho), np.transpose(np.conj(m2)))

    channel1 = gates.KrausChannel([(1,), (0, 1)], [a1, a2])
    assert channel1.target_qubits == (0, 1)
    final_rho = backend.apply_channel_density_matrix(channel1, np.copy(initial_rho), 2)
    backend.assert_allclose(final_rho, target_rho)

    a1 = gates.Unitary(a1, 1)
    a2 = gates.Unitary(a2, 0, 1)
    channel2 = gates.KrausChannel([(1,), (0, 1)], [a1, a2])
    assert channel2.target_qubits == (0, 1)
    final_rho = backend.apply_channel_density_matrix(channel2, np.copy(initial_rho), 2)
    backend.assert_allclose(final_rho, target_rho)

    with pytest.raises(NotImplementedError):
        channel1.on_qubits({})
    with pytest.raises(NotImplementedError):
        state = random_statevector(2**2, backend=backend)
        channel1.apply(backend, state, 2)


def test_controlled_by_channel_error():
    with pytest.raises(ValueError):
        gates.PauliNoiseChannel(0, [("X", 0.5)]).controlled_by(1)

    a1 = np.sqrt(0.4) * matrices.X
    a2 = np.sqrt(0.6) * np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    )
    config = ([(1,), (0, 1)], [a1, a2])
    with pytest.raises(ValueError):
        gates.KrausChannel(*config).controlled_by(1)


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
def test_kraus_channel(backend, pauli_order):
    a1 = np.sqrt(0.4) * matrices.X
    a2 = np.sqrt(0.6) * matrices.Z

    with pytest.raises(TypeError):
        gates.KrausChannel("0", [a1])
    with pytest.raises(TypeError):
        gates.KrausChannel([0, 1], [a1, a2])
    with pytest.raises(ValueError):
        gates.KrausChannel((0, 1), [a1])

    # asserting that both initialisations
    # yield the same channel
    old = gates.KrausChannel([((0,), a1)])
    new = gates.KrausChannel([(0,)], [a1])
    backend.assert_allclose(
        backend.calculate_norm(
            old.to_choi(backend=backend) - new.to_choi(backend=backend)
        )
        < PRECISION_TOL,
        True,
    )

    test_superop = np.array(
        [
            [0.6 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.4 + 0.0j],
            [0.0 + 0.0j, -0.6 + 0.0j, 0.4 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.4 + 0.0j, -0.6 + 0.0j, 0.0 + 0.0j],
            [0.4 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.6 + 0.0j],
        ]
    )
    test_choi = np.reshape(test_superop, [2] * 4).swapaxes(0, 3).reshape([4, 4])

    pauli_elements = {"I": 2.0, "X": -0.4, "Y": -2.0, "Z": 0.4}
    test_pauli = np.diag([pauli_elements[p] for p in pauli_order])

    test_superop = backend.cast(test_superop, dtype=test_superop.dtype)
    test_choi = backend.cast(test_choi, dtype=test_choi.dtype)
    test_pauli = backend.cast(test_pauli, dtype=test_pauli.dtype)

    channel = gates.KrausChannel(0, [a1, a2])

    backend.assert_allclose(
        backend.calculate_norm(channel.to_liouville(backend=backend) - test_superop)
        < PRECISION_TOL,
        True,
    )
    backend.assert_allclose(
        backend.calculate_norm(channel.to_choi(backend=backend) - test_choi)
        < PRECISION_TOL,
        True,
    )
    backend.assert_allclose(
        backend.calculate_norm(
            channel.to_pauli_liouville(pauli_order=pauli_order, backend=backend)
            - test_pauli
        )
        < PRECISION_TOL,
        True,
    )

    gates.DepolarizingChannel((0, 1), 0.98).to_choi()


def test_unitary_channel(backend):
    a1 = matrices.X
    a2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    qubits = [(0,), (2, 3)]
    probabilities = [0.4, 0.3]
    matrices_ = list(zip(probabilities, [a1, a2]))

    channel = gates.UnitaryChannel(qubits, matrices_)
    initial_state = random_density_matrix(2**4, backend=backend)
    final_state = backend.apply_channel_density_matrix(
        channel, np.copy(initial_state), 4
    )

    eye = np.eye(2)
    ma1 = np.kron(np.kron(a1, eye), np.kron(eye, eye))
    ma2 = np.kron(np.kron(eye, eye), a2)
    ma1 = backend.cast(ma1, dtype=ma1.dtype)
    ma2 = backend.cast(ma2, dtype=ma2.dtype)
    target_state = (
        0.3 * initial_state
        + 0.4 * np.dot(ma1, np.dot(initial_state, ma1))
        + 0.3 * np.dot(ma2, np.dot(initial_state, ma2))
    )
    backend.assert_allclose(final_state, target_state)

    channel.to_choi(backend=backend)

    # checking old initialisation
    old = gates.UnitaryChannel(probabilities, list(zip(qubits, [a1, a2])))
    backend.assert_allclose(
        backend.calculate_norm(
            channel.to_choi(backend=backend) - old.to_choi(backend=backend)
        )
        < PRECISION_TOL,
        True,
    )


def test_unitary_channel_probability_tolerance():
    """Create ``UnitaryChannel`` with probability sum within tolerance (see #562)."""
    nqubits = 2
    param = 0.006
    num_terms = 2 ** (2 * nqubits)
    max_param = num_terms / (num_terms - 1)
    prob_identity = 1 - param / max_param
    prob_pauli = param / num_terms
    qubits = (0, 1)
    probs = [prob_identity] + [prob_pauli] * (num_terms - 1)
    probs = np.array(probs, dtype="float64")
    matrices_ = [(p, np.random.random((4, 4))) for p in probs]
    gates.UnitaryChannel(qubits, matrices_)


def test_unitary_channel_errors():
    """Check errors raised by ``gates.UnitaryChannel``."""
    a1 = matrices.X
    a2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    qubits = [(0,), (2, 3)]
    # Invalid ops
    with pytest.raises(TypeError):
        gates.UnitaryChannel(qubits, [a1, (0.1, a2)])
    # Invalid qubit length
    with pytest.raises(ValueError):
        gates.UnitaryChannel(qubits + [(0,)], [(0.4, a1), (0.3, a2)])
    # Probability > 1
    with pytest.raises(ValueError):
        gates.UnitaryChannel(qubits, [(0.4, a1), (1.1, a2)])
    # Probability sum > 1
    with pytest.raises(ValueError):
        gates.UnitaryChannel(qubits, [(0.5, a1), (0.6, a2)])


@pytest.mark.parametrize("pauli_order", ["IXYZ", "IZXY"])
def test_pauli_noise_channel(backend, pauli_order):
    initial_rho = random_density_matrix(2**2, backend=backend)
    qubits = (1,)
    channel = gates.PauliNoiseChannel(qubits, [("X", 0.3)])
    final_rho = backend.apply_channel_density_matrix(channel, np.copy(initial_rho), 2)
    gate = gates.X(1)
    target_rho = backend.apply_gate_density_matrix(gate, np.copy(initial_rho), 2)
    target_rho = 0.3 * target_rho + 0.7 * initial_rho
    backend.assert_allclose(final_rho, target_rho)

    basis = ["X", "Y", "Z"]
    pnp = np.array([0.1, 0.02, 0.05])
    noise_elements = {
        "I": 1,
        "X": 1 - 2 * pnp[1] - 2 * pnp[2],
        "Y": 1 - 2 * pnp[0] - 2 * pnp[2],
        "Z": 1 - 2 * pnp[0] - 2 * pnp[1],
    }
    test_representation = np.diag([noise_elements[p] for p in pauli_order])

    liouville = gates.PauliNoiseChannel(0, list(zip(basis, pnp))).to_pauli_liouville(
        normalize=True, pauli_order=pauli_order, backend=backend
    )
    norm = backend.calculate_norm(backend.to_numpy(liouville) - test_representation)

    assert norm < PRECISION_TOL


def test_depolarizing_channel_errors():
    with pytest.raises(ValueError):
        gates.DepolarizingChannel(0, 1.5)


def test_depolarizing_channel(backend):
    initial_rho = random_density_matrix(2**3, backend=backend)
    lam = 0.3
    initial_rho_r = backend.partial_trace_density_matrix(initial_rho, (2,), 3)
    channel = gates.DepolarizingChannel((0, 1), lam)
    final_rho = channel.apply_density_matrix(backend, np.copy(initial_rho), 3)
    final_rho_r = backend.partial_trace_density_matrix(final_rho, (2,), 3)
    target_rho_r = (1 - lam) * initial_rho_r + lam * backend.cast(np.identity(4)) / 4
    backend.assert_allclose(final_rho_r, target_rho_r)


@pytest.mark.parametrize(
    "t1,t2,time,excpop", [(0.8, 0.5, 1.0, 0.4), (0.5, 0.8, 1.0, 0.4)]
)
def test_thermal_relaxation_channel(backend, t1, t2, time, excpop):
    """Check ``gates.ThermalRelaxationChannel`` on a 3-qubit random density matrix."""
    initial_rho = random_density_matrix(2**3, backend=backend)
    gate = gates.ThermalRelaxationChannel(0, [t1, t2, time, excpop])
    final_rho = gate.apply_density_matrix(backend, np.copy(initial_rho), 3)

    if t2 > t1:
        p0, p1, exp = (
            gate.init_kwargs["p0"],
            gate.init_kwargs["p1"],
            gate.init_kwargs["e_t2"],
        )
        matrix = np.diag([1 - p1, p1, p0, 1 - p0])
        matrix[0, -1], matrix[-1, 0] = exp, exp
        matrix = matrix.reshape(4 * (2,))
        # Apply matrix using Eq. (3.28) from arXiv:1111.6950
        target_rho = np.copy(initial_rho).reshape(6 * (2,))
        target_rho = np.einsum("abcd,aJKcjk->bJKdjk", matrix, target_rho)
        target_rho = target_rho.reshape(initial_rho.shape)
    else:
        p0, p1, pz = (
            gate.init_kwargs["p0"],
            gate.init_kwargs["p1"],
            gate.init_kwargs["pz"],
        )
        mz = np.kron(np.array([[1, 0], [0, -1]]), np.eye(4))
        mz = backend.cast(mz, dtype=mz.dtype)
        z_rho = np.dot(mz, np.dot(initial_rho, mz))

        trace = backend.to_numpy(
            backend.partial_trace_density_matrix(initial_rho, (0,), 3)
        )
        trace = np.reshape(trace, 4 * (2,))
        zeros = np.tensordot(
            trace, np.array([[1, 0], [0, 0]], dtype=trace.dtype), axes=0
        )
        ones = np.tensordot(
            trace, np.array([[0, 0], [0, 1]], dtype=trace.dtype), axes=0
        )
        zeros = np.transpose(zeros, [4, 0, 1, 5, 2, 3])
        ones = np.transpose(ones, [4, 0, 1, 5, 2, 3])

        zeros = backend.cast(zeros, dtype=zeros.dtype)
        ones = backend.cast(ones, dtype=ones.dtype)

        pi = 1 - p0 - p1 - pz
        target_rho = pi * initial_rho + pz * z_rho
        target_rho += np.reshape(p0 * zeros + p1 * ones, initial_rho.shape)

    target_rho = backend.cast(target_rho, dtype=target_rho.dtype)

    backend.assert_allclose(
        backend.calculate_norm(final_rho - target_rho) < PRECISION_TOL, True
    )

    # checking old initialisation
    old = gates.ThermalRelaxationChannel(0, t1, t2, time, excpop)
    new = gates.ThermalRelaxationChannel(0, [t1, t2, time, excpop])
    backend.assert_allclose(
        backend.calculate_norm(
            new.to_choi(backend=backend) - old.to_choi(backend=backend)
        )
        < PRECISION_TOL,
        True,
    )


@pytest.mark.parametrize(
    "params",
    [
        [0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [1.0, 0.5, 1.5, 1.5],
        [1.0, 0.5, -0.5, 0.5],
        [1.0, -0.5, 1.5, 0.5],
        [-1.0, 0.5, 1.5, 0.5],
        [1.0, 3.0, 1.5, 0.5],
    ],
)
def test_thermal_relaxation_channel_errors(params):
    with pytest.raises(ValueError):
        gates.ThermalRelaxationChannel(0, params)


def test_readout_error_channel(backend):
    with pytest.raises(ValueError):
        gates.ReadoutErrorChannel(0, np.array([[1.1, 0], [0.5, 0.5]]))

    nqubits = 1
    d = 2**nqubits

    rho = random_density_matrix(d, seed=1, backend=backend)
    P = random_stochastic_matrix(d, seed=1, backend=backend)

    probability_sum = gates.ReadoutErrorChannel(0, P).apply_density_matrix(
        backend, rho, 1
    )
    probability_sum = np.diag(probability_sum).sum().real

    backend.assert_allclose(probability_sum - 1 < PRECISION_TOL, True)


def test_reset_channel(backend):
    initial_rho = random_density_matrix(2**3, backend=backend)
    gate = gates.ResetChannel(0, [0.2, 0.2])
    final_rho = backend.reset_error_density_matrix(gate, np.copy(initial_rho), 3)

    trace = backend.to_numpy(backend.partial_trace_density_matrix(initial_rho, (0,), 3))
    trace = np.reshape(trace, 4 * (2,))

    zeros = np.tensordot(trace, np.array([[1, 0], [0, 0]], dtype=trace.dtype), axes=0)
    ones = np.tensordot(trace, np.array([[0, 0], [0, 1]], dtype=trace.dtype), axes=0)
    zeros = np.transpose(zeros, [4, 0, 1, 5, 2, 3])
    ones = np.transpose(ones, [4, 0, 1, 5, 2, 3])
    zeros = backend.cast(zeros, dtype=zeros.dtype)
    ones = backend.cast(ones, dtype=ones.dtype)

    target_rho = 0.6 * initial_rho + 0.2 * np.reshape(zeros + ones, initial_rho.shape)

    backend.assert_allclose(final_rho, target_rho)

    old = gates.ResetChannel(0, 0.2, 0.2)
    new = gates.ResetChannel(0, [0.2, 0.2])
    backend.assert_allclose(
        backend.calculate_norm(
            new.to_choi(backend=backend) - old.to_choi(backend=backend)
        )
        < PRECISION_TOL,
        True,
    )


@pytest.mark.parametrize("p0,p1", [(0, -0.1), (-0.1, 0), (0.5, 0.6), (0.8, 0.3)])
def test_reset_channel_errors(p0, p1):
    with pytest.raises(ValueError):
        gates.ResetChannel(0, [p0])
    with pytest.raises(ValueError):
        gates.ResetChannel(0, [p0, p1])
